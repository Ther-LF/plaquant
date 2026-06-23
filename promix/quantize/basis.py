"""PCA basis computation for ResQ mixed-precision quantization.

Computes per-layer covariance matrices from calibration data, performs
eigendecomposition to find basis vectors ordered by variance, and saves:
  - U (basis) file: eigenvectors for channel reordering
  - E (eigenvalue) file: eigenvalues for analysis
  - R (rotation) file: initial random orthogonal rotation matrices

Usage:
    python -m promix.quantize.basis --config promix/configs/llama-3.2-1b-resq.yaml
"""

import os
import random

import datasets
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from promix.quantize.fuse_norm import fuse_layer_norms


def get_calibration_data(tokenizer, nsamples, seqlen, seed, calib_dataset="wikitext"):
    """Load calibration data as list of (input_ids, targets) tuples."""
    if "wikitext" in calib_dataset:
        traindata = datasets.load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1"
        )["train"]
        trainenc = tokenizer(
            text="\n\n".join(traindata["text"]), return_tensors="pt"
        )
    elif "c4" in calib_dataset:
        traindata = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        trainenc = tokenizer(
            text="\n\n".join(traindata["text"][:5000]), return_tensors="pt"
        )
    else:
        raise ValueError(f"Unknown calibration dataset: {calib_dataset}")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def random_orthogonal_matrix(size, device):
    """Generate a random orthogonal matrix via QR decomposition."""
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def eigen_decompose(cov_matrix, per_head=False, num_heads=0):
    """Eigendecompose covariance matrix, return (eigenvalues, eigenvectors) sorted ascending."""
    cov_matrix = cov_matrix.cuda()
    if per_head:
        assert num_heads != 0
        evals, evecs = [], []
        for hd in range(num_heads):
            H = cov_matrix[hd]
            damp = 0.01 * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[-1]).to(device=H.device)
            H[diag, diag] = H[diag, diag] + damp
            X = torch.linalg.eigh(H.to(torch.float64))
            index = torch.argsort(X[0])
            evals.append(X[0][index])
            evecs.append(X[1][:, index])
        return torch.stack(evals), torch.stack(evecs)
    else:
        H = cov_matrix
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1]).to(device=H.device)
        H[diag, diag] = H[diag, diag] + damp
        X = torch.linalg.eigh(H.to(torch.float64))
        index = torch.argsort(X[0])
        return X[0][index], X[1][:, index]


@torch.no_grad()
def compute_basis(
    model_name: str,
    output_dir: str,
    seqlen: int = 2048,
    nsamples: int = 512,
    seed: int = 0,
    calib_dataset: str = "wikitext",
    high_fraction: float = 0.125,
    low_fraction: float = 0.0,
    sparse_fraction: float = 0.0,
    down_proj_blocksize: int = 256,
    rotation_granularity: str = "full_shared",
    o_proj_pca: str = "per_head",
):
    """Compute PCA basis and initial rotation matrices.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save output files
        seqlen: Sequence length for calibration
        nsamples: Number of calibration samples
        seed: Random seed
        calib_dataset: Calibration dataset name
        high_fraction: Fraction of channels for high-precision
        low_fraction: Fraction of channels for low-precision (0 = no low group)
        sparse_fraction: Fraction for sparsity (0 = no sparsity)
        down_proj_blocksize: Block size for down_proj covariance
        rotation_granularity: "full_shared", "per_layer", or "one_per_decoder"
        o_proj_pca: "per_head" (default, original ResQ behavior) or "full_global"
            (hidden_dim x hidden_dim PCA on the o_proj input instead of
            per-head head_dim x head_dim block-diagonal). When "full_global",
            an additional H_oproj covariance is collected from o_proj input
            activations and stored under basis_dict[f"layer.{{i}}.self_attn.o_proj_global"].
            The original per-head H_value path is also kept for backward
            compat; downstream rotation.py picks up the o_proj_global key
            when present.

    Returns:
        Tuple of (basis_path, rotation_path, eval_path)
    """
    print(f"Computing PCA basis for {model_name}")
    print(f"  seqlen={seqlen}, nsamples={nsamples}, calib={calib_dataset}")
    print(f"  high_fraction={high_fraction}, low_fraction={low_fraction}")
    print(f"  rotation_granularity={rotation_granularity}")
    print()

    # Determine output paths
    short_name = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)

    basis_path = os.path.join(
        output_dir,
        f"U-{calib_dataset}-{nsamples}-{short_name}.bin",
    )
    eval_path = os.path.join(
        output_dir,
        f"E-{calib_dataset}-{nsamples}-{short_name}.bin",
    )
    rotation_path = os.path.join(
        output_dir,
        f"R-high-{high_fraction}-low-{low_fraction}-sparse-{sparse_fraction}-{short_name}.bin",
    )

    # Load model in FP32
    print("Loading model in FP32...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.seqlen = seqlen
    transformers.set_seed(seed)
    model.eval()

    # Fuse layer norms into adjacent linears
    print("Fusing layer norms...")
    fuse_layer_norms(model)
    torch.cuda.empty_cache()

    # Load tokenizer and calibration data
    print("Loading calibration data...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=seqlen,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    train_data = get_calibration_data(tokenizer, nsamples, seqlen, seed, calib_dataset)
    nbatches = len(train_data)

    # Extract model dimensions
    hidden_dim = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    head_dim = hidden_dim // num_attention_heads
    kv_heads = model.config.num_key_value_heads
    layers = model.model.layers
    nlayers = len(layers)

    high_length_hidden = int(high_fraction * hidden_dim)
    low_length_hidden = int(low_fraction * hidden_dim)
    high_length_head = int(high_fraction * head_dim)
    low_length_head = int(low_fraction * head_dim)
    sparse_length_hidden = int(sparse_fraction * low_length_hidden)
    sparse_length_head = int(sparse_fraction * low_length_head)

    # Generate initial random rotation matrices and save
    if not os.path.exists(rotation_path):
        print("Generating initial random rotation matrices...")
        rotation_dict = {}
        R1_1 = random_orthogonal_matrix(
            hidden_dim - high_length_hidden - low_length_hidden - sparse_length_hidden, "cuda"
        )
        rotation_dict["R1_1"] = R1_1
        rotation_dict["R1_2"] = random_orthogonal_matrix(high_length_hidden, "cuda")

        if low_length_hidden != 0:
            R1_0 = random_orthogonal_matrix(low_length_hidden - sparse_length_hidden, "cuda")
            if sparse_length_hidden > 0:
                zeros = torch.zeros(
                    (sparse_length_hidden, sparse_length_hidden),
                    device=R1_1.device, dtype=R1_1.dtype,
                )
                R1_0 = torch.block_diag(zeros, R1_0)
            rotation_dict["R1_0"] = R1_0
        else:
            rotation_dict["R1_0"] = None

        R2_1 = random_orthogonal_matrix(
            head_dim - high_length_head - low_length_head, "cuda"
        )
        rotation_dict["R2_1"] = R2_1
        rotation_dict["R2_2"] = random_orthogonal_matrix(high_length_head, "cuda")

        if low_length_head != 0:
            R2_0 = random_orthogonal_matrix(low_length_head - sparse_length_head, "cuda")
            if sparse_length_hidden > 0:
                zeros = torch.zeros(
                    (sparse_length_head, sparse_length_head),
                    device=R2_1.device, dtype=R2_1.dtype,
                )
                R2_0 = torch.block_diag(zeros, R2_0)
            rotation_dict["R2_0"] = R2_0
        else:
            rotation_dict["R2_0"] = None

        torch.save(rotation_dict, rotation_path)
        print(f"  Saved rotation: {rotation_path}")
    else:
        print(f"  Rotation already exists: {rotation_path}")

    # Check if basis already exists
    if os.path.exists(basis_path):
        print(f"  Basis already exists: {basis_path}")
        return basis_path, rotation_path, eval_path

    # Collect activations layer by layer
    print("Collecting activations and computing covariance matrices...")

    # Move embedding to GPU and capture first-layer inputs
    model.model.embed_tokens = model.model.embed_tokens.cuda()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cuda()
    layers[0] = layers[0].cuda()

    inps = [0] * nbatches
    cache = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            if "position_embeddings" in kwargs:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            else:
                cache["position_embeddings"] = None
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nbatches):
        batch = train_data[i][0].cuda()
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    attention_mask = cache["attention_mask"]
    torch.cuda.empty_cache()

    outs = [0] * nbatches

    # Initialize covariance matrices
    H_attn = torch.zeros((nlayers, hidden_dim, hidden_dim))
    H_mlp = torch.zeros((nlayers, hidden_dim, hidden_dim))
    H_down_proj = torch.zeros((nlayers, down_proj_blocksize, down_proj_blocksize))
    H_value = torch.zeros((nlayers, kv_heads, head_dim, head_dim))
    H_key_pos = torch.zeros((nlayers, kv_heads, head_dim, head_dim))

    # When o_proj_pca == "full_global", collect a hidden_dim x hidden_dim
    # covariance directly from o_proj input (the post-attention concatenated
    # vector). This replaces the per-head head_dim x head_dim PCA on V with a
    # global PCA on the actual o_proj input and aligns o_proj with the
    # q/k/v/gate/up code path (which already uses hidden_dim PCA).
    use_oproj_global = o_proj_pca == "full_global"
    if use_oproj_global:
        H_oproj = torch.zeros((nlayers, hidden_dim, hidden_dim))
    else:
        H_oproj = None

    for i in tqdm(range(nlayers), desc="Collecting covariance"):
        layer = layers[i].cuda()

        # Hook state (use list to avoid closure issues with globals)
        hook_state = {}

        def hook_fn_qproj(module, input, output):
            hook_state["input_qkv"] = input[0]
            hook_state["output_qproj"] = output

        def hook_fn_kproj(module, input, output):
            hook_state["output_kproj"] = output

        def hook_fn_vproj(module, input, output):
            hook_state["output_vproj"] = output

        def hook_fn_upproj(module, input, output):
            hook_state["input_up"] = input[0]

        def hook_fn_downproj(module, input, output):
            hook_state["input_down"] = input[0]

        def hook_fn_oproj(module, input, output):
            hook_state["input_oproj"] = input[0]

        hooks = [
            layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj),
            layer.self_attn.k_proj.register_forward_hook(hook_fn_kproj),
            layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj),
            layer.mlp.up_proj.register_forward_hook(hook_fn_upproj),
            layer.mlp.down_proj.register_forward_hook(hook_fn_downproj),
        ]
        if use_oproj_global:
            hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_fn_oproj))

        for j in range(nbatches):
            outs[j] = layer(
                inps[j],
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

            input_qkv = hook_state["input_qkv"]
            input_up = hook_state["input_up"]
            input_down = hook_state["input_down"]
            output_vproj = hook_state["output_vproj"]
            output_kproj = hook_state["output_kproj"]

            # Reshape value/key outputs to per-head
            value_states = output_vproj.view(1, seqlen, kv_heads, head_dim).transpose(1, 2)

            key_states = output_kproj.view(1, seqlen, kv_heads, head_dim).transpose(1, 2)
            query_states = hook_state["output_qproj"].view(
                1, seqlen, num_attention_heads, head_dim
            ).transpose(1, 2)

            # Apply RoPE
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
            _, key_states_pos = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Accumulate covariance (in double precision)
            H_mlp[i] += torch.sum(
                input_up.double().mT @ input_up.double(), dim=0
            ).cpu()
            H_attn[i] += torch.sum(
                input_qkv.double().mT @ input_qkv.double(), dim=0
            ).cpu()
            H_value[i] += torch.sum(
                value_states.double().mT @ value_states.double(), dim=0
            ).cpu()
            H_key_pos[i] += torch.sum(
                key_states_pos.double().mT @ key_states_pos.double(), dim=0
            ).cpu()
            H_down_proj[i] += torch.sum(
                input_down.view(input_down.shape[0], -1, down_proj_blocksize)
                .double().mT
                @ input_down.view(input_down.shape[0], -1, down_proj_blocksize).double(),
                dim=0,
            ).cpu()

            if use_oproj_global:
                input_oproj = hook_state["input_oproj"]
                # input_oproj has shape (batch, seqlen, hidden_dim); accumulate
                # the full hidden_dim x hidden_dim covariance (no per-head split).
                H_oproj[i] += torch.sum(
                    input_oproj.double().mT @ input_oproj.double(), dim=0
                ).cpu()

        for hook in hooks:
            hook.remove()
        layers[i] = layers[i].cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # Eigendecomposition based on rotation_granularity
    print("Computing eigendecomposition...")
    basis_dict = {}
    eval_dict = {}

    if "full_shared" in rotation_granularity.lower():
        # Shared basis for attn+mlp across all layers
        eval_attn_mlp, evec_attn_mlp = eigen_decompose(
            (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nbatches * nlayers * seqlen)
        )
        basis_dict["config"] = "full_shared_rotation"
        basis_dict["attn_mlp"] = evec_attn_mlp.cpu()
        eval_dict["config"] = "full_shared_rotation"
        eval_dict["attn_mlp"] = eval_attn_mlp.cpu()

        for i in tqdm(range(nlayers), desc="Computing per-layer basis"):
            eval_down, evec_down = eigen_decompose(
                H_down_proj[i] / (nbatches * seqlen)
            )
            eval_value, evec_value = eigen_decompose(
                H_value[i] / (seqlen * nbatches), per_head=True, num_heads=kv_heads
            )
            eval_k_pos, evec_k_pos = eigen_decompose(
                H_key_pos[i].sum(0) / (kv_heads * nbatches * seqlen)
            )

            basis_dict[f"layer.{i}.self_attn.value"] = evec_value.cpu()
            basis_dict[f"layer.{i}.self_attn.key_pos"] = evec_k_pos.cpu()
            basis_dict[f"layer.{i}.mlp.down_proj"] = evec_down.cpu()

            eval_dict[f"layer.{i}.self_attn.value"] = eval_value.cpu()
            eval_dict[f"layer.{i}.self_attn.key_pos"] = eval_k_pos.cpu()
            eval_dict[f"layer.{i}.mlp.down_proj"] = eval_down.cpu()

            if use_oproj_global:
                # Hidden_dim x hidden_dim PCA on o_proj input. Stored under a
                # NEW key (o_proj_global) so the legacy "value" key remains
                # available for downstream code that still expects per-head.
                # rotation.py / quant pipeline must opt into reading
                # o_proj_global; until then, this is an additional artifact.
                eval_oproj_g, evec_oproj_g = eigen_decompose(
                    H_oproj[i] / (nbatches * seqlen)
                )
                basis_dict[f"layer.{i}.self_attn.o_proj_global"] = evec_oproj_g.cpu()
                eval_dict[f"layer.{i}.self_attn.o_proj_global"] = eval_oproj_g.cpu()

    elif "per_layer" in rotation_granularity.lower():
        basis_dict["config"] = "per_layer_rotation"
        for i in tqdm(range(nlayers), desc="Computing per-layer basis"):
            eval_attn, evec_attn = eigen_decompose(H_attn[i] / (nbatches * seqlen))
            eval_mlp, evec_mlp = eigen_decompose(H_mlp[i] / (nbatches * seqlen))
            eval_down, evec_down = eigen_decompose(H_down_proj[i] / (nbatches * seqlen))
            eval_value, evec_value = eigen_decompose(
                H_value[i] / (seqlen * nbatches), per_head=True, num_heads=kv_heads
            )
            eval_k_pos, evec_k_pos = eigen_decompose(
                H_key_pos[i] / (seqlen * nbatches), per_head=True, num_heads=kv_heads
            )

            basis_dict[f"layer.{i}.mlp"] = evec_mlp.cpu()
            basis_dict[f"layer.{i}.mlp.down_proj"] = evec_down.cpu()
            basis_dict[f"layer.{i}.self_attn"] = evec_attn.cpu()
            basis_dict[f"layer.{i}.self_attn.value"] = evec_value.cpu()
            basis_dict[f"layer.{i}.self_attn.key_pos"] = evec_k_pos.cpu()

    elif "one_per_decoder" in rotation_granularity.lower():
        basis_dict["config"] = "one_per_decoder"
        for i in tqdm(range(nlayers), desc="Computing per-decoder basis"):
            eval_attn_mlp, evec_attn_mlp = eigen_decompose(
                (H_attn[i] + H_mlp[i]) / (2 * nbatches * seqlen)
            )
            eval_down, evec_down = eigen_decompose(H_down_proj[i] / (nbatches * seqlen))
            eval_value, evec_value = eigen_decompose(
                H_value[i] / (seqlen * nbatches), per_head=True, num_heads=kv_heads
            )
            eval_k_pos, evec_k_pos = eigen_decompose(
                H_key_pos[i] / (seqlen * nbatches), per_head=True, num_heads=kv_heads
            )

            basis_dict[f"layer.{i}.mlp.down_proj"] = evec_down.cpu()
            basis_dict[f"layer.{i}.self_attn_mlp"] = evec_attn_mlp.cpu()
            basis_dict[f"layer.{i}.self_attn.value"] = evec_value.cpu()
            basis_dict[f"layer.{i}.self_attn.key_pos"] = evec_k_pos.cpu()
    else:
        raise ValueError(f"Unknown rotation_granularity: {rotation_granularity}")

    # Save
    torch.save(basis_dict, basis_path)
    torch.save(eval_dict, eval_path)
    print(f"\nSaved basis: {basis_path}")
    print(f"Saved eigenvalues: {eval_path}")

    return basis_path, rotation_path, eval_path


def main():
    """CLI entry point — parse config and run basis computation."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compute PCA basis for ResQ quantization")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--nsamples", type=int, default=None, help="Override nsamples")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config["paths"].get("basis_output_dir", "./rotation")
    nsamples = args.nsamples or config["calibration"]["nsamples"]

    basis_path, rotation_path, eval_path = compute_basis(
        model_name=config["model"]["name"],
        output_dir=output_dir,
        seqlen=config["calibration"]["seqlen"],
        nsamples=nsamples,
        seed=config["calibration"].get("seed", 0),
        calib_dataset=config["calibration"]["dataset"],
        high_fraction=config["quantize"]["high_fraction"],
        low_fraction=config["quantize"].get("low_fraction", 0.0),
        sparse_fraction=config["quantize"].get("sparse_fraction", 0.0),
        down_proj_blocksize=config["quantize"].get("down_proj_blocksize", 256),
        rotation_granularity=config["quantize"]["rotation_granularity"],
        o_proj_pca=config["quantize"].get("o_proj_pca", "per_head"),
    )

    print(f"\nDone! Files saved:")
    print(f"  Basis: {basis_path}")
    print(f"  Rotation: {rotation_path}")
    print(f"  Eigenvalues: {eval_path}")


if __name__ == "__main__":
    main()

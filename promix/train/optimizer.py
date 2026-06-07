"""SGDG optimizer — Stiefel manifold SGD via Cayley transform.

Optimizes orthogonal matrices while maintaining orthogonality constraint.
For parameters marked with stiefel=True, uses Cayley transform for
updates on the Stiefel manifold. Other parameters use standard SGD.

From: https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform
"""

import random

import torch
from torch.optim.optimizer import Optimizer


def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim=1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def Cayley_loop(X, W, tan_vec, t):
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()


def qr_retraction(tan_vec):
    [p, n] = tan_vec.size()
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()
    return q


epsilon = 1e-8


class SGDG(Optimizer):
    """SGD with Stiefel manifold support via Cayley transform.

    For parameter groups with stiefel=True, maintains orthogonality
    by projecting gradients onto the tangent space and using Cayley
    transform for retraction.
    """

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0,
                 nesterov=False, stiefel=False, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                unity, _ = unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)
                    lr = group["lr"]

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(g.t().size())
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state["momentum_buffer"].cuda()

                    V = param_state["momentum_buffer"]
                    V = momentum * V - g.t()
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (matrix_norm_one(W) + epsilon)
                    alpha = min(t, lr)

                    p_new = Cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t())
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)
                else:
                    d_p = p.grad.data
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p, alpha=1 - group["dampening"])
                        if group["nesterov"]:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    p.data.add_(d_p, alpha=-group["lr"])

        return loss

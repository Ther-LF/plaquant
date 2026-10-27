import sys, os, time, torch, transformers

def random_orthogonal_matrix(size, device):
    if size == 0:
        return torch.zeros(0,0, dtype=torch.float64, device=device)
    m = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(m)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def gen(hidden, head_dim, r, outpath, seed=0):
    transformers.set_seed(seed)
    hh = int(r*hidden); hd = int(r*head_dim)
    d = {}
    d['R1_1'] = random_orthogonal_matrix(hidden-hh, 'cuda')
    d['R1_2'] = random_orthogonal_matrix(hh, 'cuda')
    d['R1_0'] = None
    d['R2_1'] = random_orthogonal_matrix(head_dim-hd, 'cuda')
    d['R2_2'] = random_orthogonal_matrix(hd, 'cuda')
    d['R2_0'] = None
    torch.save(d, outpath)
    print('WROTE', outpath, 'R1_1',tuple(d['R1_1'].shape),'R1_2',tuple(d['R1_2'].shape),'R2_1',tuple(d['R2_1'].shape),'R2_2',tuple(d['R2_2'].shape))

short=sys.argv[1]; hidden=int(sys.argv[2]); head_dim=int(sys.argv[3])
rs=[float(x) for x in sys.argv[4].split(',')]
outdir='/home/luofan/plaquant/rotation'
for r in rs:
    p=os.path.join(outdir, f'R-high-{r}-low-0.0-sparse-0.0-{short}.bin')
    if os.path.exists(p):
        print('EXISTS', p); continue
    t=time.time(); gen(hidden, head_dim, r, p); print('  %.2fs'%(time.time()-t))

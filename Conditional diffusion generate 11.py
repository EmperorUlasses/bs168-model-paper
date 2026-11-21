# -*- coding: utf-8 -*-
"""
Conditional‑diffusion‑based Python script for controllable promoter DNA sequence generation.
Supports original intensity classes together with upward and downward intensity extrapolation,
and exports generated DNA sequences to a CSV file.
"""


import os
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from tqdm import tqdm


NUCLEOTIDES = ['A', 'C', 'T', 'G']
IMAGE_SIZE = 96
CHANNELS = 1
TIMESTEPS = 50

TOTAL_CLASS_NUMBER = 9      
N_FRAMES = 7                

N_CLASSES_OUT = 11          
N_PER_CLASS = 1200          
N_SAMPLE_BATCH = 64         
COND_WEIGHT = 2.0           
TEMPERATURE = 0.8           

CKPT = 'Conditional diffusion/diffusion_final_model.pth'
OUT_CSV = 'Conditional diffusion/all_generated_sequences_11class.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"


def default(val, d):
    return val if val is not None else (d() if callable(d) else d)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(self.half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) \
            if time_emb_dim is not None else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        return self.block2(self.block1(x, scale_shift=scale_shift)) + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        return self.to_out(rearrange(out, 'b h c (x y) -> b (h c) x y',
                                     h=self.heads, x=h, y=w))


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        return self.to_out(rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w))


class Unet(nn.Module):
    def __init__(self, dim, init_dim=IMAGE_SIZE, dim_mults=(1, 2, 4), channels=CHANNELS,
                 resnet_block_groups=8, learned_sinusoidal_dim=18,
                 num_classes=TOTAL_CLASS_NUMBER, num_frames=N_FRAMES):
        super().__init__()
        self.channels = channels
        init_dim = init_dim or dim
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(learned_sinusoidal_dim),
            nn.Linear(learned_sinusoidal_dim + 1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim))
        self.label_emb = nn.Embedding(num_classes, time_dim)
        self.frame_emb = nn.Embedding(num_frames, time_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim,
                                      groups=resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim,
                                      groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim,
                                           groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x, time, frames, class_vec):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) + class_vec + self.frame_emb(frames)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


def linear_beta_schedule(timesteps, beta_end=0.2):
    beta_start = 0.0001
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas_lin = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas_lin, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas_lin)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def get_class_vector(model, class_idx, alpha):
    W = model.label_emb.weight.detach()
    if 0 <= class_idx < TOTAL_CLASS_NUMBER:
        return W[class_idx].clone()
    elif class_idx == 9:
        e_hi, e_lo = W[8], W[7]
        return e_hi + alpha * (e_hi - e_lo)
    elif class_idx == -1:
        e_hi, e_lo = W[1], W[0]
        return e_lo - alpha * (e_hi - e_lo)
    else:
        raise ValueError(f"unsupported class_idx {class_idx}")


def get_null_vector(model):
    return model.label_emb.weight.detach()[0].clone()


@torch.no_grad()
def p_sample_guided(model, x, frames, class_vec, t, t_index, cond_weight=COND_WEIGHT):
    batch_size = x.shape[0]
    t_double = t.repeat(2)
    x_double = x.repeat(2, 1, 1, 1)

    betas_t = extract(betas, t_double, x_double.shape)
    sqrt_one_minus_t = extract(sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_double, x_double.shape)

    preds = model(x_double, time=t_double, frames=frames.repeat(2), class_vec=class_vec)

    eps = (1 + cond_weight) * preds[:batch_size] - cond_weight * preds[batch_size:]

    model_mean = sqrt_recip_alphas_t[:batch_size] * (
        x - betas_t[:batch_size] * eps / sqrt_one_minus_t[:batch_size])

    if t_index == 0:
        return model_mean
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)


@torch.no_grad()
def sample_class(model, class_idx, frame_idx, batch_size, alpha, cond_weight=COND_WEIGHT):
    e_pos = get_class_vector(model, class_idx, alpha)
    e_neg = get_null_vector(model)
    class_vec = torch.stack([e_pos] * batch_size + [e_neg] * batch_size, dim=0)

    shape = (batch_size, CHANNELS, 4, IMAGE_SIZE)
    img = torch.randn(shape, device=device)
    frames = torch.full((batch_size,), frame_idx, dtype=torch.long, device=device)

    if class_idx == -1:
        desc_text = f'  c0(ext)-F{frame_idx}'
    elif class_idx ==9:
        desc_text = f'  c10(ext)-F{frame_idx}'
    else:
        desc_text = f'  c{class_idx+1}-F{frame_idx}'
    for i in tqdm(reversed(range(0, TIMESTEPS)), desc=desc_text,
        total=TIMESTEPS, leave=False):
        img = p_sample_guided(model, x=img, frames=frames, class_vec=class_vec,
                              t=torch.full((batch_size,), i, device=device, dtype=torch.long),
                              t_index=i, cond_weight=cond_weight)
    return img.cpu().numpy()


FRAME_COORDS = {
    0: {'35': 54, '10': 31, 'sd': 14},
    1: {'35': 64, '10': 41, 'sd': 14},
    2: {'35': 74, '10': 51, 'sd': 14},
    3: {'35': 51, '10': 28, 'sd': 11},
    4: {'35': 61, '10': 38, 'sd': 11},
    5: {'35': 71, '10': 48, 'sd': 11},
}

P35 = "TTGACA"
P10 = "TATAAT"
SD  = "AGGAGG"


def protected_positions(frame_idx):
    coords = FRAME_COORDS[frame_idx]
    pos = set()
    for key in ['35', '10', 'sd']:
        start = 95 - coords[key]
        for offset in range(6):
            pos.add(start + offset)
    return pos


def decode_images(images, frame_idx, temperature=TEMPERATURE, force_fixed_frame=False):
    prot = protected_positions(frame_idx)
    coords = FRAME_COORDS[frame_idx]

    pos_35 = 95 - coords['35']
    pos_10 = 95 - coords['10']
    pos_sd = 95 - coords['sd']

    out = []
    for image in images:
        if image.ndim == 3:
            image = image.squeeze(0)
        img95 = image[:, 1:]          
        chars = []
        for col in range(95):
            probs = img95[:, col].copy()
            if col in prot:
                idx = int(np.argmax(probs))       
            else:
                probs = probs / temperature
                probs = probs - np.max(probs)
                exp_probs = np.exp(probs)
                probs = exp_probs / np.sum(exp_probs)
                idx = int(np.random.choice(len(probs), p=probs))
            chars.append(NUCLEOTIDES[idx])
        seq = ''.join(chars)

        if force_fixed_frame:
            lst = list(seq)
            lst[pos_35:pos_35+6] = list(P35)
            lst[pos_10:pos_10+6] = list(P10)
            lst[pos_sd:pos_sd+6] = list(SD)
            seq = ''.join(lst)
        out.append(seq)
    return out


def generate(alpha=1.0, n_per_class=N_PER_CLASS, ckpt=CKPT, out_csv=OUT_CSV):
    model = Unet(dim=IMAGE_SIZE, channels=CHANNELS, dim_mults=(1, 2, 4),
                 resnet_block_groups=4).to(device)
    try:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    W = model.label_emb.weight.detach()
    step_up = (W[8] - W[7]).norm().item()
    step_down = (W[1] - W[0]).norm().item()
    print("=" * 70)
    print(f"权重        : {ckpt}")
    print(f"档位        : {N_CLASSES_OUT} 类；‑1向下外推，0‑8原始9档，9向上外推")
    print(f"外推系数    : alpha = {alpha}（上下外推共用同一个alpha）")
    print(f"向上外推步长||w8‑w7|| = {step_up:.4f}；向下外推步长||w1‑w0||={step_down:.4f}")
    print(f"CFG 权重    : {COND_WEIGHT}    解码温度: {TEMPERATURE}")
    print("说明        : ‑1、0、8、9档=2400条；其余档位=1200条")
    print("=" * 70)


    generated = []
    total_plan = 0
    for class_idx in range(-1, N_CLASSES_OUT-1):
        if class_idx in (-1, 0, 8, 9):
            n_this_class = 1200
        else:
            n_this_class = 1200

        total_plan += n_this_class

        if class_idx == -1:
            tag = f"强度档 {class_idx}（向下外推）"
        elif class_idx ==9:
            tag = f"强度档 {class_idx}（向上外推）"
        else:
            tag = f"强度档 {class_idx}"

        print(f"\n正在生成 {tag} ，本档目标 {n_this_class} 条 ...")

        n_per_frame = max(1, n_this_class // 6)
        for frame_idx in range(6):
            done = 0
            while done < n_per_frame:
                bs = min(N_SAMPLE_BATCH, n_per_frame - done)
                imgs = sample_class(model, class_idx, frame_idx, bs, alpha)
                force_fixed = (class_idx ==9) or (class_idx == -1)

                for seq in decode_images(imgs, frame_idx, temperature=TEMPERATURE, force_fixed_frame=force_fixed):
                    generated.append({'class': class_idx, 'frame': frame_idx, 'sequence': seq})
                done += bs

    print(f"\n计划总条数: {total_plan}，实际共生成 {len(generated)} 条序列。")
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(generated).sample(frac=1).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"已保存至 {out_csv}")

    meta_path = os.path.join(out_dir if out_dir else '.', 'generation_meta.txt')
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"alpha={alpha}\n")
        f.write(f"spec: class‑1(down‑ext)=2400, class0=2400, class8=2400, class9(up‑ext)=2400, others=1200\n")

        f.write(f"plan_total={total_plan}\nreal_total={len(generated)}\n")
        f.write(f"cond_weight={COND_WEIGHT}\n")
        f.write(f"temperature={TEMPERATURE}\nckpt={ckpt}\n")
    print(f"参数记录已存至 {meta_path}")



def main():
    ap = argparse.ArgumentParser(description='11档版启动子生成：‑1向下外推 +原始0‑8 +9向上外推')

    ap.add_argument('--alpha', type=float, default=1.0,
                    help='上下外推共用系数；alpha=0等于原档位，建议0.25/0.5/1/2')

    ap.add_argument('--n-per-class', type=int, default=N_PER_CLASS,
                    help='每档生成条数，默认 1200')
    ap.add_argument('--ckpt', type=str, default=CKPT)
    ap.add_argument('--out', type=str, default=OUT_CSV)
    args = ap.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f"device = {device}")
    generate(alpha=args.alpha, n_per_class=args.n_per_class,
             ckpt=args.ckpt, out_csv=args.out)


if __name__ == '__main__':
    main()

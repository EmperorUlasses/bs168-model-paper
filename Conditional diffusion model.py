"""
Training script for dual‑condition‑guided DNA conditional diffusion model based on UNet‑Diffusion, 
used to generate 95 bp synthetic promoter sequences.
"""
import os
os.getpid()
from functools import partial
from scipy.stats import zscore
import torch
import copy
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Adam
import matplotlib
import math
from functools import partial
import scipy
from scipy.special import rel_entr
from torch import nn, einsum
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
import gc
from sklearn.model_selection import train_test_split

# Global seed
GLOBAL_SEED = 42
NUCLEOTIDES = ['A', 'C', 'T', 'G']

EPOCHS = 5000
EPOCHS_LOSS_SHOW = 5
IMAGE_SIZE = 96
CHANNELS = 1
LEARNING_RATE = 1e-4
TIMESTEPS = 50
RESNET_BLOCK_GROUPS = 4
BATCH_SIZE = 48

TOTAL_CLASS_NUMBER = 9      
N_FRAMES = 7                 

def seed_everything(seed=GLOBAL_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

def one_hot_encode(seq, nucleotides, max_seq_len, pad_side='left'):
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(nucleotides)))
    if pad_side == 'right':
        for i in range(seq_len):
            seq_array[i, nucleotides.index(seq[i])] = 1
    else:
        start_idx = max_seq_len - seq_len
        for i in range(seq_len):
            seq_array[start_idx + i, nucleotides.index(seq[i])] = 1
    return seq_array

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None: return new
        return old * self.beta + (1 - self.beta) * new
    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def linear_beta_schedule(timesteps, beta_end=0.2):
    beta_start = 0.0001
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None: noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, classes, frames, noise=None, loss_type="huber", p_uncond=0.1):
    device = x_start.device
    if noise is None: noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)
    classes = (classes * context_mask).type(torch.long)
    frames = (frames * context_mask).type(torch.long)
    
    predicted_noise = denoise_model(x_noisy, t, classes, frames)
    
    if loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    elif loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    else:
        loss = F.mse_loss(noise, predicted_noise)
    return loss


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs): return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(dim, dim_out or dim, 3, padding=1))

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, dim_out or dim, 4, 2, 1)

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
    def forward(self, x): return self.fn(self.norm(x))

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
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
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
        self.scale = dim_head**-0.5
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
        return self.to_out(rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w))

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
    def __init__(self, dim, init_dim=IMAGE_SIZE, dim_mults=(1, 2, 4), channels=CHANNELS, resnet_block_groups=8, learned_sinusoidal_dim=18, num_classes=TOTAL_CLASS_NUMBER, num_frames=N_FRAMES):
        super().__init__()
        self.channels = channels
        init_dim = init_dim or dim
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(LearnedSinusoidalPosEmb(learned_sinusoidal_dim), nn.Linear(learned_sinusoidal_dim + 1, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
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
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x, time, classes, frames):
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time) + self.label_emb(classes) + self.frame_emb(frames)
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


class SequenceDataset(Dataset):
    def __init__(self, seqs, c_bin, c_frame, transform=None):
        self.seqs = seqs
        self.c_bin = c_bin
        self.c_frame = c_frame
        self.transform = transform
    def __len__(self): return len(self.seqs)
    def __getitem__(self, index):
        return self.transform(self.seqs[index]), self.c_bin[index], self.c_frame[index]

class DataProcessor:
    def __init__(self, csv_file, test_size=0.2, shuffle_size=0.2, random_state=42):
        self.csv_file = csv_file
        self.test_size = test_size
        self.shuffle_size = shuffle_size
        self.random_state = random_state
        self.train_df, self.test_df, self.shuffle_df = self.process_data()
    def process_data(self):
        df = pd.read_csv(self.csv_file)
        df = df[['raw_sequence', 'component', 'frame']]
        df.columns = ['seq', 'label', 'frame']
        df['label'] = df['label'] - 1
        frame_map = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4, 'F5': 5, 'F_none': 6}
        df['frame'] = df['frame'].map(frame_map)
        
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        train_df, shuffle_df = train_test_split(train_df, test_size=self.shuffle_size, random_state=self.random_state)
        return train_df, test_df, shuffle_df

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(dim=IMAGE_SIZE, channels=CHANNELS, dim_mults=(1, 2, 4), resnet_block_groups=RESNET_BLOCK_GROUPS).to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

DNA_DataProcessor = DataProcessor(csv_file='data/95bp_数据集合并_分箱结果.csv')

valid_mask = [('N' not in x) for x in DNA_DataProcessor.train_df['seq']]
seqs_filtered = DNA_DataProcessor.train_df['seq'][valid_mask]
X_train = np.array([one_hot_encode(x, NUCLEOTIDES, 96, pad_side='left').T.tolist() for x in tqdm(seqs_filtered)])
X_train[X_train == 0] = -1

x_train_cell_type = torch.from_numpy(DNA_DataProcessor.train_df['label'].to_numpy()[valid_mask])
x_train_frame_type = torch.from_numpy(DNA_DataProcessor.train_df['frame'].to_numpy()[valid_mask])

train_dl = DataLoader(SequenceDataset(X_train, x_train_cell_type, x_train_frame_type, transform=T.Compose([T.ToTensor()])), 
                      BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

val_mask = [('N' not in x) for x in DNA_DataProcessor.shuffle_df['seq']]
val_seqs = DNA_DataProcessor.shuffle_df['seq'][val_mask]
X_val = np.array([one_hot_encode(x, NUCLEOTIDES, 96, pad_side='left').T.tolist() for x in tqdm(val_seqs)])
X_val[X_val == 0] = -1

y_val_bin = torch.from_numpy(DNA_DataProcessor.shuffle_df['label'].to_numpy()[val_mask])
y_val_frame = torch.from_numpy(DNA_DataProcessor.shuffle_df['frame'].to_numpy()[val_mask])

val_dl = DataLoader(SequenceDataset(X_val, y_val_bin, y_val_frame, transform=T.Compose([T.ToTensor()])),
                    BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

gc.collect()
torch.cuda.empty_cache()
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

generated_sequences = []
os.makedirs('Conditional diffusion', exist_ok=True)
best_val_loss = float('inf')

for epoch in tqdm(range(EPOCHS), desc='Training'):
    model.train()
    for step, batch in enumerate(train_dl):
        x, y_bin, y_frame = batch
        x = x.type(torch.float32).to(device)
        y_bin = y_bin.type(torch.long).to(device)
        y_frame = y_frame.type(torch.long).to(device)
        batch_size = x.shape[0]

        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
        loss = p_losses(model, x, t, y_bin, y_frame, loss_type="huber")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                x_val, y_val_bin_b, y_val_frame_b = batch
                x_val = x_val.type(torch.float32).to(device)
                y_val_bin_b = y_val_bin_b.type(torch.long).to(device)
                y_val_frame_b = y_val_frame_b.type(torch.long).to(device)
                t_val = torch.randint(0, TIMESTEPS, (x_val.shape[0],), device=device).long()
                val_loss = p_losses(model, x_val, t_val, y_val_bin_b, y_val_frame_b,
                                    loss_type="huber", p_uncond=0.0)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'Conditional diffusion/diffusion_best_model.pth')
            print(f"  [Best] val_loss improved to {best_val_loss:.4f}, model saved.")

        model.train()
        print(f" Epoch {epoch + 1} | train_loss: {loss.item():.4f} | val_loss: {val_loss:.4f}")
    elif (epoch % EPOCHS_LOSS_SHOW) == 0:
        print(f" Epoch {epoch + 1} | train_loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'Conditional diffusion/diffusion_final_model.pth')
torch.save(ema_model.state_dict(), 'Conditional diffusion/diffusion_ema_model.pth')
print("模型权重已保存至: Conditional diffusion/diffusion_final_model.pth")
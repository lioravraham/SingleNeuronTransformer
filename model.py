# --- Standard Library ---
import os
import sys
import time
import glob
from typing import Tuple

# --- Third-Party Libraries ---
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, explained_variance_score, f1_score,
    mean_squared_error, mean_absolute_error, accuracy_score
)
# from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset

import torch._dynamo
torch._dynamo.config.suppress_errors = True
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")

# --- Set multiprocessing strategy and device ---
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Local Imports ---
from dataloader_layer5 import SimulationData

# --- Debugging Info ---
print("Python executable:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)              # None == CPU-only build
print("torch.backends.cudnn.is_available():", torch.backends.cudnn.is_available())
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))

####################################################################################################
# MINIMAL CHANGE ADDITIONS (masking GT soma > -55 instead of clamping)
VOLT_MAX = -55.0

def masked_mse_loss(y_pred, y_gt, mask, eps=1e-12):
    # y_pred, y_gt: (B,T) ; mask: bool same shape
    mask_f = mask.float()
    se = (y_pred - y_gt) ** 2
    se = se * mask_f
    denom = mask_f.sum().clamp_min(eps)
    return se.sum() / denom

def masked_temporal_derivative_mse(y_pred, y_gt, mask, eps=1e-12):
    # only count derivatives where BOTH consecutive points are valid
    if y_pred.dim() == 3: y_pred = y_pred.squeeze(-1)
    if y_gt.dim() == 3:   y_gt   = y_gt.squeeze(-1)
    if mask.dim() == 3:   mask   = mask.squeeze(-1)

    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    dy_gt   = y_gt[:, 1:]   - y_gt[:, :-1]

    dm = mask[:, 1:] & mask[:, :-1]
    dm_f = dm.float()
    se = (dy_pred - dy_gt) ** 2
    se = se * dm_f
    denom = dm_f.sum().clamp_min(eps)
    return se.sum() / denom
####################################################################################################

def plot_spikes_and_voltage(
    y_spikes_gt, y_spikes_pred, y_soma_gt, y_soma_pred,
    save_dir, thresh=0.5, prefix="comparison"
):
    os.makedirs(save_dir, exist_ok=True)
    T = len(y_spikes_gt)
    time_arr = np.arange(T)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Spikes: binary 0/1
    axs[0].step(time_arr, y_spikes_gt, where='post', label="Spikes GT", color='red')
    spike_probs = torch.sigmoid(torch.tensor(y_spikes_pred)).numpy()
    axs[0].step(time_arr, spike_probs, where='post', label="Spikes Pred", color='blue')
    axs[0].set_ylabel("Spike (0/1)")
    axs[0].legend()
    axs[0].set_title("Spike Ground Truth vs Prediction")

    # Soma voltage: continuous
    # MINIMAL CHANGE: hide GT>VOLT_MAX points in plot (remove them visually)
    y_soma_gt = np.array(y_soma_gt, dtype=np.float64)
    y_soma_pred = np.array(y_soma_pred, dtype=np.float64)
    invalid = (y_soma_gt > VOLT_MAX)
    y_soma_gt_plot = y_soma_gt.copy()
    y_soma_pred_plot = y_soma_pred.copy()
    y_soma_gt_plot[invalid] = np.nan
    y_soma_pred_plot[invalid] = np.nan

    axs[1].plot(time_arr, y_soma_gt_plot, label="Soma GT", color='red')
    axs[1].plot(time_arr, y_soma_pred_plot, label="Soma Pred", color='blue', alpha=0.7)
    axs[1].set_ylabel("Soma Voltage (mV)")
    axs[1].set_xlabel("Time step")
    axs[1].legend()
    axs[1].set_title("Soma Voltage Ground Truth vs Prediction (GT>-55 removed)")

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{prefix}_spikes_soma.png")
    plt.savefig(fname)
    plt.close(fig)
    print(f"[INFO] Saved comparison plot to {fname}")

def find_latest_state_dict(models_folder):
    # prefer "best", else latest "epoch*"
    bests = sorted(glob.glob(os.path.join(models_folder, "*_best.pt")), key=os.path.getmtime, reverse=True)
    if bests:
        return bests[0], "best"
    epochs = sorted(glob.glob(os.path.join(models_folder, "*_epoch*.pt")), key=os.path.getmtime, reverse=True)
    if epochs:
        return epochs[0], "epoch"
    return None, None
import math

def temporal_derivative_mse(y_pred, y_gt):
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(-1)
    if y_gt.dim() == 3:
        y_gt = y_gt.squeeze(-1)
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    dy_gt   = y_gt[:, 1:]   - y_gt[:, :-1]
    return F.mse_loss(dy_pred, dy_gt)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return self.scale * x

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff_mult=4):
        super().__init__()
        inner = int(d_ff_mult * d_model)
        self.w1 = nn.Linear(d_model, inner * 2)  # split => gate + value
        self.w2 = nn.Linear(inner, d_model)
    def forward(self, x):
        u, v = self.w1(x).chunk(2, dim=-1)          # (B,T,inner) x2
        return self.w2(F.silu(u) * v)

def rope_apply(x, sin, cos):
    # x: (B, T, H, D)  with D even
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotx1 = x1 * cos - x2 * sin
    rotx2 = x1 * sin + x2 * cos
    x_out = torch.stack((rotx1, rotx2), dim=-1).flatten(-2)
    return x_out

def build_rope_cache(T, dim, device, base=10000):
    # returns (sin, cos) shaped (T, dim/2)
    half = dim // 2
    pos = torch.arange(T, device=device).float()
    freqs = torch.exp(-math.log(base) * torch.arange(0, half, device=device).float() / half)
    ang = torch.outer(pos, freqs)                   # (T, half)
    sin, cos = ang.sin(), ang.cos()
    return sin, cos

class LocalRoPEBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.05, d_ff_mult=4, window=64):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff_mult=d_ff_mult)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.window = window

    def _build_local_causal_mask(self, T, device):
        # allow attention to last `window` טיימסטפס אחורה בלבד (כולל עצמי), חסום קדימה
        mask = torch.full((T, T), float('-inf'), device=device)
        for t in range(T):
            start = max(0, t - self.window + 1)
            mask[t, start:t+1] = 0.0
        return mask  # shape (T,T)

    def forward(self, x):
        # x: (B, T, d)
        B, T, D = x.shape
        sin, cos = build_rope_cache(T, self.head_dim, x.device)
        qkv_in = self.norm1(x)
        W_q = self.attn.in_proj_weight[:D, :]
        W_k = self.attn.in_proj_weight[D:2*D, :]
        W_v = self.attn.in_proj_weight[2*D:, :]
        b_q = self.attn.in_proj_bias[:D]
        b_k = self.attn.in_proj_bias[D:2*D]
        b_v = self.attn.in_proj_bias[2*D:]

        q = F.linear(qkv_in, W_q, b_q).view(B, T, self.nhead, self.head_dim)
        k = F.linear(qkv_in, W_k, b_k).view(B, T, self.nhead, self.head_dim)
        v = F.linear(qkv_in, W_v, b_v)

        assert (self.head_dim % 2) == 0, "head_dim must be even for RoPE"
        sinH = sin.view(1, T, 1, -1)  # (1,T,1,half)
        cosH = cos.view(1, T, 1, -1)
        q = rope_apply(q, sinH, cosH)
        k = rope_apply(k, sinH, cosH)
        q = q.flatten(2, 3)  # (B,T,D)
        k = k.flatten(2, 3)  # (B,T,D)

        attn_mask = self._build_local_causal_mask(T, x.device)  # (T,T)
        attn_out, _ = self.attn(q, k, v, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class LocalRoPETransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.05, d_ff_mult=4, window=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            LocalRoPEBlock(d_model, nhead, dropout, d_ff_mult, window) for _ in range(num_layers)
        ])
    def forward(self, x):  # (B,T,d)
        for blk in self.blocks:
            x = blk(x)
        return x
    
class SimpleTransformerModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 450,  # number of different locations on the dendrites
        kernel_size: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        max_seq_len: int = 4096,
        dropout_rate: float = 0.05,
    ):
        super(SimpleTransformerModel, self).__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1) Conv1D embedding layers 
        self.conv_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=0
        )
        self.conv_ln = nn.LayerNorm(input_channels)
        self.conv_embed2 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=9,
            padding=0
        )
        self.conv_ln2 = nn.LayerNorm(input_channels)

        # 2) Linear projection & pos embedding
        self.input_projection = nn.Linear(input_channels, d_model)
       
        self.dropout = nn.Dropout(dropout_rate)

        # 3) Transformer encoder
        self.transformer_encoder = LocalRoPETransformer(
            d_model=d_model, nhead=nhead, num_layers=6, dropout=dropout_rate,
            d_ff_mult=4, window=64  
        )

        self.soma_rnn = nn.LSTM(d_model, d_model, num_layers=3, bidirectional=False, batch_first=True)

        self.soma_conv_k = 5
        self.soma_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=self.soma_conv_k,
            groups=d_model,
            bias=True
        )

        # 4) Output heads
        self.spike_head = nn.Linear(d_model, 1)
        self.soma_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.soma_head = nn.Linear(d_model, 1)

        nn.init.xavier_uniform_(self.spike_head.weight)
        nn.init.zeros_(self.spike_head.bias)
        nn.init.xavier_uniform_(self.soma_head.weight)
        nn.init.zeros_(self.soma_head.bias)

        self.short_name = f"Transformer_MLP_ks{kernel_size}_{nhead}heads_conv_layer5"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, T)
        """
        B, C, T = x.shape

        x = x.view(B, C, T) # Causal conv stack

        # conv1 (k=3) with left padding of k-1
        x = F.pad(x, (self.conv_embed.kernel_size[0] - 1, 0))
        x = self.conv_embed(x)
        x = self.conv_ln(x.permute(0, 2, 1)).permute(0, 2, 1)

        # conv2 (k=9) with left padding of k-1
        x = F.pad(x, (self.conv_embed2.kernel_size[0] - 1, 0))
        x = self.conv_embed2(x)
        x = self.conv_ln2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x.permute(0, 2, 1)  # (B, T, C)

        # projection + pos-emb BEFORE encoder
        x = self.input_projection(x)
        x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
        x = self.dropout(x)

        # heads
        spikes_out = self.spike_head(x)           # (B, seq_len, 1)
        rnn_out, _ = self.soma_rnn(x)
        soma_feats = self.soma_mlp(rnn_out) + rnn_out
        soma_out = self.soma_head(soma_feats)

        y_spikes = spikes_out[:, -T:, :]
        y_soma   = soma_out[:, -T:, :]
        return y_spikes, y_soma

def pick_ckpt_by_ap_with_auc_floor(model, models_folder, valid_dataset,
                                  criterion_spikes, criterion_soma,
                                  auc_floor=0.965, max_ckpts=15):
    """
    Evaluate a few checkpoints and choose the one with highest AP
    among those with AUC >= auc_floor.
    Minimal: uses your existing evaluate_model_on_dataset().
    """
    ckpts = []
    ckpts += glob.glob(os.path.join(models_folder, f"{model.short_name}_best_auc.pt"))
    ckpts += glob.glob(os.path.join(models_folder, f"{model.short_name}_best.pt"))
    ckpts += sorted(glob.glob(os.path.join(models_folder, f"{model.short_name}_epoch*.pt")),
                    key=os.path.getmtime, reverse=True)[:max_ckpts]

    ckpts = list(dict.fromkeys(ckpts))  # de-dup, keep order

    results = []
    for p in ckpts:
        try:
            model.load_state_dict(torch.load(p, map_location=device), strict=False)
            md, *_ = evaluate_model_on_dataset(
                model, valid_dataset, criterion_spikes, criterion_soma,
                batch_size=batch_size, verbose=0
            )
            results.append((p, float(md["AUC_score"]), float(md["AP_score"])))
            print(f"[CKPT] AUC={results[-1][1]:.4f}  AP={results[-1][2]:.4f}  {os.path.basename(p)}")
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if not results:
        return None

    good = [r for r in results if r[1] >= auc_floor]
    if not good:
        good = sorted(results, key=lambda x: x[1], reverse=True)  # fallback best AUC

    best = sorted(good, key=lambda x: x[2], reverse=True)[0]     # best AP among good AUC
    print(f"[INFO] Selected: {os.path.basename(best[0])}  (AUC={best[1]:.4f}, AP={best[2]:.4f})")
    return best[0]

def sweep_thresholds(y_prob, y_true):
    y_prob = y_prob.ravel()
    y_true = y_true.ravel().astype(int)

    thresholds = np.linspace(0.05, 0.95, 19)
    best = None
    for t in thresholds:
        y_hat = (y_prob > t).astype(int)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        f = f1_score(y_true, y_hat, zero_division=0)
        f2 = f_beta_score(p, r, beta=2.0)

        print(f"thr={t:.2f} | P={p:.3f} R={r:.3f} F1={f:.3f} F2={f2:.3f}")
        if best is None or f > best[0]:
            best = (f, t, p, r)
    print(f"[BEST F1 in sweep] thr={best[1]:.2f}  P={best[2]:.3f}  R={best[3]:.3f}  F1={best[0]:.3f}")
    return best

def apply_refractory(y_hat, refractory_bins=3):
    out = y_hat.copy()
    last = -1e9
    for i in range(len(out)):
        if out[i] == 1:
            if i - last <= refractory_bins:
                out[i] = 0
            else:
                last = i
    return out

def f_beta_score(p, r, beta=2.0, eps=1e-12):
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + eps)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits, targets: same shape
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)      # prob of the true class
        loss = self.alpha * (1 - pt).pow(self.gamma) * bce
        return loss.mean()


if __name__ == "__main__":
    start_time = time.perf_counter()
    print('----------------------------')
    print(f'Using device: {device}')
    print('----------------------------')

    # ========================== DATASET SETUP ==========================
    models_folder = '/ems/elsc-labs/london-m/lior.avraham1/layer5/david_model_empty' # for the model with david's changes
    os.makedirs(models_folder, exist_ok=True)

    train_time_window_size = 1024
    valid_time_window_size = 1024
    batch_size = 8
    # preload_data = True
    # preload_data = False
    dataset_folder = "/ems/elsc-labs/segev-i/ido.aizenbud/Data/neuron_as_deep_net2_data/Rat_L5b_PC_2_Hay_160um_wide_weighted1_5_1_wide_fr5_mc200_pipeline_4/simulation_dataset"

    valid_name = 'test'
    train_name = 'train'
    sim_number = 100
    used_v_offset = 1e9
    v_clip = 1e9
    num_workers = 1

    # load the full train and valid sets
    train_data_full = SimulationData(f'{dataset_folder}/{train_name}')
    valid_data_full = SimulationData(f'{dataset_folder}/{valid_name}')
    print("=== Dataset timing info ===")
    print(f"Raw simulation length (ms): {train_data_full.simulation_duration_in_ms}")
    print(f"Window size (ms):           {train_data_full.window_size}")
    print(f"Overlap size (ms):          {train_data_full.overlap_size}")
    print(f"Start_t (ms):               {train_data_full.start_t}")
    print(f"Windows per simulation:     {train_data_full.num_per_sim}")
    print(f"Total simulations:          {train_data_full.count_simulations}")
    print(f"Total samples:              {len(train_data_full)}")


    print("number of simulation in valid_data_full", len(valid_data_full.simulation_indices))
    print("number of simulation in train_data_full", len(train_data_full.simulation_indices))
    # choose some indices from each
    # train_idx = np.random.choice(len(train_data_full), size=20000, replace=False)
    train_idx = np.random.choice(len(train_data_full), size=100, replace=False)
    valid_idx = np.arange(len(valid_data_full))
    # valid_idx = np.random.choice(len(valid_data_full), size=4000, replac

    # wrap them
    train_dataset = Subset(train_data_full, train_idx)
    valid_dataset = Subset(valid_data_full, valid_idx)

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    print('----------------------------------------------------')
    print('fetch a batch')
    print('-------------')
    batch = next(iter(train_dataloader))
    #test for the dat aand spikes imbalance 
   

    x = batch["sps_in"]
    y = batch["sps_out"]

    print("sps_in shape:", x.shape, "dtype:", x.dtype,
        "min/max:", x.min().item(), x.max().item())

    print("sps_out shape:", y.shape, "dtype:", y.dtype,
        "min/max:", y.min().item(), y.max().item())

    # check if sps_out is binary
    u = torch.unique(y)
    print("unique sps_out values (up to 20):", u[:20], "count:", u.numel())

    #end test 
    X_spikes = batch['sps_in'].float().to(device)
    y_soma_GT = batch['somatic_voltage_out'].float().to(device)
    y_spikes_GT = batch['sps_out'].float().to(device)
    print(f'X_spikes.shape: {X_spikes.shape}')
    print(f'y_spikes_GT.shape: {y_spikes_GT.shape}')
    print(f'y_soma_GT.shape: {y_soma_GT.shape}')
    print("Mean spike rate:", y_spikes_GT.mean().item())
    print('----------------------------------------------------')

    # compute normalization from train set (AFTER clamping to -55 mV)
    all_somas = []
    for b in train_dataloader:
        y_soma_tmp = b['somatic_voltage_out']  
        all_somas.append(y_soma_tmp.flatten())
    all_somas = torch.cat(all_somas)
    # MINIMAL CHANGE: instead of clamp, FILTER OUT values > -55
    all_somas = all_somas[all_somas <= VOLT_MAX]
    if all_somas.numel() == 0:
        raise RuntimeError(f"No soma values <= {VOLT_MAX} found for normalization!")

    # ========================== SPIKE STATS / IMBALANCE ==========================
    print("\n==================== Spike stats (train subset) ====================")

    # 1) Inspect a single batch: what are sps_in / sps_out?
    x0 = batch["sps_in"]
    y0 = batch["sps_out"]

    print("sps_in  shape:", tuple(x0.shape), "dtype:", x0.dtype,
        "min/max:", float(x0.min()), float(x0.max()))
    print("sps_out shape:", tuple(y0.shape), "dtype:", y0.dtype,
        "min/max:", float(y0.min()), float(y0.max()))

    # Check whether sps_out is binary
    u = torch.unique(y0)
    print("unique sps_out values (up to 20):", u[:20].cpu().numpy(), "count:", u.numel())

    # Decide how to binarize sps_out robustly
    # (If it's already 0/1, this keeps it unchanged)
    def binarize_sps_out(y, thr=0.5):
        return (y.float() > thr).to(torch.int32)

    # 2) Compute imbalance + mean spike rate per window on the TRAIN dataset you actually use (Subset)
    pos_bins = 0
    total_bins = 0

    window_mean_rates = []   # mean spike rate per window/sample

    for b in tqdm(train_dataloader, desc="Computing spike stats (train)"):
        y = b["sps_out"]                  # shape (B,T) or (B,T,1)
        y_bin = binarize_sps_out(y)       # int {0,1}

        pos_bins += int(y_bin.sum().item())
        total_bins += int(y_bin.numel())

        # mean spikes per window (per sample in batch)
        B = y_bin.shape[0]
        y_flat = y_bin.view(B, -1)        # (B, T*)
        window_mean_rates.extend(y_flat.float().mean(dim=1).cpu().tolist())

    pos_rate = pos_bins / max(total_bins, 1)
    neg_bins = total_bins - pos_bins
    neg_pos_ratio = neg_bins / max(pos_bins, 1)

    print("\n---- TRAIN (your Subset) imbalance ----")
    print(f"Total bins:       {total_bins}")
    print(f"Spike bins (pos): {pos_bins}")
    print(f"Non-spike (neg):  {neg_bins}")
    print(f"Pos rate:         {pos_rate:.6e}")
    print(f"Neg/Pos ratio:    {neg_pos_ratio:.3f}  (non-spike bins per spike bin)")

    # 3) Mean spike rate per window/sample + distribution summary
    wm = np.array(window_mean_rates, dtype=np.float64)
    print("\n---- TRAIN mean spike rate per window (sample/window) ----")
    print(f"Num windows:      {wm.size}")
    print(f"Mean:             {wm.mean():.6e}")
    print(f"Std:              {wm.std():.6e}")
    print(f"Median:           {np.median(wm):.6e}")
    print(f"95th percentile:  {np.percentile(wm, 95):.6e}")
    print("====================================================================\n")
    # ============================================================================


    V_bias  = all_somas.mean().item()
    V_scale = all_somas.std().item()
    print(f'V_bias: {V_bias}')
    print(f'V_scale: {V_scale}')

    spikes_loss_weight = 0.1*10 
    soma_loss_weight = 0.1

    # ========================== MODEL / OPTIM ==========================
    model = SimpleTransformerModel(input_channels=450, d_model=312, nhead=12, dropout_rate=0.1).to(device)
    best_auc_candidates = [
        os.path.join(models_folder, f"{model.short_name}_best_auc.pt"),
        os.path.join(models_folder, f"{model.short_name}_best_auc_offline.pt"),
    ]
    loaded_best_auc = False
    for p in best_auc_candidates:
        if os.path.isfile(p):
            state = torch.load(p, map_location=device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[INFO] Initialized weights from best-by-AUC: {p}")
            if missing:   print(f"[WARN] missing keys: {missing}")
            if unexpected:print(f"[WARN] unexpected keys: {unexpected}")
            loaded_best_auc = True
            break

    if not loaded_best_auc:
        print("[INFO] No best-by-AUC checkpoint found; using random init for now.")

    ckpt_path, ckpt_kind = find_latest_state_dict(models_folder)
    if ckpt_path:
        print(f"[INFO] Loading {ckpt_kind} weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] missing keys: {missing}")
            print(f"[WARN] unexpected keys: {unexpected}")
    else:
        print("[INFO] No previous state_dict checkpoint found; training from scratch.")

    torch.cuda.empty_cache()
    print('------------------------------------------------------------------------------------------')
    print(f'full model: "{model.short_name}"')
    print('------------------------------------------------------------------------------------------')
    _ = model(X_spikes.to(device))  # dry run
    print('------------------------------------------------------------------------------------------')

    # Training parameters
    num_epochs = 350
    learning_rate = 5e-4

    # ---------- PERIODIC VALIDATION FREQUENCY ----------
    num_batches_per_valid_eval = max(1, len(train_dataloader) // 4)  # validate ~4x per epoch

    # Containers for ITERATION-LEVEL curves (periodic validation)
    train_iter_list = []
    train_losses_spikes = []
    train_losses_soma = []
    train_losses_total = []

    valid_iter_list = []
    valid_losses_spikes = []
    valid_losses_soma = []
    valid_losses_total = []

    last_valid_spike = None
    last_valid_soma  = None

    # Losses / optimizer / scheduler
    pos_weight_value = 5.0
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    # criterion_spikes = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #this is the correct loss according to David! 
    criterion_spikes = FocalLoss(alpha=0.5, gamma=2.0)

    criterion_soma = nn.MSELoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, cooldown=0, min_lr=1e-6
    )

    iter_num = 0
    best_auc = -1.0

    from torch.cuda.amp import GradScaler
    scaler = GradScaler("cuda")

    resume_path = os.path.join(models_folder, f"{model.short_name}_resume.pt")
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_auc  = -1.0  
    best_val_ap = -1.0

    if os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        print(f"[INFO] Resuming training from {resume_path}")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        V_bias  = float(ckpt.get("V_bias", V_bias))
        V_scale = float(ckpt.get("V_scale", V_scale))

        best_val_auc = float(ckpt.get("best_val_auc", best_val_auc))
        print(f"[INFO] start_epoch={start_epoch}, best_val_loss={best_val_loss:.4f}, "
              f"V_bias={V_bias:.3f}, V_scale={V_scale:.3f}, best_val_auc={best_val_auc:.4f}")
    else:
        print("[INFO] No resume checkpoint found; starting from scratch (but you may have loaded a state_dict above).")

    prefer_auc = True
    if prefer_auc:
        print("using weights according to best auc")
        auc_path = os.path.join(models_folder, f"{model.short_name}_best_auc.pt")
        if os.path.isfile(auc_path):
            print(f"[INFO] Overriding with best-by-AUC: {auc_path}")
            model.load_state_dict(torch.load(auc_path, map_location=device))


    # ============================ TRAINING LOOP ============================
    # extra_epoch = 250
    # end_epoch = max(start_epoch + extra_epoch, num_epochs)
    start_epoch = 0
    end_epoch = 1
    for epoch in range(start_epoch, end_epoch):
        model.train()
        print(f'Epoch {epoch + 1}/{end_epoch}{" " * 60}(spikes loss, soma loss)')

        progress_bar = tqdm(train_dataloader, desc='Training')
        for batch_idx, batch in enumerate(progress_bar):
            X_spikes   = batch['sps_in'].float().to(device)

            # MINIMAL CHANGE: build mask from RAW GT, then normalize, and compute soma loss only on mask
            y_soma_raw = batch['somatic_voltage_out'].float().squeeze().to(device)
            soma_mask = (y_soma_raw <= VOLT_MAX)
            y_soma_GT = (y_soma_raw - V_bias) / V_scale

            y_spikes_GT= batch['sps_out'].float().squeeze().to(device)

            # forward
            y_spikes_pred, y_soma_pred = model(X_spikes)
            y_spikes_pred = y_spikes_pred.squeeze()
            y_soma_pred   = y_soma_pred.squeeze()

            # losses
            loss_spikes = criterion_spikes(y_spikes_pred, y_spikes_GT)

            # MINIMAL CHANGE: replace MSE with masked MSE (ignore GT>-55)
            loss_soma   = masked_mse_loss(y_soma_pred, y_soma_GT, soma_mask)

            lambda_taylor_soma  = 0.1  
            lambda_taylor_spike = 0.02  

            # MINIMAL CHANGE: derivative loss masked on valid soma points
            taylor_soma  = masked_temporal_derivative_mse(y_soma_pred, y_soma_GT, soma_mask)
            taylor_spike = temporal_derivative_mse(torch.sigmoid(y_spikes_pred), y_spikes_GT)

            loss = spikes_loss_weight * loss_spikes + soma_loss_weight * loss_soma \
                + lambda_taylor_soma  * taylor_soma \
                + lambda_taylor_spike * taylor_spike
            # loss = loss_spikes   # spike-head fine-tune

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # store iteration metrics
            train_iter_list.append(iter_num)
            train_losses_spikes.append(loss_spikes.item())
            train_losses_soma.append(loss_soma.item())
            train_losses_total.append(loss.item())

            # progress bar strings
            train_str = f'({loss_spikes.item():.4f}, {loss_soma.item():.4f})'
            if last_valid_spike is None or last_valid_soma is None:
                val_str = "(N/A, N/A)"
            else:
                val_str = f'({last_valid_spike:.4f}, {last_valid_soma:.4f})'
            progress_bar.set_postfix({'train': train_str, 'val': val_str})

            if (iter_num + 1) % num_batches_per_valid_eval == 0:
                model.eval()
                v_spk_batch = []
                v_soma_batch = []
                v_total_batch = []
                with torch.no_grad():
                    for vb in valid_dataloader:
                        Xv = vb['sps_in'].float().to(device)
                        yv_spk = vb['sps_out'].float().squeeze().to(device)

                        # MINIMAL CHANGE: mask instead of clamp
                        yv_soma_raw = vb['somatic_voltage_out'].float().squeeze().to(device)
                        v_soma_mask = (yv_soma_raw <= VOLT_MAX)
                        yv_soma = (yv_soma_raw - V_bias) / V_scale

                        pv_spk, pv_soma = model(Xv)
                        pv_spk = pv_spk.squeeze()
                        pv_soma = pv_soma.squeeze()

                        lv_spk = criterion_spikes(pv_spk, yv_spk)
                        lv_soma = masked_mse_loss(pv_soma, yv_soma, v_soma_mask)
                        lv_total = spikes_loss_weight * lv_spk + soma_loss_weight * lv_soma

                        v_spk_batch.append(lv_spk.item())
                        v_soma_batch.append(lv_soma.item())
                        v_total_batch.append(lv_total.item())

                v_spk_mean = float(np.mean(v_spk_batch))
                v_soma_mean = float(np.mean(v_soma_batch))
                v_total_mean = float(np.mean(v_total_batch))

                valid_iter_list.append(iter_num)
                valid_losses_spikes.append(v_spk_mean)
                valid_losses_soma.append(v_soma_mean)
                valid_losses_total.append(v_total_mean)

                # scheduler step on periodic val
                print("scheduler with spikes and soma (v_total_mean)")
                scheduler.step(v_total_mean)
                
                # track for progress bar
                last_valid_spike = v_spk_mean
                last_valid_soma  = v_soma_mean

                # save "best" on periodic evaluation (by loss)
                if v_total_mean < best_val_loss:
                    best_val_loss = v_total_mean
                    ckpt_name = f"{model.short_name}_best.pt"
                    torch.save(model.state_dict(), os.path.join(models_folder, ckpt_name))
                    print(f"[INFO] New best (iter {iter_num+1}) val loss {best_val_loss:.4f} -> saved {ckpt_name}")

                with torch.no_grad():
                    all_logits = []
                    all_targets = []
                    for vb in valid_dataloader:
                        Xv = vb['sps_in'].float().to(device)
                        yv_spk = vb['sps_out'].float().squeeze().to(device)
                        pv_spk, _ = model(Xv)
                        all_logits.append(pv_spk.detach().reshape(-1))
                        all_targets.append(yv_spk.detach().reshape(-1))
                    all_logits  = torch.cat(all_logits)
                    all_targets = torch.cat(all_targets).float()
                    # roc_auc_score expects probabilities:
                    val_auc = roc_auc_score(
                        all_targets.cpu().numpy(),
                        torch.sigmoid(all_logits).cpu().numpy()
                    )
                    val_ap = average_precision_score(
                        all_targets.cpu().numpy(),
                        torch.sigmoid(all_logits).cpu().numpy()
                    )
                if val_ap > best_val_ap:
                    best_val_ap = val_ap
                    torch.save(model.state_dict(), os.path.join(models_folder, f"{model.short_name}_best_ap.pt"))
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    ckpt_name_auc = f"{model.short_name}_best_auc.pt"
                    torch.save(model.state_dict(), os.path.join(models_folder, ckpt_name_auc))
                    print(f"[INFO] New best (iter {iter_num+1}) val AUC {best_val_auc:.4f} -> saved {ckpt_name_auc}")
                torch.cuda.empty_cache()
                model.train()

            iter_num += 1  

        ckpt_name = f"{model.short_name}_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), os.path.join(models_folder, ckpt_name))
        print(f"[INFO] Saved checkpoint at epoch {epoch+1}: {ckpt_name}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_auc": best_val_auc,  
            "V_bias": V_bias,
            "V_scale": V_scale,
        }, os.path.join(models_folder, f"{model.short_name}_resume.pt"))

        torch.cuda.empty_cache()

    print('----------------------------------------------------')
    print('Training finished!')
    print('----------------------------------------------------')
    if len(train_losses_spikes) > 0 and len(valid_losses_spikes) > 0:
        print(f'Final spikes loss (train, valid): {train_losses_spikes[-1]:.5f}, {valid_losses_spikes[-1]:.5f}')
        print(f'Final soma V loss (train, valid): {train_losses_soma[-1]:.5f}, {valid_losses_soma[-1]:.5f}')
    else:
        print("[INFO] No training was run in this execution (loss lists are empty). Skipping training-summary prints.")
        print('----------------------------------------------------')

    # =======================  PLOTTING (iteration-based)  =======================
    save_dir_learning_curves = '/ems/elsc-labs/london-m/lior.avraham1/layer5/training_net_job_output2/learning_curves'
    os.makedirs(save_dir_learning_curves, exist_ok=True)
    matplotlib.use('Agg')

    y_scale_list = ['linear', 'log', 'log']
    x_scale_list = ['linear', 'linear', 'log']
    if len(train_iter_list) == 0:
        print("[INFO] No iterations were run; skipping learning-curve plots.")
    else:
        for y_scale, x_scale in zip(y_scale_list, x_scale_list):
            plt.figure(figsize=(10, 8))

            # Spikes loss
            plt.subplot(3, 1, 1)
            plt.plot(train_iter_list, train_losses_spikes, label='Train')
            plt.plot(valid_iter_list, valid_losses_spikes, label='Valid')
            plt.title('Spikes Loss')
            plt.legend(fontsize=12)
            plt.yscale(y_scale)
            plt.xscale(x_scale)
            plt.grid(True)
            plt.xlim([0 if x_scale == 'linear' else 30, max(train_iter_list)])

            # Soma loss
            plt.subplot(3, 1, 2)
            plt.plot(train_iter_list, train_losses_soma, label='Train')
            plt.plot(valid_iter_list, valid_losses_soma, label='Valid')
            plt.title('Soma Loss')
            plt.legend()
            plt.yscale(y_scale)
            plt.xscale(x_scale)
            plt.grid(True)
            plt.xlim([0 if x_scale == 'linear' else 30, max(train_iter_list)])

            # Total loss
            plt.subplot(3, 1, 3)
            plt.plot(train_iter_list, train_losses_total, label='Train')
            plt.plot(valid_iter_list, valid_losses_total, label='Valid')
            plt.title('Total Loss')
            plt.xlabel('Train Iteration')
            plt.legend()
            plt.yscale(y_scale)
            plt.xscale(x_scale)
            plt.grid(True)
            plt.xlim([0 if x_scale == 'linear' else 30, max(train_iter_list)])

            plt.tight_layout()
            filename = f'loss_plot_y-{y_scale}_x-{x_scale}.png'
            filepath = os.path.join(save_dir_learning_curves, filename)
            plt.savefig(filepath)
            print(f"Saving plot to {filepath}")
            plt.close()

    print('----------------------------------------------------')
    if len(train_losses_spikes) > 0 and len(valid_losses_spikes) > 0:
        print(f'Final spikes loss (train, valid): {train_losses_spikes[-1]:.5f}, {valid_losses_spikes[-1]:.5f}')
        print(f'Final soma V loss (train, valid): {train_losses_soma[-1]:.5f}, {valid_losses_soma[-1]:.5f}')
    else:
        print("[INFO] No training was run in this execution (loss lists are empty). Skipping training-summary prints.")
        print('----------------------------------------------------')

    # ============================== Evaluation ==============================
    def predict_on_all_simulations(model, dataset, batch_size=batch_size, V_bias=V_bias, V_scale=V_scale):
        model.eval()
        dev = next(model.parameters()).device

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_y_spikes_prob   = []
        all_y_spikes_logits = []
        all_y_soma_pred = []
        all_y_spikes_gt = []
        all_y_soma_gt = []

        with torch.no_grad():
            for i, b in enumerate(dataloader):
                X_sp = b['sps_in'].float().to(dev)

                # MINIMAL CHANGE: keep raw GT (no clamp) so we can filter later
                y_sv  = b['somatic_voltage_out'].float().squeeze().to(dev)

                y_spgt = b['sps_out'].float().squeeze().to(dev)

                y_sp_logits, y_soma_pred = model(X_sp)   # y_sp_logits are logits
                y_soma_pred = V_scale * y_soma_pred + V_bias

                # Save BOTH logits and probabilities
                sp_logits_np = y_sp_logits.detach().cpu().numpy()
                sp_prob_np   = torch.sigmoid(y_sp_logits).detach().cpu().numpy()

                all_y_spikes_logits.append(sp_logits_np)
                all_y_spikes_prob.append(sp_prob_np)

                all_y_spikes_gt.append(y_spgt.detach().cpu().numpy())
                all_y_soma_gt.append(y_sv.detach().cpu().numpy())
                all_y_soma_pred.append(y_soma_pred.detach().cpu().numpy())

        y_spikes_logits = np.concatenate([x.squeeze() for x in all_y_spikes_logits], axis=0)
        y_spikes_prob   = np.concatenate([x.squeeze() for x in all_y_spikes_prob], axis=0)
        y_spikes_gt     = np.concatenate([x.squeeze() for x in all_y_spikes_gt], axis=0)
        y_soma_gt       = np.concatenate([x.squeeze() for x in all_y_soma_gt], axis=0)
        y_soma_pred     = np.concatenate([x.squeeze() for x in all_y_soma_pred], axis=0)

        return y_spikes_logits, y_spikes_prob, y_soma_pred, y_spikes_gt, y_soma_gt

    from sklearn.metrics import precision_recall_curve

    def calculate_metrics(
        y_spikes_gt, y_spikes_prob, y_spikes_logits, y_soma_gt, y_soma_pred,
        criterion_cls, criterion_reg,
        num_datapoints_in_scatter=20000,
        print_metrics=True
    ):
        # ---------- ROC metrics (still useful) ----------
        fpr, tpr, roc_thresholds = roc_curve(y_spikes_gt.ravel(), y_spikes_prob.ravel())
        AUC_score = auc(fpr, tpr)

        # ---------- PR metrics (more relevant for rare spikes) ----------
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
            y_spikes_gt.ravel(), y_spikes_prob.ravel()
        )
        AP_score = average_precision_score(y_spikes_gt.ravel(), y_spikes_prob.ravel())

        # precision_arr/recall_arr have length N+1, thresholds length N
        eps = 1e-12
        f1_arr = (2 * precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + eps)
        best_idx = int(np.argmax(f1_arr))
        thresh = float(pr_thresholds[best_idx])

        # Apply chosen threshold
        y_spikes_pred_binary = (y_spikes_prob > thresh).astype(int)

        f1 = f1_score(y_spikes_gt.ravel(), y_spikes_pred_binary.ravel(), zero_division=0)
        accuracy = accuracy_score(y_spikes_gt.ravel(), y_spikes_pred_binary.ravel())
        precision = precision_score(y_spikes_gt.ravel(), y_spikes_pred_binary.ravel(), zero_division=0)
        recall    = recall_score(y_spikes_gt.ravel(), y_spikes_pred_binary.ravel(), zero_division=0)

        # MINIMAL CHANGE: filter soma metrics to ONLY GT<=-55
        soma_mask = (y_soma_gt.ravel() <= VOLT_MAX)
        y_soma_gt_f = y_soma_gt.ravel()[soma_mask]
        y_soma_pred_f = y_soma_pred.ravel()[soma_mask]

        # Regression metrics
        soma_explained_variance = explained_variance_score(y_soma_gt_f, y_soma_pred_f)
        soma_explained_variance_percent = 100.0 * soma_explained_variance
        soma_RMSE = np.sqrt(mean_squared_error(y_soma_gt_f, y_soma_pred_f))
        soma_MAE = mean_absolute_error(y_soma_gt_f, y_soma_pred_f)

        # Loss values (IMPORTANT: BCEWithLogitsLoss expects LOGITS, not probabilities)
        spikes_loss = criterion_cls(
            torch.tensor(y_spikes_logits, device=device, dtype=torch.float32),
            torch.tensor(y_spikes_gt.astype(float), device=device, dtype=torch.float32)
        )
        # MINIMAL CHANGE: soma loss on filtered only
        soma_loss = criterion_reg(
            torch.tensor(y_soma_pred_f, device=device, dtype=torch.float32),
            torch.tensor(y_soma_gt_f, device=device, dtype=torch.float32)
        )

        # Scatter subsample (filtered only)
        n_sc = len(y_soma_gt_f)
        if n_sc == 0:
            scatter_soma_gt = np.array([])
            scatter_soma_pred = np.array([])
        else:
            selected_indices = np.random.choice(n_sc, min(num_datapoints_in_scatter, n_sc), replace=True)
            scatter_soma_gt = y_soma_gt_f[selected_indices]
            scatter_soma_pred = y_soma_pred_f[selected_indices]

        metrics_dict = {
            # classification curves
            'false_positive_rate': fpr,
            'true_positive_rate': tpr,
            'roc_thresholds': roc_thresholds,
            'precision_curve': precision_arr,
            'recall_curve': recall_arr,
            'pr_thresholds': pr_thresholds,

            # headline scores
            'AUC_score': AUC_score,
            'AP_score': AP_score,

            # point metrics at chosen threshold
            'optimal_spike_threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,

            # regression
            'soma_explained_variance_percent': soma_explained_variance_percent,
            'soma_RMSE': soma_RMSE,
            'soma_MAE': soma_MAE,

            # losses
            'spikes_loss': spikes_loss.item(),
            'soma_loss': soma_loss.item(),

            # scatter
            'scatter_soma_voltage_GT': scatter_soma_gt,
            'scatter_soma_voltage_pred': scatter_soma_pred,
        }

        if print_metrics:
            print(f"Best-F1 threshold (from PR curve) = {thresh:.6f}")
            print(f"AP (PR-AUC) = {AP_score:.4f}")
            print(f"AUC (ROC)   = {AUC_score:.4f}")
            print(f"precision   = {precision:.4f}")
            print(f"recall      = {recall:.4f}")
            print(f"f1          = {f1:.4f}")
            print(f"Accuracy    = {accuracy:.4f}")
            print(f"soma explained variance (GT<=-55 only) = {soma_explained_variance_percent:.2f}%")
            print(f"soma RMSE (GT<=-55 only) = {soma_RMSE:.2f} mV")
            print(f"soma MAE  (GT<=-55 only) = {soma_MAE:.2f} mV")

        return metrics_dict

    def plot_evaluation_figures(metrics_dict, voltage_granularity=8, voltage_setpoint=V_bias):
        """Create visualization of model performance metrics."""
        save_dir = '/ems/elsc-labs/london-m/lior.avraham1/layer5/training_net_job_output2/evaluation_figures'
        os.makedirs(save_dir, exist_ok=True)
        matplotlib.use('Agg')

        plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2)
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.3)

        # ROC Curve
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(metrics_dict['false_positive_rate'],
                 metrics_dict['true_positive_rate'],
                 color='k')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f"ROC Curve (AUC = {metrics_dict['AUC_score']:.4f})")
        ax1.grid(True)

        # Inset zoom
        axins = ax1.inset_axes([0.6, 0.1, 0.35, 0.35])
        axins.plot(metrics_dict['false_positive_rate'], metrics_dict['true_positive_rate'], color='k')
        axins.set_xlim(0, 0.01)
        axins.set_ylim(0, 1)
        axins.grid(True)
        ax1.indicate_inset_zoom(axins)

        # Voltage scatter
        ax2 = plt.subplot(gs[0, 1])
        selected_GT = metrics_dict['scatter_soma_voltage_GT']
        selected_pred = metrics_dict['scatter_soma_voltage_pred']
        ax2.scatter(selected_GT, selected_pred, s=1, alpha=0.5)

        if len(selected_GT) > 0:
            voltage_lims = [
                np.floor(min(selected_GT.min(), selected_pred.min())),
                np.ceil(max(selected_GT.max(), selected_pred.max()))
            ]
            ax2.plot(voltage_lims, voltage_lims, 'k--', alpha=0.5)

            voltage_ticks = np.arange(
                np.floor(voltage_lims[0] / voltage_granularity) * voltage_granularity,
                np.ceil(voltage_lims[1] / voltage_granularity) * voltage_granularity,
                voltage_granularity
            )
            ax2.set_xticks(voltage_ticks)
            ax2.set_yticks(voltage_ticks)

        ax2.set_xlabel('Ground Truth Voltage (mV)')
        ax2.set_ylabel('Predicted Voltage (mV)')
        ax2.set_title(f"Soma Voltage Prediction (GT<=-55 only)\n(R² = {metrics_dict['soma_explained_variance_percent']:.1f}%)")
        ax2.grid(True)

        # Error histogram
        ax3 = plt.subplot(gs[1, :])
        if len(selected_GT) > 0:
            voltage_errors = selected_pred - selected_GT
            ax3.hist(voltage_errors, bins=100, density=True)
            ax3.set_title(
                f"Voltage Prediction Error Distribution\nRMSE = {metrics_dict['soma_RMSE']:.2f} mV, "
                f"MAE = {metrics_dict['soma_MAE']:.2f} mV"
            )
        else:
            ax3.set_title("Voltage Prediction Error Distribution (no filtered points)")
        ax3.set_xlabel('Prediction Error (mV)')
        ax3.set_ylabel('Density')
        ax3.grid(True)

        fname = "evaluation.png"
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path)
        print(f"[INFO] Saved plot to {out_path}")
        plt.close()

    def evaluate_model_on_dataset(model, dataset, criterion_cls, criterion_reg,
                                  batch_size=batch_size, num_datapoints_in_scatter=20000, verbose=1):
        """Complete model evaluation pipeline."""
        print("validation start!")
        # predictions
        y_spikes_logits, y_spikes_prob, y_soma_pred, y_spikes_gt, y_soma_gt = predict_on_all_simulations(
            model, dataset, batch_size=batch_size, V_bias=V_bias, V_scale=V_scale
        )

        # metrics
        metrics_dict = calculate_metrics(
            y_spikes_gt=y_spikes_gt,
            y_spikes_prob=y_spikes_prob,
            y_spikes_logits=y_spikes_logits,
            y_soma_gt=y_soma_gt,
            y_soma_pred=y_soma_pred,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            num_datapoints_in_scatter=num_datapoints_in_scatter,
            print_metrics=(verbose > 0)
        )

        return metrics_dict, y_spikes_prob, y_soma_pred, y_spikes_gt, y_soma_gt
    
    # ============================== Load best checkpoint for final eval ==============================
    selected_ckpt = pick_ckpt_by_ap_with_auc_floor(
        model, models_folder, valid_dataset,
        criterion_spikes, criterion_soma,
        auc_floor=0.965, max_ckpts=15
    )

    if selected_ckpt is not None:
        model.load_state_dict(torch.load(selected_ckpt, map_location=device), strict=False)
    else:
        print("[WARN] Could not select checkpoint; keeping current weights.")

    # ---- Freeze backbone, train only spike head ----
    for p in model.parameters(): p.requires_grad = False
    for p in model.spike_head.parameters(): p.requires_grad = True
    for p in model.transformer_encoder.blocks[-1].parameters(): p.requires_grad = True

    # Evaluate model on validation set
    valid_metrics_dict, y_spikes_prob, y_soma_pred, y_spikes_gt, y_soma_gt = evaluate_model_on_dataset(
        model, valid_dataset, criterion_spikes, criterion_soma,
        batch_size=batch_size, verbose=1
    )
    # ---- threshold sweep ----
    sweep_thresholds(y_spikes_prob, y_spikes_gt)

    # ---- try one threshold + refractory ----
    t = 0.55   # pick from sweep output
    y_hat = (y_spikes_prob.ravel() > t).astype(int)
    y_hat = apply_refractory(y_hat, refractory_bins=3)

    print("\n[POST-PROCESS RESULT]")
    print("Precision:", precision_score(y_spikes_gt.ravel(), y_hat, zero_division=0))
    print("Recall   :", recall_score(y_spikes_gt.ravel(), y_hat, zero_division=0))
    print("F1       :", f1_score(y_spikes_gt.ravel(), y_hat, zero_division=0))

    # ============================== TRAIN METRICS ==============================
    print("\n===================================================")
    print("Evaluating model on TRAINING set (for comparison)...")
    print("===================================================\n")

    train_metrics_dict, y_spikes_prob_train, y_soma_pred_train, y_spikes_gt_train, y_soma_gt_train = evaluate_model_on_dataset(
        model, train_dataset, criterion_spikes, criterion_soma,
        batch_size=batch_size, verbose=1
    )

    print('--------------------------------------------------')
    print('Comparison of metrics (train vs valid)')
    print('--------------------------------------------------')

    for key in ['AUC_score', 'f1_score', 'accuracy',
                'soma_explained_variance_percent', 'soma_RMSE', 'soma_MAE']:
        train_val = train_metrics_dict[key]
        valid_val = valid_metrics_dict[key]
        print(f"{key:40s} | train: {train_val:10.4f} | valid: {valid_val:10.4f}")

    print('--------------------------------------------------')

    save_trace_dir = '/ems/elsc-labs/london-m/lior.avraham1/layer5/training_net_job_output2/trace_graphs'
    os.makedirs(save_trace_dir, exist_ok=True)

    n_samples = y_spikes_gt.shape[0] if y_spikes_gt.ndim > 1 else 1
    for i in range(n_samples):
        plot_spikes_and_voltage(
            y_spikes_gt=y_spikes_gt[i],
            y_spikes_pred=y_spikes_prob[i],
            y_soma_gt=y_soma_gt[i],
            y_soma_pred=y_soma_pred[i],
            save_dir=save_trace_dir,
            prefix=f"sim_{i}",
            thresh=valid_metrics_dict['optimal_spike_threshold']
        )

    # Print metrics summary
    print('--------------------------------------------------')
    print('valid_metrics_dict.keys():')
    print('--------------------------')
    for key in valid_metrics_dict.keys():
        print(f"  '{key}'")
    print('--------------------------------------------------')
    interesting_keys = ['requested_false_positive_rate', 'true_positive_at_FP', 'AUC_score', 'precision',
                        'recall', 'f1_score', 'soma_explained_variance_percent', 'soma_RMSE', 'soma_MAE',
                        'spikes_loss', 'soma_loss']
    for key in interesting_keys:
        if key not in valid_metrics_dict:
            print(f"[INFO] {key} not found in valid_metrics_dict (skipping)")
            continue
        start_string = f"valid_metrics_dict['{key}']"
        filler_string = ' ' * (53 - len(start_string))
        val = valid_metrics_dict[key]
        if isinstance(val, (float, int, np.floating, np.integer)):
            print(f"{start_string} {filler_string} = {float(val):.5f}")
        else:
            print(f"{start_string} {filler_string} = {val}")

    # Plot evaluation figures
    plot_evaluation_figures(valid_metrics_dict)

    # Save full model
    checkpoint_model_name_pt = model.short_name + f"_ckpt_{iter_num + 1}.pt"
    torch.save(model, os.path.join(models_folder, checkpoint_model_name_pt))
    print(f'Model saved to folder "{models_folder}"')
    print(f'checkpoint name = "{checkpoint_model_name_pt}"')

    # Load back (optional)
    load_previously_saved_model = True
    if load_previously_saved_model:
        model = torch.load(os.path.join(models_folder, checkpoint_model_name_pt), weights_only=False)
        model.eval()
        #test for inference time: 
        import time
        import torch

        model.eval()

        B = batch_size       # 8 אצלך
        C = 450
        T = valid_time_window_size  # למשל 1024

        # דוגמת קלט סינתטית (או באטצ' אמיתי אחד)
        x = torch.randn(B, C, T).to(device)

        # Warm-up (מאוד חשוב ב-GPU)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # מדידה
        n_runs = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        end = time.perf_counter()

        avg_batch_time = (end - start) / n_runs
        avg_sample_time = avg_batch_time / B

        print(f"Inference time per batch:  {avg_batch_time*1000:.2f} ms")
        print(f"Inference time per sample: {avg_sample_time*1000:.2f} ms")
        #end test 
        print(f'Model loaded from "{models_folder}"')
        print(f'checkpoint name = "{checkpoint_model_name_pt}"')

    end_time = time.perf_counter()
    runtime = (end_time - start_time) / 60
    print(f"Runtime: {round(runtime, 3)} minuts")

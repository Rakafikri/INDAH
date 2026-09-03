# ============================================================
# INDAH: Instrumental Detail Amplifier & Harmonizer  
# Training Notebook - Cell 1: Setup & Configuration
# ============================================================

import os
import sys
import time
import shutil
import warnings
import json
import inspect
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import librosa
from tqdm import tqdm
from scipy.signal import correlate
import random
from torch.utils.checkpoint import checkpoint
import bitsandbytes as bnb

warnings.filterwarnings("ignore")

# GPU Checking
print("Checking device...")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

#  Preset Checkpointing Function
def get_checkpointing_preset(preset_name="balanced"):
    presets = {
        "none": {"downsample": [False]*4, "bottleneck": False, "upsample": [False]*4, "preset": "none"},
        "minimal": {"downsample": [False]*4, "bottleneck": True, "upsample": [False]*4, "preset": "minimal"},
        "balanced": {"downsample": [False, False, False, True], "bottleneck": True, "upsample": [True, False, False, False], "preset": "balanced"},
        "aggressive": {"downsample": [False, True, True, True], "bottleneck": True, "upsample": [True, True, False, False], "preset": "aggressive"},
        "max": {"downsample": [True]*4, "bottleneck": True, "upsample": [True]*4, "preset": "max"}
    }
    return presets.get(preset_name, presets["balanced"])

PROJECT_ROOT = os.path.abspath(os.getcwd())

CONFIG = {
    # Folder Structure
    "base_dir": PROJECT_ROOT,
    "train_subdir": "train",
    "input_dir_name": "input",
    "target_dir_name": "target",
    
    # Output Folder
    "tmp_dir": os.path.join(PROJECT_ROOT, "tmp"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    
    # Training
    "max_training_hours": 50.0,
    "num_epochs": 500,
    "save_every_epochs": 1000,
    "patience": 300,
    
    # Audio
    "sample_rate": 44100,
    "chunk_duration_sec": 8,
    
    # Model
    "base_channels": 48,
    "use_stereo": True,
    
    # VRAM
    "batch_size": 3,
    "gradient_accumulation": 5,
    "mixed_precision": True,
    "learning_rate": 8e-5,
    "gradient_clip": 1,
    
    # Checkpointing Config
    "checkpointing": get_checkpointing_preset("max"),  
    
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# Make Folder Output
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["tmp_dir"], exist_ok=True)
print(f"Directories ready:")
print(f"   - Root: {PROJECT_ROOT}")
print(f"   - Input: {CONFIG['base_dir']}/{CONFIG['train_subdir']}/{CONFIG['input_dir_name']}")
print(f"   - Target: {CONFIG['base_dir']}/{CONFIG['train_subdir']}/{CONFIG['target_dir_name']}")
print(f"   - Checkpoints: {CONFIG['checkpoint_dir']}")

# Checking Dataset Folder
input_dir = Path(CONFIG["base_dir"]) / CONFIG["train_subdir"] / CONFIG["input_dir_name"]
target_dir = Path(CONFIG["base_dir"]) / CONFIG["train_subdir"] / CONFIG["target_dir_name"]

if not input_dir.exists() or not target_dir.exists():
    print(f"ERROR: Directories not found!")
    sys.exit(1)

input_files = set(f.stem for f in input_dir.glob("*.wav"))
target_files = set(f.stem for f in target_dir.glob("*.wav"))
paired_files = input_files & target_files

print(f"Dataset ready: {len(paired_files)} song pairs.")

# Save Config
config_path = f"{CONFIG['checkpoint_dir']}/config.json"
with open(config_path, "w") as f:
    json.dump(CONFIG, f, indent=2)

print(f"Setup complete! Proceed to Cell 2.")

# ============================================================
# Training Notebook - Cell 2: Model, Auto-Generate & Dataset
# ============================================================

# === INDAH Model ===

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_checkpoint=False):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.groupnorm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.groupnorm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.silu = nn.SiLU()
        self.use_checkpoint = use_checkpoint
    
    def _inner_forward(self, x):
        residual = x
        x = self.silu(self.groupnorm1(self.conv1(x)))
        x = self.groupnorm2(self.conv2(x))
        return x + residual

    def forward(self, x):
        if self.training and self.use_checkpoint and x.requires_grad:
            return checkpoint(self._inner_forward, x, use_reentrant=False)
        else:
            return self._inner_forward(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res_block = ResidualBlock(out_channels, use_checkpoint=use_checkpoint)
        self.use_checkpoint = use_checkpoint

    def _inner_forward(self, x):
        return self.res_block(self.conv(x))

    def forward(self, x):
        if self.training and self.use_checkpoint and x.requires_grad:
            return checkpoint(self._inner_forward, x, use_reentrant=False)
        else:
            return self._inner_forward(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv_after_up = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_reduce = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
        self.res_block = ResidualBlock(out_channels, use_checkpoint=use_checkpoint)
        self.use_checkpoint = use_checkpoint
    
    def _inner_forward(self, x, skip=None):
        x = self.conv_after_up(self.upsample(x))
        if skip is not None:
            x = self.conv_reduce(torch.cat([x, skip], dim=1))
        return self.res_block(x)

    def forward(self, x, skip=None):
        if self.training and self.use_checkpoint and x.requires_grad:
            return checkpoint(self._inner_forward, x, skip, use_reentrant=False)
        else:
            return self._inner_forward(x, skip)

class IndahModel(nn.Module):
    # INDAH: Residual Refiner (Output = Learned Residual)
    def __init__(self, base_channels=48, checkpointing_config=None):
        super().__init__()
        if checkpointing_config is None:
            checkpointing_config = get_checkpointing_preset("balanced")

        # Input channels = 2 (stereo)
        self.initial_conv = nn.Conv1d(2, base_channels, kernel_size=7, padding=3)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2, use_checkpoint=checkpointing_config["downsample"][0])
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4, use_checkpoint=checkpointing_config["downsample"][1])
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8, use_checkpoint=checkpointing_config["downsample"][2])
        self.down4 = DownsampleBlock(base_channels * 8, base_channels * 16, use_checkpoint=checkpointing_config["downsample"][3])

        self.bottleneck = ResidualBlock(base_channels * 16, use_checkpoint=checkpointing_config["bottleneck"])

        self.up4 = UpsampleBlock(base_channels * 16, base_channels * 8, use_checkpoint=checkpointing_config["upsample"][0])
        self.up3 = UpsampleBlock(base_channels * 8, base_channels * 4, use_checkpoint=checkpointing_config["upsample"][1])
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2, use_checkpoint=checkpointing_config["upsample"][2])
        self.up1 = UpsampleBlock(base_channels * 2, base_channels, use_checkpoint=checkpointing_config["upsample"][3])

        self.final_norm = nn.GroupNorm(8, base_channels)
        # Output channels = 2 (stereo residual)
        self.final_conv = nn.Conv1d(base_channels, 2, kernel_size=3, padding=1)
        
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.final_conv.bias, 0.0)

    def forward(self, x):
        skip1 = self.initial_conv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        x = self.bottleneck(x)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.final_conv(F.silu(self.final_norm(x)))


# === Auto-Generate Model Architecture Script ===
def generate_model_file():
    print("\n Generating model_architecture.py...")
    # Elements to be copied (ordered from the lowest dependency)
    model_elements = [
        get_checkpointing_preset,
        ResidualBlock,
        DownsampleBlock,
        UpsampleBlock,
        IndahModel
    ]
    
    model_code = ""
    for element in model_elements:
        src = inspect.getsource(element)
        model_code += textwrap.dedent(src) + "\n\n"
    
    full_code = f"""# AUTO-GENERATED MODEL ARCHITECTURE (INDAH CNN 1D)
# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
# Base Dir: {CONFIG["base_dir"]}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

{model_code}
"""
    model_path = Path(CONFIG["checkpoint_dir"]) / "model_architecture.py"
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    print(f"Saved in: {model_path}")
    return str(model_path)

MODEL_DEF_PATH = generate_model_file()


# === Rms Matching Function & Cross-Correlation Alignment ===
def match_rms(y1, y2):
    # Align RMS y2 to y1 (for every channel)
    if y1.ndim == 1: y1 = np.vstack([y1, y1])
    if y2.ndim == 1: y2 = np.vstack([y2, y2])
    
    for ch in range(y1.shape[0]):
        rms1 = np.sqrt(np.mean(y1[ch]**2))
        rms2 = np.sqrt(np.mean(y2[ch]**2))
        if rms1 < 0.001 or rms2 < 0.001:
            continue
        if rms2 > 1e-6:
            y2[ch] = y2[ch] * (rms1 / rms2)
    return y1, y2

def align_audio_by_cross_correlation(ref_audio, target_audio_to_align):
    # Use the mono version for correlation
    ref_mono = ref_audio.mean(axis=0)
    target_mono = target_audio_to_align.mean(axis=0)
    
    # Cross correlation with fft for speed
    corr = correlate(ref_mono, target_mono, mode='full', method='fft')
    delay_samples = corr.argmax() - (len(target_mono) - 1)
    
    # Apply the offset by trimming the audio
    if delay_samples > 0:
        aligned_ref = ref_audio[:, delay_samples:]
        aligned_target = target_audio_to_align
    elif delay_samples < 0:
        offset = abs(delay_samples)
        aligned_target = target_audio_to_align[:, offset:]
        aligned_ref = ref_audio
    else:
        aligned_ref = ref_audio
        aligned_target = target_audio_to_align
        
    min_len = min(aligned_ref.shape[1], aligned_target.shape[1])
    aligned_ref = aligned_ref[:, :min_len]
    aligned_target = aligned_target[:, :min_len]
    
    return aligned_ref, aligned_target

# === Generate Chunks With Chunk-Based System===
def generate_chunks_to_tmp(config, chunk_size=200):
    print("Generating aligned chunks with chunk-based system...")
    
    if os.path.exists(config["tmp_dir"]):
        shutil.rmtree(config["tmp_dir"])
    os.makedirs(config["tmp_dir"], exist_ok=True)
    
    input_dir = Path(config["base_dir"]) / config["train_subdir"] / config["input_dir_name"]
    target_dir = Path(config["base_dir"]) / config["train_subdir"] / config["target_dir_name"]
    
    all_files = sorted(set(f.stem for f in input_dir.glob("*.wav")))
    random.seed(42)
    random.shuffle(all_files)
    
    print(f"   Total file found: {len(all_files)}")
    
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx] if len(all_files) > 1 else all_files
    val_files = all_files[split_idx:] if len(all_files) > 1 else all_files[:1]
    
    all_splits = [(train_files, "train"), (val_files, "val")]
    
    for file_list, split in all_splits:
        print(f"   Processing {split} files...")
        
        chunk_counter = 0
        file_counter = 0
        
        chunk_dir_input = Path(config["tmp_dir"]) / split / "input" / f"chunk_{chunk_counter:04d}"
        chunk_dir_target = Path(config["tmp_dir"]) / split / "target" / f"chunk_{chunk_counter:04d}"
        chunk_dir_input.mkdir(parents=True, exist_ok=True)
        chunk_dir_target.mkdir(parents=True, exist_ok=True)
        
        for file_stem in tqdm(file_list, desc=f"Processing {split}", leave=False):
            input_path = input_dir / f"{file_stem}.wav"
            target_path = target_dir / f"{file_stem}.wav"
            
            if not (input_path.exists() and target_path.exists()): 
                continue
            
            inp_audio, _ = librosa.load(input_path, sr=config["sample_rate"], mono=False)
            tgt_audio, _ = librosa.load(target_path, sr=config["sample_rate"], mono=False)
            
            if inp_audio.ndim == 1: inp_audio = np.vstack([inp_audio, inp_audio])
            if tgt_audio.ndim == 1: tgt_audio = np.vstack([tgt_audio, tgt_audio])
            
            inp_audio, tgt_audio = match_rms(inp_audio, tgt_audio)
            
            try:
                aligned_inp, aligned_tgt = align_audio_by_cross_correlation(inp_audio, tgt_audio)
            except Exception as e:
                print(f"Failed to align {file_stem}: {e}")
                continue
            
            raw_chunk = int(config["chunk_duration_sec"] * config["sample_rate"])
            valid_chunk = (raw_chunk // 16) * 16
            min_samples = int(0.8 * valid_chunk)
            
            for i in range(0, aligned_inp.shape[1], valid_chunk):
                end_idx = i + valid_chunk
                if end_idx > aligned_inp.shape[1]: 
                    break
                
                # STEREO: take both channels at once
                inp_chunk = aligned_inp[:, i:end_idx]   # shape [2, samples]
                tgt_chunk = aligned_tgt[:, i:end_idx]   # shape [2, samples]
                
                # Trim remainder to keep length divisible by 16
                trim = inp_chunk.shape[1] % 16
                if trim != 0:
                    inp_chunk = inp_chunk[:, :-trim]
                    tgt_chunk = tgt_chunk[:, :-trim]
                
                if inp_chunk.shape[1] >= min_samples:
                    residual_chunk = tgt_chunk - inp_chunk   # shape [2, samples]
                    
                    # Save stereo .npy files
                    np.save(chunk_dir_input / f"sample_{file_counter:06d}.npy", inp_chunk.astype(np.float32))
                    np.save(chunk_dir_target / f"sample_{file_counter:06d}.npy", residual_chunk.astype(np.float32))
                    
                    file_counter += 1
                    
                    # Once the chunk_size has been reached, move on to the next chunk
                    if file_counter >= chunk_size:
                        chunk_counter += 1
                        file_counter = 0
                        chunk_dir_input = Path(config["tmp_dir"]) / split / "input" / f"chunk_{chunk_counter:04d}"
                        chunk_dir_target = Path(config["tmp_dir"]) / split / "target" / f"chunk_{chunk_counter:04d}"
                        chunk_dir_input.mkdir(parents=True, exist_ok=True)
                        chunk_dir_target.mkdir(parents=True, exist_ok=True)
        
        print(f"   {split}: {chunk_counter + 1} chunks created, {file_counter} files in last chunk")

# === Chunk Iterable Dataset ===
class ChunkIterableDataset(torch.utils.data.IterableDataset):
    """
    A dataset that processes one chunk at a time.
    Each iteration returns a batch of the same chunk.
    """
    def __init__(self, tmp_dir, split="train", chunk_size=200, batch_size=2, shuffle=True):
        self.tmp_dir = Path(tmp_dir)
        self.split = split
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # List of all chunk folders
        input_chunk_path = self.tmp_dir / split / "input"
        self.chunk_dirs = sorted([
            d for d in input_chunk_path.iterdir() 
            if d.is_dir() and d.name.startswith("chunk_")
        ])
        
        print(f"   {split}: {len(self.chunk_dirs)} chunks available")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Specify the chunks to be processed by this worker
        if worker_info is None:
            # Single-process data loading
            chunks_to_process = self.chunk_dirs
        else:
            # Multi-process data loading: distribute chunks among workers
            per_worker = int(np.ceil(len(self.chunk_dirs) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.chunk_dirs))
            chunks_to_process = self.chunk_dirs[start_idx:end_idx]
        
        if self.shuffle:
            random.shuffle(chunks_to_process)

        # Process each chunk in sequence
        for chunk_dir in chunks_to_process:
            chunk_id = chunk_dir.name
            target_dir = self.tmp_dir / self.split / "target" / chunk_id
            
            # Get all the files in this chunk
            input_files = sorted(chunk_dir.glob("sample_*.npy"))
            target_files = sorted(target_dir.glob("sample_*.npy"))
            
            if not input_files:
                continue
                
            # Load all the data into this chunk
            chunk_inputs = []
            chunk_targets = []
            
            for inp_file, tgt_file in zip(input_files, target_files):
                inp_data = np.load(inp_file)   # shape [2, samples]
                tgt_data = np.load(tgt_file)   # shape [2, samples]
                chunk_inputs.append(inp_data)
                chunk_targets.append(tgt_data)
            
            # Convert to tensor
            inputs_tensor = torch.stack([torch.from_numpy(x) for x in chunk_inputs])   # [N, 2, L]
            targets_tensor = torch.stack([torch.from_numpy(x) for x in chunk_targets]) # [N, 2, L]
            
            # No unsqueeze needed, already [N, 2, L]
            
            # Shuffle in chunks if necessary
            if self.shuffle:
                indices = torch.randperm(inputs_tensor.size(0))
                inputs_tensor = inputs_tensor[indices]
                targets_tensor = targets_tensor[indices]
            
            # Divide into batches
            num_samples = inputs_tensor.size(0)
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                
                batch_input = inputs_tensor[start_idx:end_idx]
                batch_target = targets_tensor[start_idx:end_idx]
                
                yield batch_input, batch_target
            
            # Clean up memory after the chunk is finished
            del inputs_tensor, targets_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __len__(self):
        # Estimated total number of batches
        total_samples = 0
        for chunk_dir in self.chunk_dirs:
            sample_files = list(chunk_dir.glob("sample_*.npy"))
            total_samples += len(sample_files)
        
        return (total_samples + self.batch_size - 1) // self.batch_size

# === Test & Generate ===
print("\nTesting residual model with configurable checkpointing...")
model = IndahModel(
    base_channels=CONFIG["base_channels"],
    checkpointing_config=CONFIG["checkpointing"]
)
sample_rate = CONFIG["sample_rate"]
chunk_sec = CONFIG["chunk_duration_sec"]
valid_length = (int(chunk_sec * sample_rate) // 16) * 16

# Dummy input stereo: [1, 2, valid_length]
dummy = torch.randn(1, 2, valid_length)
with torch.no_grad():
    out = model(dummy)
print(f"Residual model output shape: {out.shape}")  # Expect [1, 2, valid_length]
print(f"Output range: {out.min().item():.4f} to {out.max().item():.4f}")

# Generate chunks
if not os.path.exists(CONFIG["tmp_dir"]) or len(os.listdir(CONFIG["tmp_dir"])) == 0:
    print("\ntmp/ directory is empty, starting chunk generation...")
    generate_chunks_to_tmp(CONFIG, chunk_size=200)
else:
    train_input_path = Path(CONFIG["tmp_dir"]) / "train" / "input"
    val_input_path = Path(CONFIG["tmp_dir"]) / "val" / "input"
    
    train_chunks = len([d for d in train_input_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    val_chunks = len([d for d in val_input_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    
    if train_chunks > 0 and val_chunks > 0:
        print(f"\ntmp/ already contains {train_chunks} train chunks and {val_chunks} val chunks.")
    else:
        print("\nChunk-based structure not found, regenerating...")
        generate_chunks_to_tmp(CONFIG, chunk_size=200)

# ============================================================
# Training Notebook - Cell 3: Reconstruction Training
# ============================================================

# === Loss Function: Multi-Resolution STFT ===
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft_size, hop_size, win_length):
        # x shape: [batch, channels, time]
        # Reshape so channels are treated as part of the batch dimension for STFT
        # Each channel is processed independently, then losses are averaged
        batch, ch, time = x.shape
        x = x.view(batch * ch, time)  # [batch*ch, time]
        return torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, 
                          window=torch.hann_window(win_length).to(x.device),
                          return_complex=True)

    def forward(self, x_fake, x_real):
        sc_loss = 0.0
        mag_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_fake_stft = self.stft(x_fake, fft_size, hop_size, win_length)
            x_real_stft = self.stft(x_real, fft_size, hop_size, win_length)
            x_fake_mag = torch.abs(x_fake_stft)
            x_real_mag = torch.abs(x_real_stft)
            sc_loss += torch.norm(x_real_mag - x_fake_mag, p="fro") / (torch.norm(x_real_mag, p="fro") + 1e-9)
            mag_loss += F.l1_loss(torch.log(x_real_mag + 1e-9), torch.log(x_fake_mag + 1e-9))
        return (sc_loss / len(self.fft_sizes)) + (mag_loss / len(self.fft_sizes))

stft_criterion = MultiResolutionSTFTLoss().to(DEVICE)

# === CRITERION: RECONSTRUCTION LOSS ===
def criterion(pred_residual, target_residual, input_roformer):
    """
    Core idea: Input + Predicted Residual should equal the original instrumental.
    """
    # A. RECONSTRUCTION — check the final audio output
    pred_audio = input_roformer + pred_residual
    target_audio = input_roformer + target_residual 
    loss_recon_stft = stft_criterion(pred_audio, target_audio)
    
    # B. RESIDUAL DIRECT — check the raw waveform of the residual
    loss_res_l1 = F.l1_loss(pred_residual, target_residual)
    
    # C. PHASE CORRELATION — ensure correct phase direction
    pred_flat = pred_residual.reshape(pred_residual.shape[0], -1)
    tgt_flat = target_residual.reshape(target_residual.shape[0], -1)
    loss_corr = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=1).mean()
    
    # Combined loss (reconstruction dominates to control output volume)
    return loss_recon_stft + loss_res_l1 + loss_corr

print("Starting training...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === DATASET SETUP ===
print("\nLoading datasets...")
train_dataset = ChunkIterableDataset(CONFIG["tmp_dir"], split="train", chunk_size=200, batch_size=CONFIG["batch_size"], shuffle=True)
val_dataset = ChunkIterableDataset(CONFIG["tmp_dir"], split="val", chunk_size=200, batch_size=CONFIG["batch_size"], shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=0, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=0, pin_memory=True)

# === MODEL & OPTIMIZER ===
print("\nInitializing model...")
model = IndahModel(base_channels=CONFIG["base_channels"], checkpointing_config=CONFIG["checkpointing"]).to(DEVICE)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
scaler = GradScaler(enabled=CONFIG["mixed_precision"])

# === LOAD CHECKPOINT ===
start_epoch = 0
best_val_loss = float('inf')
patience_counter = 0
last_checkpoint = Path(CONFIG["checkpoint_dir"]) / "indah_last.pth"

if last_checkpoint.exists():
    print("Loading checkpoint...")
    try:
        ckpt = torch.load(last_checkpoint, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        patience_counter = 0  # reset patience as well
        if "scheduler_state_dict" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"   Resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"   Failed to load: {e}. Starting fresh.")

# === VALIDATION FUNCTION ===
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_corr = 0.0
    val_batches = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False, ncols=100)
        for inp, tgt_residual in pbar:
            inp, tgt_residual = inp.to(device), tgt_residual.to(device)
            pred_residual = model(inp)
            
            # Use reconstruction loss (requires Roformer input)
            loss = criterion(pred_residual, tgt_residual, inp)
            
            # Correlation check
            pred_flat = pred_residual.reshape(pred_residual.shape[0], -1)
            tgt_flat = tgt_residual.reshape(tgt_residual.shape[0], -1)
            batch_corr = F.cosine_similarity(pred_flat, tgt_flat, dim=1).mean()
            
            val_loss += loss.item()
            val_corr += batch_corr.item()
            val_batches += 1
            pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        pbar.close()
            
    avg_loss = val_loss / val_batches if val_batches > 0 else float('inf')
    avg_corr = val_corr / val_batches if val_batches > 0 else 0.0
    print(f"   [Validation Result] Loss: {avg_loss:.4f} | Avg Corr: {avg_corr:.4f}")
    return avg_loss, avg_corr

# === SAVE CHECKPOINT ===
def save_checkpoint(epoch, val_loss, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), 
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "config": CONFIG,
        "val_loss": val_loss
    }
    torch.save(checkpoint, f'{CONFIG["checkpoint_dir"]}/indah_last.pth')
    if is_best:
        torch.save(checkpoint, f'{CONFIG["checkpoint_dir"]}/indah_best.pth')
        print(f"Best model saved! (Loss: {val_loss:.6f})")

# === TRAINING LOOP ===
print("\nStarting training...")
start_time = time.time()
max_seconds = CONFIG["max_training_hours"] * 3600
log_file = Path(CONFIG["checkpoint_dir"]) / "training_log.txt"

for epoch in range(start_epoch, CONFIG["num_epochs"]):
    elapsed = time.time() - start_time
    if elapsed > max_seconds: break
    
    model.train()
    train_loss = 0.0
    batch_count = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False, ncols=100)
    
    for batch_idx, (inp, tgt_residual) in enumerate(pbar):
        inp, tgt_residual = inp.to(DEVICE), tgt_residual.to(DEVICE)
        
        with autocast(enabled=CONFIG["mixed_precision"]):
            pred_residual = model(inp)
            
            loss = criterion(pred_residual, tgt_residual, inp)
            
            loss = loss / CONFIG["gradient_accumulation"]
        
        scaler.scale(loss).backward()
        train_loss += loss.item() * CONFIG["gradient_accumulation"]
        batch_count += 1
        
        if (batch_idx + 1) % CONFIG["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        pbar.set_postfix({'Loss': f'{loss.item()*CONFIG["gradient_accumulation"]:.4f}'})
    pbar.close()
    
    # Validation
    avg_val_loss, avg_val_corr = validate_model(model, val_loader, DEVICE)
    
    # Scheduler step
    scheduler.step(avg_val_loss)
    
    avg_train_loss = train_loss / batch_count
    
    improvement = ""
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(epoch, avg_val_loss, is_best=True)
        improvement = "NEW BEST"
        patience_counter = 0
    else:
        save_checkpoint(epoch, avg_val_loss, is_best=False)
        patience_counter += 1
        
    log_msg = f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} (Corr: {avg_val_corr:.4f}) | {improvement}"
    print(log_msg)
    with open(log_file, "a", encoding="utf-8") as f: 
        f.write(log_msg + "\n")
    
    if patience_counter >= CONFIG["patience"]:
        print("Early stopping.")
        break

print("\nDone.")
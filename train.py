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

# === CONFIGURATION ===
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
    
    # Model (BS-Roformer specific)
    "model_params": {
        "dim": 384,
        "depth": 8,
        "num_heads": 8,
        "num_bands": 64,
        "n_fft": 2048,
        "hop_length": 512,
        "input_channels": 2,  # Stereo
    },
    "use_stereo": True,
    
    # VRAM
    "batch_size": 1,
    "gradient_accumulation": 8,
    "mixed_precision": True,
    "learning_rate": 5e-5,
    "gradient_clip": 5.0,
    
    # Metadata
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

# === Helper Functions ===

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, pos_emb):
    return (x * pos_emb.cos()) + (rotate_half(x) * pos_emb.sin())

# === ROTARY EMBEDDING ===
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, use_hyper_conn=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_hyper_conn = use_hyper_conn
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        if use_hyper_conn:
            self.mix_gate = nn.Linear(dim, num_heads) 
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, prev_v):
        B, S, D = x.shape
        skip = x
        x_norm = self.norm1(x)
        
        qkv = self.qkv(x_norm).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        current_v = v 
        if self.use_hyper_conn:
            if prev_v.dtype != v.dtype:
                prev_v = prev_v.to(dtype=v.dtype)
            mix_factor = torch.sigmoid(self.mix_gate(x_norm)).unsqueeze(-1)
            v = v + (prev_v * mix_factor)
            current_v = v 

        pos_emb = self.rope(q, S)
        q = apply_rotary_pos_emb(q, pos_emb)
        k = apply_rotary_pos_emb(k, pos_emb)
        
        # Flash Attention / SDPA
        x_attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            dropout_p=0.0 if not self.training else 0.1
        )
        x_attn = x_attn.transpose(1, 2).reshape(B, S, D)
        x = skip + self.proj(x_attn)
        x = x + self.mlp(self.norm2(x))
        
        return x, current_v

# === Output Head ===
class GLUSmoothingHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.conv_smooth = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, padding_mode='reflect')
        self.glu = nn.GLU(dim=1) 
        self.final_proj = nn.Conv1d(dim, out_dim, kernel_size=1) 

    def forward(self, x):
        # x shape: [Batch, Frames, Dim]
        x = x.transpose(1, 2) # [Batch, Dim, Frames]
        x = self.conv_smooth(x)
        x = self.glu(x)         
        x = self.final_proj(x)  
        x = x.transpose(1, 2) # [Batch, Frames, OutDim]
        return x

# === BS-Roformer Model ===
class IndahModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                "dim": 384, "depth": 8, "num_heads": 8,
                "num_bands": 64, "n_fft": 2048, "hop_length": 512,
                "input_channels": 2
            }
        
        self.dim = config["dim"]
        self.depth = config["depth"]
        self.num_heads = config["num_heads"]
        self.num_bands = config["num_bands"]
        self.n_fft = config["n_fft"]
        self.hop_length = config["hop_length"]
        self.win_length = self.n_fft
        self.input_channels = config.get("input_channels", 2)
        
        self.valid_bins = self.n_fft // 2
        self.bins_per_band = self.valid_bins // self.num_bands
        
        self.band_split = nn.Linear(self.bins_per_band * 2, self.dim)
        
        self.layers = nn.ModuleList([])
        for i in range(self.depth):
            self.layers.append(nn.ModuleList([
                TransformerBlock(self.dim, self.num_heads),  # Time
                TransformerBlock(self.dim, self.num_heads)   # Freq
            ]))
            
        self.output_head = GLUSmoothingHead(self.dim, self.bins_per_band * 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Input:  [B, C, T] where C = input_channels (stereo = 2)
        Output: [B, C, T] residual
        """
        B, C, T = x.shape
        
        # Reshape to process all channels at the same time
        # [B, C, T] → [B*C, T]
        x_flat = x.reshape(B * C, T)
        
        # 1. STFT
        window = torch.hann_window(self.win_length).to(x.device)
        stft = torch.stft(
            x_flat, n_fft=self.n_fft, hop_length=self.hop_length, 
            win_length=self.win_length, window=window, return_complex=True
        )
        stft_view = torch.view_as_real(stft)  # [B*C, Freqs+1, Frames, 2]
        stft_cut = stft_view[:, :self.valid_bins, :, :]
        
        BC, Freqs, Frames, Complex = stft_cut.shape 
        
        # 2. Band Splitting
        x_bands = stft_cut.permute(0, 2, 1, 3).reshape(BC, Frames, self.num_bands, self.bins_per_band * 2)
        x_bands = self.band_split(x_bands)
        
        # 3. Transformer Processing
        head_dim = self.dim // self.num_heads
        current_dtype = x_bands.dtype 
        
        prev_v_time = torch.zeros(BC * self.num_bands, Frames, self.num_heads, head_dim, 
                                  dtype=current_dtype, device=x.device)
        prev_v_freq = torch.zeros(BC * Frames, self.num_bands, self.num_heads, head_dim, 
                                  dtype=current_dtype, device=x.device)
        
        for time_transformer, freq_transformer in self.layers:
            # Time Domain
            x_time = x_bands.permute(0, 2, 1, 3).reshape(BC * self.num_bands, Frames, self.dim)
            if self.training:
                x_time, next_v_time = checkpoint(time_transformer, x_time, prev_v_time, use_reentrant=False)
            else:
                x_time, next_v_time = time_transformer(x_time, prev_v_time)
            prev_v_time = next_v_time 
            
            x_bands = x_time.view(BC, self.num_bands, Frames, self.dim).permute(0, 2, 1, 3)
            
            # Freq Domain
            x_freq = x_bands.reshape(BC * Frames, self.num_bands, self.dim)
            if self.training:
                x_freq, next_v_freq = checkpoint(freq_transformer, x_freq, prev_v_freq, use_reentrant=False)
            else:
                x_freq, next_v_freq = freq_transformer(x_freq, prev_v_freq)
            prev_v_freq = next_v_freq
            
            x_bands = x_freq.view(BC, Frames, self.num_bands, self.dim)
            
        # 4. Reconstruction
        x_for_head = x_bands.permute(0, 2, 1, 3).reshape(BC * self.num_bands, Frames, self.dim)
        out_bands = self.output_head(x_for_head)
        
        out_bands = out_bands.view(BC, self.num_bands, Frames, self.bins_per_band * 2)
        out_stft = out_bands.permute(0, 2, 1, 3).reshape(BC, Frames, self.valid_bins, 2)
        out_stft = out_stft.permute(0, 2, 1, 3)
        
        # Pad Nyquist bin
        out_stft = F.pad(out_stft, (0, 0, 0, 0, 0, 1))
        
        # 5. ISTFT
        out_complex = torch.complex(out_stft[..., 0], out_stft[..., 1])
        waveform = torch.istft(
            out_complex, n_fft=self.n_fft, hop_length=self.hop_length, 
            win_length=self.win_length, window=window, length=T
        )
        
        # Reshape back ke [B, C, T]
        waveform = waveform.reshape(B, C, T)
        
        return waveform


# === Auto-Generate Model Architecture ===
def generate_model_file():
    print("\nGenerating model_architecture.py...")
    
    model_elements = [
        rotate_half,
        apply_rotary_pos_emb,
        RotaryEmbedding,
        TransformerBlock,
        GLUSmoothingHead,
        IndahModel
    ]
    
    model_code = ""
    for element in model_elements:
        src = inspect.getsource(element)
        model_code += textwrap.dedent(src) + "\n\n"
    
    full_code = f"""# AUTO-GENERATED MODEL ARCHITECTURE (BS-ROFORMER)
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
    
    print(f"   Total files found: {len(all_files)}")
    
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
print("\nTesting BS-Roformer model (on GPU)...")

# 1. Move model to GPU if available
model = IndahModel(config=CONFIG["model_params"])
if DEVICE == "cuda":
    model = model.to(DEVICE)

sample_rate = CONFIG["sample_rate"]
chunk_sec = CONFIG["chunk_duration_sec"]
valid_length = (int(chunk_sec * sample_rate) // 16) * 16

# 2. Create dummy input on GPU
dummy = torch.randn(1, 2, valid_length)
if DEVICE == "cuda":
    dummy = dummy.to(DEVICE)

# 3. Run forward pass (with autocast for efficiency)
try:
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out = model(dummy)
    print(f"Residual model output shape: {out.shape}")  # Expect [1, 2, valid_length]
    print(f"Output range: {out.min().item():.4f} to {out.max().item():.4f}")
    if DEVICE == "cuda":
        print("VRAM Test Passed! Model fits in GPU memory.")
except torch.cuda.OutOfMemoryError as e:
    print(f"VRAM OOM Error: {e}")
    print("   Suggestion: Reduce 'num_bands' or 'dim' in model_params")
    sys.exit(1)

# 4. Clean up VRAM after test
del model, dummy, out
if DEVICE == "cuda":
    torch.cuda.empty_cache()

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

# === Loss Function: Hybrid (Time + Spectrogram) ===
class IndahHybridLoss(nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred, target):
        # 1. Time Domain Loss (L1)
        loss_time = F.l1_loss(pred, target)

        # 2. Multi-Resolution STFT Loss
        loss_mag = 0.0
        loss_complex = 0.0
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = n_fft
            window = torch.hann_window(win).to(pred.device)
            
            # Process all channels as batch
            pred_flat = pred.reshape(-1, pred.shape[-1])
            target_flat = target.reshape(-1, target.shape[-1])
            
            s_pred = torch.stft(pred_flat, n_fft, hop, win, window, return_complex=True)
            s_true = torch.stft(target_flat, n_fft, hop, win, window, return_complex=True)
            
            # Magnitude Loss
            loss_mag += F.l1_loss(torch.abs(s_pred), torch.abs(s_true))
            
            # Complex Loss (phase preservation)
            loss_complex += F.l1_loss(torch.view_as_real(s_pred), torch.view_as_real(s_true))

        return (20.0 * loss_time) + (loss_mag / len(self.fft_sizes)) + (loss_complex / len(self.fft_sizes))


criterion_hybrid = IndahHybridLoss().to(DEVICE)
def criterion(pred_residual, target_residual):
    return criterion_hybrid(pred_residual, target_residual)

print("Starting BS-Roformer training with Hybrid Loss (Stereo)...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Dataset Setup ===
print("\nLoading datasets...")
train_dataset = ChunkIterableDataset(CONFIG["tmp_dir"], split="train", chunk_size=200, 
                                     batch_size=CONFIG["batch_size"], shuffle=True)
val_dataset = ChunkIterableDataset(CONFIG["tmp_dir"], split="val", chunk_size=200, 
                                   batch_size=CONFIG["batch_size"], shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=0, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, num_workers=0, pin_memory=True)


# === Model & Optimizer ===
print("\nInitializing model...")
model = IndahModel(config=CONFIG["model_params"]).to(DEVICE)
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
scaler = GradScaler(enabled=CONFIG["mixed_precision"])


# === Load Checkpoint ===
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
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)  # Preserve best loss
        patience_counter = ckpt.get("patience_counter", 0)
        if "scheduler_state_dict" in ckpt: 
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"   Resuming from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
    except Exception as e:
        print(f"   Failed to load: {e}. Starting fresh.")

# === Validation Function ===
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_corr = 0.0
    val_batches = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False, ncols=100)
        for inp, tgt_residual in pbar:
            inp, tgt_residual = inp.to(device), tgt_residual.to(device)
            
            with autocast(enabled=CONFIG["mixed_precision"]):
                pred_residual = model(inp)
                loss = criterion(pred_residual, tgt_residual)
            
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

# === Save Checkpoint ===
def save_checkpoint(epoch, val_loss, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_params": CONFIG["model_params"],
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

# === Training Loop ===
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
            loss = criterion(pred_residual, tgt_residual)
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
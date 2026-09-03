# 🎵 INDAH - Instrumental Detail Amplifier & Harmonizer  
### *U-Net Stereo Residual Refiner*  

[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Branch](https://img.shields.io/badge/Branch-INDAH--Unet-orange)](../../tree/INDAH-Unet)

**INDAH** is an AI-powered audio enhancement framework that reconstructs and amplifies lost details in instrumental stems after AI separation. Built on a stereo U-Net CNN architecture with residual learning, it restores subtle frequencies, reduces artifacts, and brings instrumental tracks closer to studio-quality fidelity.

> 🔹 **This branch (`INDAH-Unet`)** implements the stereo U-Net residual refiner architecture.  
> 🔹 For other architectures, see dedicated branches in this repository.

---

## ✨ Features

### 🔊 Audio Enhancement
- **Stereo Residual Learning**: Predicts and adds the missing "detail residual" to instrumental stems
- **U-Net CNN Architecture**: Skip connections preserve fine-grained temporal & spectral information
- **Multi-Resolution STFT Loss**: Ensures perceptual quality across frequency bands
- **Sequential VRAM Management**: Loads models one-at-a-time → runs on consumer GPUs (≥6GB VRAM)

### 🎨 User Interface
- **Gradio Web UI** with NoCrypt/miku theme and easy to use
- **Dual Input Methods**: Upload local files OR download from YouTube/SoundCloud/TikTok via yt-dlp
- **Output Format Selector**: MP3 (default), WAV, FLAC, OGG, AAC — with inline quality descriptions
- **Real-time Progress Tracking**: Visual feedback for separation, enhancement, and conversion

### 🔬 Analysis & Development
- **`
analysis.ipynb`**: Modular Jupyter notebook for:
  - Alignment quality testing (RMS + cross-correlation)
  - Residual comparison (predicted vs. ground truth)
  - Objective metrics: correlation, MSE, energy ratio
  - Visualizations: waveforms, spectrograms, scatter plots
- **Auto-Generated Model Architecture**: `model_architecture.py` exported from training → guaranteed inference compatibility

### 🛠️ Deployment
- **One-Click Windows Installer**: `INDAH-installer.bat` handles Miniconda, CUDA 12.1, dependencies
- **Modular Launchers**: `run-INDAH.bat` (training) / `run-inference.bat` (Gradio UI)
- **Optional Jupyter Support**: `install-ipykernel.bat` for VS Code/notebook integration

---

## 🚀 Quick Start

### Option 1: Windows Installer (Recommended)
```bash
# 1. Download or clone this branch
git clone -b INDAH-Unet https://github.com/YOUR_USERNAME/INDAH.git
cd INDAH

# 2. Run installer (requires internet, ~10-15 min)
.\INDAH-installer.bat

# 3. Launch inference UI
.\run-inference.bat
# → Open http://localhost:7860 in your browser
```

### Option 2: Manual Install (Advanced / Linux / macOS)
```bash
# 1. Create & activate environment
conda create -n indah python=3.10 -y
conda activate indah

# 2. Install PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📦 Usage

### 🔁 Training
1. Prepare dataset:
   ```
   train/
   ├── input/     # Roformer-separated instrumentals (WAV, stereo)
   └── target/    # Clean reference instrumentals (WAV, stereo, time-aligned)
   ```
2. Launch training:
   ```bash
   .\run-INDAH.bat
   # Or manually: python train.py
   ```
3. Checkpoints saved to `checkpoints/`:
   - `indah_last.pth` → latest epoch (resumable)
   - `indah_best.pth` → lowest validation loss
   - `model_architecture.py` → auto-generated for inference compatibility

### 🎧 Inference (Gradio UI)
1. Run:
   ```bash
   .\run-inference.bat
   ```
2. Open `http://localhost:7860` in browser
3. Use either:
   - **Upload**: Select local audio file (WAV/MP3/FLAC)
   - **Download**: Paste YouTube/SoundCloud/TikTok URL → auto-download & convert
4. Select output format (MP3 recommended for sharing)
5. Click **START PROCESS** → wait for 3 stems:
   - `Vocal` (from separation)
   - `Instrumental (Raw)` (pre-enhancement)
   - `Instrumental (INDAH Enhanced)` (post-enhancement)
6. Preview, adjust volume (native player), and download stems

### 🔬 Analysis Notebook
1. Install Jupyter kernel (optional):
   ```bash
   .\install-ipykernel.bat
   ```
2. Open `analisis.ipynb` in VS Code or Jupyter Lab
3. Run cells sequentially:
   - **Cell 1**: Setup & config loading
   - **Cell 2**: Alignment tester → generates `analysis_output/` files
   - **Cell 3**: Residual comparator → evaluates model performance
4. Results saved to `analysis_output/` with plots and metrics

---

## ⚙️ Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1060 (6GB) | RTX 3060+ (12GB+) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 20 GB free | 50+ GB (for datasets) |

### Software
- **OS**: Windows 10/11 (installer), Linux, or macOS (manual install)
- **Python**: 3.10.x
- **CUDA**: 12.1 (auto-installed via PyTorch index)
- **FFmpeg**: Required for yt-dlp and format conversion  
  → Install via [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/):  
  ```powershell
  winget install ffmpeg
  ```

### Python Dependencies
See [`requirements.txt`](requirements.txt). Key packages:
```txt
torch==2.5.1+cu121
gradio==5.49.1
audio-separator[gpu]==0.39.1
yt-dlp>=2024.10.22
librosa>=0.10.0
bitsandbytes>=0.43.0  # 8-bit optimizer for VRAM efficiency
```

---

## 🗂️ Repository Structure
```
INDAH-repo/
├── train.py                      # Training pipeline (U-Net stereo)
├── inference.py                  # Gradio inference UI
├── analisis.ipynb                # Analysis notebook (alignment + residual eval)
├── INDAH-installer.bat          # Windows installer (Miniconda + CUDA)
├── install-ipykernel.bat        # Optional: Jupyter kernel installer
├── run-INDAH.bat                # Launcher: training
├── run-inference.bat            # Launcher: Gradio UI (+ yt-dlp update)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # Apache License 2.0
├── .gitignore                   # Excluded files (checkpoints, tmp, etc.)
│
├── train/
│   ├── input/                   # Roformer-separated instrumentals (dataset)
│   └── target/                  # Clean reference instrumentals (dataset)
│
├── checkpoints/                 # Model weights & architecture
│   ├── indah_best.pth          # Best model (lowest val loss)
│   ├── indah_last.pth          # Latest epoch (resumable)
│   ├── model_architecture.py   # Auto-generated for inference
│   └── config.json             # Training configuration
│
├── models/                      # Separation models (auto-downloaded)
│   └── melband_roformer_instvox_duality_v2.ckpt
│
├── downloads/                   # yt-dlp temporary storage (auto-cleaned)
├── tmp/                         # Processing temporaries (auto-cleaned)
└── analysis_output/             # Notebook analysis results (user-generated)
```

---

## 📄 License

**Non-Commercial Use Only**

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

✅ **You CAN**:
- Use for personal learning and experimentation
- Use in academic research and education  
- Fork and modify for non-commercial projects
- Study the source code

❌ **You CANNOT**:
- Use this software for commercial purposes
- Sell the software or derivative works
- Use in business/enterprise environments
- Host as a paid service

For commercial licensing inquiries, please contact the author.

Full license text: [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)** by @nomadkaraoke — for robust MDX separation backend
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — for reliable audio extraction from 1000+ sites
- **[UVR5-UI](https://github.com/Eddycrack864/UVR5-UI)** by @Eddycrack864 — for UI inspiration
- **[Mel-Band Roformer InstVoc Duality V2](https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/tree/main)** by @pcunwa for the stem separation model
- **[NoCrypt/miku Gradio theme](https://huggingface.co/spaces/NoCrypt/miku)** — for beautiful UI theme
- The open-source PyTorch, Librosa, and Hugging Face communities

---

> 💡 **Tip**: For best results, use high-quality source material (44.1kHz, 16-bit+ WAV) and ensure input/target pairs are precisely time-aligned before training.
import os
import sys
import shutil
import time
import requests
import gc
import subprocess
from pathlib import Path

import gradio as gr
import torch
import numpy as np
import librosa
import soundfile as sf
from audio_separator.separator import Separator as AudioSeparator
import yt_dlp

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TMP_DIR = PROJECT_ROOT / "tmp"
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"

for d in [CHECKPOINT_DIR, TMP_DIR, MODELS_DIR, DOWNLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INDAH_MODEL_PATH = CHECKPOINT_DIR / "indah_best.pth"
INDAH_ARCH_PATH = CHECKPOINT_DIR / "model_architecture.py"

AUDIO_SEP_MODEL = "melband_roformer_instvox_duality_v2.ckpt"
AUDIO_SEP_YAML = "config_melbandroformer_instvoc_duality.yaml"
AUDIO_SEP_OUTPUT_DIR = TMP_DIR / "audio_sep_output"

ROFORMER_PARAMS = {
    "segment_size": 2048,
    "override_model_segment_size": True,
    "overlap": 10,
    "batch_size": 10,
}

SAMPLE_RATE = 44100
CHUNK_DURATION = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FORMAT_CHOICES = [
    ("MP3 - Good quality, small file size", "MP3"),
    ("WAV - Lossless, uncompressed", "WAV"),
    ("FLAC - Lossless, compressed", "FLAC"),
    ("OGG - Open source, efficient", "OGG"),
    ("AAC - Good quality, Apple standard", "AAC")
]

# ============================================================================
# LOAD INDAH MODEL
# ============================================================================
def load_indah_model():
    if not INDAH_ARCH_PATH.exists():
        raise FileNotFoundError(f"Model architecture not found at {INDAH_ARCH_PATH}")
    
    sys.path.insert(0, str(CHECKPOINT_DIR))
    from model_architecture import IndahModel
    
    # BS-Roformer has default config embedded in __init__
    model = IndahModel()
    
    if INDAH_MODEL_PATH.exists():
        checkpoint = torch.load(INDAH_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"INDAH model weights loaded: {INDAH_MODEL_PATH.name}")
    else:
        print("WARNING: Weights not found, using initialized model.")
        
    model.to(DEVICE).eval()
    return model

# ============================================================================
# Indah Enhancement Function
# ============================================================================

def enhance_with_indah(model, audio_input: np.ndarray, sr: int, progress: gr.Progress = None) -> np.ndarray:
    """
    Enhance stereo audio using INDAH model.
    Input: audio_input shape [channels, samples] (usually 2, samples)
    Output: enhanced audio same shape.
    """
    # Ensure stereo
    if audio_input.ndim == 1:
        audio_input = np.vstack([audio_input, audio_input])
    elif audio_input.shape[0] == 1:
        audio_input = np.vstack([audio_input[0], audio_input[0]])
    
    total_samples = audio_input.shape[1]
    
    raw_chunk = int(CHUNK_DURATION * sr)
    chunk_size = (raw_chunk // 16) * 16
    
    output = np.zeros_like(audio_input)
    
    total_steps = (total_samples + chunk_size - 1) // chunk_size
    step = 0
    
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        chunk = audio_input[:, start:end]
        chunk_len = chunk.shape[1]
        
        if chunk_len < chunk_size:
            pad_width = ((0,0), (0, chunk_size - chunk_len))
            chunk = np.pad(chunk, pad_width, mode='constant')
        
        input_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            residual = model(input_tensor)
            enhanced = input_tensor + residual
        
        enhanced_np = enhanced.squeeze(0).cpu().numpy()
        output[:, start:start+chunk_len] = enhanced_np[:, :chunk_len]
        
        step += 1
        if progress:
            progress(step / total_steps, desc="Enhancing (simple)...")
    
    # Peak normalization
    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output * (0.99 / peak)
    return output



# ============================================================================
# Auto Download Model
# ============================================================================
def download_model_with_fallback():
    model_path = MODELS_DIR / AUDIO_SEP_MODEL
    yaml_path = MODELS_DIR / AUDIO_SEP_YAML
    
    # Skip download if both files already exist
    if model_path.exists() and yaml_path.exists():
        return True

    # Define sources in priority order: HuggingFace (pcunwa) → GitHub (nomadkaraoke)
    sources = [
        {
            "name": "HuggingFace (pcunwa)",
            "model": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt",
            "yaml": "https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml"
        },
        {
            "name": "GitHub (nomadkaraoke)",
            "model": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/melband_roformer_instvox_duality_v2.ckpt",
            "yaml": "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/config_melbandroformer_instvoc_duality.yaml"
        }
    ]

    # Try each source in order
    for source in sources:
        print(f" Trying {source['name']}...")
        try:
            import requests
            from tqdm import tqdm
            
            # Download model (.ckpt)
            print(f"   → Downloading {AUDIO_SEP_MODEL}...")
            response = requests.get(source["model"], stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc=AUDIO_SEP_MODEL,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            # Download config (.yaml)
            print(f"   → Downloading {AUDIO_SEP_YAML}...")
            response = requests.get(source["yaml"], stream=True, timeout=30)
            response.raise_for_status()
            with open(yaml_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Model downloaded successfully from {source['name']}!")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"{source['name']} returned 404, trying next source...")
                continue  # Try next source
            else:
                print(f"{source['name']} failed: {e}, trying next source...")
                continue
        except requests.exceptions.RequestException as e:
            print(f"{source['name']} connection error: {e}, trying next source...")
            continue
        except Exception as e:
            print(f"{source['name']} unexpected error: {e}, trying next source...")
            continue
    
    # All sources failed
    error_msg = f"""
Failed to download model from all sources.

Required files:
  • {AUDIO_SEP_MODEL}          (instvox dengan x)
  • {AUDIO_SEP_YAML}           (instvoc dengan c)

Please download manually:
  1. Open UVR5-UI: https://github.com/Eddycrack864/UVR5-UI
  2. Models tab → Download "MelBand Roformer Kim | InstVoc Duality V2 by Unwa"
  3. Copy both files to: {MODELS_DIR}

Or download directly:
  • HuggingFace (pcunwa):
    - Model: https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt
    - YAML:  https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/config_melbandroformer_instvoc_duality.yaml
  • GitHub (nomadkaraoke):
    - Model: https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/melband_roformer_instvox_duality_v2.ckpt
    - YAML:  https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/config_melbandroformer_instvoc_duality.yaml
    """.strip()
    
    print(error_msg)
    raise FileNotFoundError(error_msg)

# ============================================================================
# Download Audio Via YT-DLP
# ============================================================================
def download_audio_from_url(url: str, progress=gr.Progress()):
    if not url or not url.strip():
        raise gr.Error("Please enter a valid URL")
    
    progress(0.0, desc="Starting download...")

    def progress_hook(d):
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            if total > 0:
                percent = downloaded / total
                progress_val = 0.1 + percent * 0.8
                progress(progress_val, desc=f"Downloading... {percent*100:.1f}%")
        elif d['status'] == 'finished':
            progress(0.9, desc="Download finished, converting...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(DOWNLOADS_DIR / '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'playlist_items': '1',
        'progress_hooks': [progress_hook],
        'retries': 3,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'downloaded_audio')
            ydl.download([url])
        
        # Wait for WAV file
        wav_path = None
        for _ in range(20):
            wav_files = list(DOWNLOADS_DIR.glob("*.wav"))
            if wav_files:
                wav_path = max(wav_files, key=os.path.getmtime)
                if time.time() - wav_path.stat().st_mtime > 2:
                    break
            time.sleep(0.5)
        
        if not wav_path:
            raise FileNotFoundError("Converted WAV file not found.")
        
        progress(1.0, desc="Download complete!")
        return str(wav_path)
        
    except yt_dlp.utils.DownloadError as e:
        raise gr.Error(f"Download error: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Unexpected error: {str(e)}")

# ============================================================================
# Format Conversion Helper
# ============================================================================
def convert_stems(vocal_path, raw_path, enhanced_path, output_format, tmp_dir):
    """Convert WAV stems to selected output format using ffmpeg"""
    if output_format == "WAV":
        return vocal_path, raw_path, enhanced_path
    
    ext = output_format.lower()
    converted = []
    
    for wav_path in [vocal_path, raw_path, enhanced_path]:
        base_name = Path(wav_path).stem
        out_path = str(tmp_dir / f"{base_name}.{ext}")
        
        cmd = ["ffmpeg", "-y", "-i", wav_path]
        
        if ext == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
        elif ext == "flac":
            cmd += ["-codec:a", "flac"]
        elif ext == "ogg":
            cmd += ["-codec:a", "libvorbis", "-q:a", "6"]
        elif ext == "aac":
            cmd += ["-codec:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-codec:a", "copy"]
            
        cmd.append(out_path)
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            converted.append(out_path)
        except Exception as e:
            raise RuntimeError(f"FFmpeg conversion failed: {str(e)}\nEnsure ffmpeg is installed.")
            
    return converted

# ============================================================================
# Sequential Vram Management
# ============================================================================

def process_audio(input_path: str, output_format: str, progress=gr.Progress(track_tqdm=True)):
    try:
        progress(0.0, desc="Preparing files...")
        download_model_with_fallback()
        
        # Stage 1: Roformer Separation
        progress(0.0, desc="Loading Roformer Model...")
        use_autocast = True if DEVICE == "cuda" else False
        sep = AudioSeparator(
            output_dir=str(AUDIO_SEP_OUTPUT_DIR),
            model_file_dir=str(MODELS_DIR),
            sample_rate=SAMPLE_RATE,
            output_format="WAV",
            normalization_threshold=0.9,
            use_autocast=use_autocast,
            mdxc_params=ROFORMER_PARAMS
        )
        
        if hasattr(sep, 'model_dict'):
            if "Roformer" not in sep.model_dict: sep.model_dict["Roformer"] = {}
            sep.model_dict["Roformer"][AUDIO_SEP_MODEL] = AUDIO_SEP_YAML
        
        sep.load_model(model_filename=AUDIO_SEP_MODEL)
        
        separated_files = sep.separate(input_path)
        
        vocal_file = next(f for f in separated_files if "vocal" in f.lower())
        inst_file = next(f for f in separated_files if "instrumental" in f.lower())
        
        original_filename = Path(input_path).stem
        vocal_path = str(TMP_DIR / f"{original_filename}_vocal.wav")
        raw_path = str(TMP_DIR / f"{original_filename}_raw.wav")
        shutil.move(str(AUDIO_SEP_OUTPUT_DIR / vocal_file), vocal_path)
        shutil.move(str(AUDIO_SEP_OUTPUT_DIR / inst_file), raw_path)

        progress(1.0, desc="Separation complete. Cleaning VRAM...")
        del sep
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 2: INDAH Enhancement
        progress(0.6, desc="Checking INDAH Model...")
        
        # Check if INDAH model exists
        if not INDAH_MODEL_PATH.exists():
            # Show warning and skip enhancement
            gr.Warning("INDAH model is missing. Please ensure 'indah_best.pth' exists in the checkpoints folder to enable enhancement features.")
            
            # Convert only vocal and raw stems (skip enhanced)
            # Pass vocal_path as dummy for third arg, ignore output with _
            vocal_out, raw_out, _ = convert_stems(vocal_path, raw_path, vocal_path, output_format, TMP_DIR)
            enhanced_out = None
            
            progress(1.0, desc="Processing Complete (Enhancement Skipped)!")
            return vocal_out, raw_out, enhanced_out
        
        # Normal enhancement flow (model exists)
        progress(0.6, desc="Loading INDAH Model to VRAM...")
        indah_model = load_indah_model()
        
        progress(0.05, desc="Loading instrumental audio...")
        inst_audio, sr = librosa.load(raw_path, sr=SAMPLE_RATE, mono=False)
        
        # Call enhanced function
        enhanced = enhance_with_indah(indah_model, inst_audio, sr, progress=progress)
        
        enhanced_path = str(TMP_DIR / f"{original_filename}_enhanced.wav")
        sf.write(enhanced_path, enhanced.T, sr)

        progress(1.0, desc=f"Converting to {output_format}...")
        vocal_out, raw_out, enhanced_out = convert_stems(
            vocal_path, raw_path, enhanced_path, output_format, TMP_DIR
        )

        progress(1.0, desc="Unloading INDAH & cleaning...")
        del indah_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress(1.0, desc="Processing Complete!")
        return vocal_out, raw_out, enhanced_out
        
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"Error: {str(e)}")

# ============================================================================
# Interface
# ============================================================================
def create_ui():
    theme = "NoCrypt/miku"

    custom_css = """
    .audio_input_container {
        min-height: 120px;
    }
    .dropdown_container {
        min-height: 120px;
        display: flex;
        align-items: center;
    }
    .full_width {
        width: 100%;
    }
    """

    with gr.Blocks(theme=theme, title="INDAH v1.0", css=custom_css) as demo:
        gr.Markdown("# 🎵 INDAH - Instrumental Detail Amplifier 🎵")
        gr.Markdown("Enhance instrumental stems with AI separation and refinement.")
        with gr.Tabs():
            with gr.TabItem("Main Processing"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=350, elem_classes="audio_input_container"):
                        audio_input = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            interactive=True,
                            show_download_button=False,
                            elem_classes="full_width"
                        )
                    with gr.Column(scale=1, min_width=180, elem_classes="dropdown_container"):
                        output_format = gr.Dropdown(
                            choices=OUTPUT_FORMAT_CHOICES,
                            value="MP3",
                            label="Output Format",
                            info="Select format for all stems",
                            elem_classes="full_width"
                        )

                # Row 2: Download URL and button
                with gr.Row(equal_height=True):
                    url_input = gr.Textbox(
                        label="Video/Audio URL (YouTube, SoundCloud, etc.)",
                        lines=1,
                        max_lines=1,
                        scale=4,
                        min_width=350
                    )
                    btn_download = gr.Button(
                        "Download",
                        variant="secondary",
                        scale=1,
                        min_width=100
                    )

                with gr.Row():
                    btn_process = gr.Button(
                        "START PROCESS",
                        variant="primary",
                        size="lg",
                        elem_classes="full_width"
                    )

                gr.Markdown("### Output Stems")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Vocal")
                        audio_vocal = gr.Audio(
                            label="Vocal",
                            type="filepath",
                            interactive=False,
                            show_download_button=True
                        )
                    with gr.Column():
                        gr.Markdown("#### Instrumental (Raw)")
                        audio_raw = gr.Audio(
                            label="Instrumental Raw",
                            type="filepath",
                            interactive=False,
                            show_download_button=True
                        )
                    with gr.Column():
                        gr.Markdown("#### Instrumental (INDAH Enhanced)")
                        audio_enh = gr.Audio(
                            label="Enhanced",
                            type="filepath",
                            interactive=False,
                            show_download_button=True
                        )

                # Event Handlers
                def on_url_download(url):
                    if not url or not url.strip():
                        gr.Warning("Please enter a valid URL")
                        return gr.update()
                    try:
                        downloaded_path = download_audio_from_url(url)
                        gr.Info(f"Downloaded: {Path(downloaded_path).name}")
                        return gr.update(value=downloaded_path, visible=True)
                    except Exception as e:
                        gr.Error(f"Download failed: {str(e)}")
                        return gr.update()

                btn_download.click(
                    fn=on_url_download,
                    inputs=[url_input],
                    outputs=[audio_input],
                    show_progress="full"
                )

                btn_process.click(
                    fn=process_audio,
                    inputs=[audio_input, output_format],
                    outputs=[audio_vocal, audio_raw, audio_enh],
                    show_progress="full"
                )

            with gr.TabItem("About"):
                gr.Markdown("""
                ## INDAH - Instrumental Detail Amplifier and Harmonizer
                
                This tool combines two state-of-the-art AI models:
                
                1. **MelBand Roformer** (`melband_roformer_instvox_duality_v2`)  
                   Separates audio into high-quality vocal and instrumental stems.
                   
                2. **INDAH**  
                   A custom enhancement model that refines the instrumental stem,
                   bringing out subtle details and improving clarity.
                
                **Sequential VRAM Management**  
                The models are loaded and unloaded sequentially to minimize GPU memory usage,
                allowing the tool to run on consumer-grade hardware.
                
                **Credits**
                - python-audio-separator: [beveradb](https://github.com/beveradb)
                - yt-dlp: [yt-dlp team](https://github.com/yt-dlp/yt-dlp)
                """)

        gr.Markdown("---")
        gr.Markdown("INDAH v1.0 - Instrumental Detail Amplifier and Harmonizer")

    return demo

if __name__ == "__main__":
    print(f"Starting INDAH Gradio Interface...")
    print(f"Device: {DEVICE}")
    print(f"Separation Model: {AUDIO_SEP_MODEL}")
    print(f"Enhancement Model: {INDAH_MODEL_PATH.name if INDAH_MODEL_PATH.exists() else 'NOT FOUND'}")
    
    if not INDAH_MODEL_PATH.exists():
        print(f"Warning: INDAH model not found at {INDAH_MODEL_PATH}")
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
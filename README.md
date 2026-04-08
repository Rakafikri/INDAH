# INDAH: Instrumental Detail Amplifier & Harmonizer

![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red)

**INDAH (Instrumental Detail Amplifier & Harmonizer)** is an AI/Machine Learning framework designed for high-quality audio processing. The overarching goal of INDAH is to clean, amplify, and harmonize instrumental tracks using advanced neural network techniques.

The system aims to reconstruct and restore information and audio frequencies that are often lost or degraded after the AI stem separation process. **Designed exclusively for instrumental stems**, INDAH is trained to recover details that are missing following separation by AI models such as (for now) **MelBand Roformer Kim | InstVoc Duality V2 by Unwa**, allowing the processed instrumental audio to match a pristine, high-fidelity target.

## 🌿 Repository Structure & Branches

This repository is organized into multiple branches to keep experiments structured. The `main` branch serves as the introductory landing page. 

**Different neural network architectures and models are actively developed and maintained in their own dedicated branches.** 
Please switch to the respective architecture branches in this repository to explore the actual source code, training scripts, dependencies, and Jupyter notebooks.

## ⚙️ Prerequisites

To run the models on the architecture branches, you generally will need:
- **Python**: `3.10.x`
- **GPU**: NVIDIA GPU with CUDA support (highly recommended for accelerated training and inference).
- **Libraries**: Typically PyTorch (2.5+) and other audio processing libraries (like Librosa). *Specific dependencies are listed in the `requirements.txt` of each branch.*

## 🚀 Core Concept

- **Problem**: Separating vocals from instruments often leaves bleeding artifacts or degrades the sound quality of the instrumental, stripping away subtle frequency details.
- **Approach**: By training the AI on the difference between imperfect source tracks (post-separation) and pristine target tracks, INDAH learns a profile of the lost information. It then uses this to mathematically correct and "heal" the audio, restoring lost details and harmonies back to near-studio quality.

## 📄 License

This project includes a `LICENSE` file in the root directory. Please review it prior to using, modifying, or distributing the code.

![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red)

**INDAH (Instrumental Detail Amplifier & Harmonizer)** is an AI/Machine Learning framework designed for high-quality audio processing. The overarching goal of INDAH is to clean, amplify, and harmonize instrumental tracks using advanced neural network techniques.

The system aims to learn and predict the residual "anti-noise" required to reconstruct and dramatically improve an input audio track (such as vocal-separated stems or degraded instrumentals) to match a pristine, high-fidelity target.

## 🌿 Repository Structure & Branches

This repository is organized into multiple branches to keep experiments structured. The `main` branch serves as the introductory landing page. 

**Different neural network architectures and models are actively developed and maintained in their own dedicated branches.** 
Please switch to the respective architecture branches in this repository to explore the actual source code, training scripts, dependencies, and Jupyter notebooks.

## ⚙️ Prerequisites

To run the models on the architecture branches, you generally will need:
- **Python**: `3.10.x`
- **GPU**: NVIDIA GPU with CUDA support (highly recommended for accelerated training and inference).
- **Libraries**: Typically PyTorch (2.5+) and other audio processing libraries (like Librosa). *Specific dependencies are listed in the `requirements.txt` of each branch.*

## 🚀 Core Concept

- **Problem**: Separating vocals from instruments often leaves bleeding artifacts or degrades the sound quality of the instrumental.
- **Approach**: By training AI on the difference (the residual) between imperfect source tracks and pristine target tracks, INDAH generates the lost frequencies profile that mathematically corrects and "heals" the audio, restoring lost details and harmonies.

## 📄 License

This project includes a `LICENSE` file in the root directory. Please review it prior to using, modifying, or distributing the code.

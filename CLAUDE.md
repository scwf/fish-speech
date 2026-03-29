# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Installation
```bash
# Using conda
conda create -n fish-speech python=3.12
conda activate fish-speech
pip install -e .

# Using UV
uv sync --python 3.12
```

### Required Dependencies
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Download Models
```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## Common Commands

### Inference Options
```bash
# WebUI (Gradio interface)
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq

# HTTP API Server
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq

# Command line inference (3-step process)
# 1. Extract VQ tokens
python fish_speech/models/dac/inference.py -i "ref_audio.wav" --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"

# 2. Generate semantic tokens
python fish_speech/models/text2semantic/inference.py \
    --text "Your text here" \
    --prompt-text "Reference text" \
    --prompt-tokens "fake.npy" \
    --compile

# 3. Generate audio
python fish_speech/models/dac/inference.py -i "codes_0.npy"
```

### Performance Flags
- `--compile`: Enable torch.compile for ~10x speedup (CUDA only)
- `--half`: Use half precision for GPUs without bf16 support
- `--device cpu`: Force CPU usage

## Architecture Overview

### Core Components

**Fish Speech** is a multilingual text-to-speech system using a two-stage approach:

1. **Text-to-Semantic Model** (`fish_speech/models/text2semantic/`)
   - Based on Llama architecture with LoRA support
   - Converts text to semantic tokens
   - Handles multilingual input without phoneme dependency

2. **VQ-GAN Decoder** (`fish_speech/models/dac/`)
   - Descript Audio Codec (DAC) based neural vocoder
   - Converts semantic tokens to audio waveforms
   - Supports 16kHz audio generation

3. **Inference Engine** (`fish_speech/inference_engine/`)
   - Thread-safe queue management
   - Unified interface for both models
   - Handles reference audio processing for voice cloning

4. **Tokenizer** (`fish_speech/tokenizer.py`)
   - Custom tokenizer for semantic tokens
   - Maps between text and model representations

### Key Directories

- `fish_speech/models/text2semantic/`: LLM-based semantic generation
- `fish_speech/models/dac/`: VQ-GAN audio decoder
- `fish_speech/inference_engine/`: Unified inference interface
- `tools/`: CLI tools, web UI, and API server
- `configs/`: Training and model configurations
- `docs/`: Documentation in multiple languages

### Configuration Files

- `fish_speech/configs/base.yaml`: Base training configuration
- `fish_speech/configs/modded_dac_vq.yaml`: VQ-GAN model config
- `API_FLAGS.txt`: Default API server flags

### Model Formats

- **S1**: 4B parameter flagship model (available at fish.audio)
- **S1-mini**: 0.5B distilled model (checkpoints/openaudio-s1-mini)
- Both support voice cloning with 10-30 second reference audio

### Inference Modes

1. **WebUI**: Gradio-based browser interface
2. **API Server**: HTTP REST API
3. **Command Line**: 3-step process for batch processing
4. **GUI Client**: PyQt6 desktop application (separate repo)

## System Requirements

- **GPU**: 12GB VRAM recommended (RTX 4090 optimal)
- **System**: Linux/WSL preferred, Windows/macOS supported
- **Python**: 3.10+ required
- **Dependencies**: PyTorch 2.5+, transformers, lightning, gradio
# PyTorch Implementation of Audio-to-Audio Schrodinger Bridges

**Zhifeng Kong, Kevin J Shih, Weili Nie, Arash Vahdat, Sang-gil Lee, Joao Felipe Santos, Ante Jukic, Rafael Valle, Bryan Catanzaro**


<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2501.11311"><img src="https://img.shields.io/badge/arXiv-2501.11311-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/A2SB/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/diffusion-audio-restoration"><img src='https://img.shields.io/badge/Github-Diffusion_Audio_Restoration-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/diffusion-audio-restoration/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/diffusion-audio-restoration.svg?style=social"></a>
</div>
<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge"><img src="https://img.shields.io/badge/🤗-Checkpoints_(1_split)-ED5A22.svg" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge"><img src="https://img.shields.io/badge/🤗-Checkpoints_(2_split)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

# Overview

This repo contains the PyTorch implementation of [A2SB: Audio-to-Audio Schrodinger Bridges](https://arxiv.org/abs/2501.11311). A2SB is an audio restoration model tailored for high-res music at 44.1kHz. It is capable of both bandwidth extension (predicting high-frequency components) and inpainting (re-generating missing segments). Critically, A2SB is end-to-end without need of a vocoder to predict waveform outputs, and able to restore hour-long audio inputs. A2SB is capable of achieving state-of-the-art bandwidth extension and inpainting quality on several out-of-distribution music test sets.

- We propose A2SB, a state-of-the-art, end-to-end, vocoder-free, and multi-task diffusion Schrodinger Bridge model for 44.1kHz high-res music restoration, using an effective factorized audio representation.

- A2SB is the first long audio restoration model that could restore hour-long audio without
boundary artifacts.

---

# 🔧 Fork Modifications

This fork includes the following modifications for easier local inference:

### Changes Made:
1. **`inference/A2SB_upsample_chunked.py`** - New standalone script for chunked audio processing
   - Supports **stereo audio** (processes each channel separately, then combines)
   - **Memory-efficient chunked processing** with overlap and crossfade
   - Automatic spectral rolloff frequency detection
   - Progress bar for each chunk
   - Works with long audio files (tested on 7+ minute files)

2. **`configs/ensemble_2split_sampling.yaml`**:
   - Removed SLURM plugin for local execution
   - Changed strategy from `ddp` to `auto` for single GPU inference
   - Updated checkpoint paths to local `checkpoints/ckpt/` directory

3. **`A2SB_lightning_module_api.py`**:
   - Commented out `ssr_eval` import (not needed for inference)

4. **`plotting_utils.py`**:
   - Replaced deprecated `moviepy.video.io.bindings.mplfig_to_npimage` with custom implementation

5. **`inference/A2SB_upsample_api.py`**:
   - Fixed paths and subprocess calls for Windows compatibility

---

# 🚀 Quick Start (Inference)

## 1. Create Environment
```bash
conda create -n a2sb python=3.10 -y
conda activate a2sb
```

## 2. Install Dependencies
```bash
pip install numpy scipy matplotlib jsonargparse librosa soundfile einops rotary_embedding_torch pyyaml tqdm huggingface_hub lightning moviepy
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # For CUDA 12.8
pip install "jsonargparse[signatures]>=4.27.7"
```

## 3. Download Checkpoints
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/audio_to_audio_schrodinger_bridge', local_dir='checkpoints')"
```

## 4. Run Inference

### Chunked Processing (Recommended for long files, supports stereo)
```bash
cd inference
python A2SB_upsample_chunked.py -f "input.wav" -o "output.wav" -n 50 -c 30.0 --overlap 2.0
```

**Parameters:**
- `-f` / `--input_file` — Input audio file (mono or stereo)
- `-o` / `--output_file` — Output audio file
- `-n` / `--n_steps` — Diffusion steps (default: 50, use 25 for faster preview)
- `-c` / `--chunk_duration` — Chunk duration in seconds (default: 30)
- `--overlap` — Overlap between chunks in seconds (default: 2)
- `--checkpoint_dir` — Path to checkpoint directory (default: `../checkpoints/ckpt`)

### Original API (Single file, mono only)
```bash
cd inference
python A2SB_upsample_api.py -f "input.wav" -o "output.wav" -n 50
```

---

# Usage

## Data preparation

Prepare your data into a ```DATASET_NAME_manifest.csv``` file in the following format:
```
split,file_path,duration
train,PATH/TO/AUDIO.wav,10.0
...
validation,PATH/TO/AUDIO.wav,10.0
...
test,PATH/TO/AUDIO.wav,10.0
...
```
You could have multiple manifests, one for each dataset, and you could use different audio formats as long as ```SoundFile``` supports it. After you prepare all of them, write down their paths and names in config files under ```configs/```. 

We train our models on the permissively licensed subsets of the following datasets: FMA, Medley-Solos-DB, MUSAN, Musical Instrument, MusicNet, Slakh, FreeSound, FSD50K, GTZAN, and NSynth. 

## Training 

- For pretraining, the script is

```python main.py fit --config configs/pretrain.yaml```

- For T-finetuning, first copy the pretrained checkpoint to the T-finetune experiment folder as initialization. Then, T-finetuning resumes from this checkpoint. 

Here's an example of running T-finetuning of 2-splits. These 2 models will be trained separately. For the first split, run

```python main.py fit --config configs/t_finetune_2split_0.0_0.5.yaml```

For the second split, copy this config and modify ```model.train_t_min -> 0.5, model.train_t_max -> 1.0```, setup a different experiment name and path, and run training in a similar way. 

- Misc: you may need to adjust batch size, num devices, num nodes, and gradient accumulation in the configs based on your GPU configurations. 


## Inference

- If you would like to run inference of the entire dataset, use
```
cd inference/
python A2SB_upsample_dataset.py -dn DATASET_NAME -exp ensemble_2split_sampling -cf 4000
python A2SB_inpaint_dataset.py -dn DATASET_NAME -exp ensemble_2split_sampling -inp_len 0.3 -inp_every 5.0
```

- If you would like to run a simple bandwidth extension API for arbitrarily long audio with automatic rolloff frequency detection, use
```
cd inference/
python A2SB_upsample_api.py -f DEGRADED.wav -o RESTORED.wav -n N_STEPS
```

## Requirements

```
numpy, scipy, matplotlib, jsonargparse[signatures], librosa, soundfile, torch, torchaudio, einops, pytorch_lightning, lightning, rotary_embedding_torch, pyyaml, tqdm, huggingface_hub, moviepy
```


# Citation
```
@article{kong2025a2sb,
  title={A2SB: Audio-to-Audio Schrodinger Bridges},
  author={Kong, Zhifeng and Shih, Kevin J and Nie, Weili and Vahdat, Arash and Lee, Sang-gil and Santos, Joao Felipe and Jukic, Ante and Valle, Rafael and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2501.11311},
  year={2025}
}
```

# License/Terms of Use:
The model is provided under the NVIDIA OneWay NonCommercial License. 

The code is under [NVIDIA Source Code License - Non Commercial](https://github.com/NVlabs/I2SB/blob/master/LICENSE). Some components are adapted from other sources. The training code is adapted from [I2SB](https://github.com/NVlabs/I2SB) under the [NVIDIA Source Code License - Non Commercial](https://github.com/NVlabs/I2SB/blob/master/LICENSE). The model architecture is adapted from [Improved Diffusion](https://github.com/openai/improved-diffusion/blob/main/LICENSE) under the MIT License. 

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).


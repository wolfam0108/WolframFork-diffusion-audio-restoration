# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Modified for chunked processing with overlap
# ---------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import yaml
import torch
import torchaudio
import soundfile as sf
import librosa
from tqdm import tqdm
from collections import OrderedDict
import copy

# Import model components
from networks import AttnUNetF, SinusoidalTemporalEmbedding
from diffusion import Diffusion
from audio_transforms.transforms import apply_audio_transforms, ComplexSpectrogram, ComplexToMagInstPhase, SpectrogramDropDCTerm, PowerScaleSpectrogram, InverseComplexSpectrogram, MagInstPhaseToComplex, SpectrogramAddDCTerm, SVDFixMagInstPhase


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def compute_rolloff_freq(audio, sr, roll_percent=0.99):
    """Compute spectral rolloff frequency"""
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=roll_percent)[0]
    return int(np.mean(rolloff))


def create_lowpass_mask(n_fft, cutoff_freq, sr, device):
    """Create a lowpass mask for corruption simulation"""
    freq_bins = n_fft // 2 + 1
    frequencies = torch.linspace(0, sr / 2, freq_bins, device=device)
    mask = (frequencies <= cutoff_freq).float()
    return mask


class A2SBChunkedInference:
    def __init__(self, checkpoint_paths, t_cutoffs=[0.5], device='cuda', 
                 n_fft=2048, hop_length=512, sampling_rate=44100):
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.t_cutoffs = t_cutoffs
        
        # Build model architecture
        self.vf_model = AttnUNetF(
            n_updown_levels=5,
            in_channels=3,
            hidden_channels=[128, 256, 512, 768, 1024, 2048],
            out_channels=3,
            emb_channels=128,
            rotary_dims=16,
            band_embedding_dim=16,
            n_attn_heads=8,
            attention_levels=[3, 4],
            use_attn_input_norm=True,
            num_res_blocks=2
        )
        
        # Load checkpoints
        self.models = []
        for ckpt_path in checkpoint_paths:
            print(f"Loading checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'vf_model' in key:
                    new_key = key.replace('vf_model.', '')
                    new_state_dict[new_key] = value
            
            model = copy.deepcopy(self.vf_model)
            model.load_state_dict(new_state_dict)
            model = model.to(device)
            model.eval()
            self.models.append(model)
        
        # Diffusion scheduler
        self.ddpm = Diffusion(beta_max=1.0)
        
        # Time embedding
        self.t_to_emb = SinusoidalTemporalEmbedding(n_bands=64, min_freq=0.5).to(device)
        
        # Transforms for spectrogram processing
        self.transforms_gt = [
            ComplexSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop_length),
            ComplexToMagInstPhase(),
            SpectrogramDropDCTerm(),
            PowerScaleSpectrogram(power=0.25, channels=[0])
        ]
        
        self.inv_transforms = [
            PowerScaleSpectrogram(power=4, channels=[0]),
            SpectrogramAddDCTerm(),
            SVDFixMagInstPhase(),
            MagInstPhaseToComplex(),
            InverseComplexSpectrogram(n_fft=n_fft, win_length=n_fft, hop_length=hop_length)
        ]
    
    def get_model_for_t(self, t):
        """Get appropriate model based on time step"""
        model_idx = 0
        for idx, thresh in enumerate(self.t_cutoffs):
            if t >= thresh:
                model_idx = idx + 1
        return self.models[model_idx]
    
    def audio_to_spectrogram(self, audio):
        """Convert audio to spectrogram representation"""
        x = audio
        for transform in self.transforms_gt:
            x = transform(x)
        return x
    
    def spectrogram_to_audio(self, spec):
        """Convert spectrogram back to audio"""
        x = spec
        for transform in self.inv_transforms:
            x = transform(x)
        return x
    
    def apply_lowpass_corruption(self, spec, cutoff_freq):
        """Apply lowpass filter as corruption"""
        freq_bins = spec.shape[1]  # Height dimension (frequency)
        max_freq = self.sampling_rate / 2
        cutoff_bin = int(cutoff_freq / max_freq * freq_bins)
        
        # Create corruption mask  
        mask = torch.ones_like(spec)
        mask[:, cutoff_bin:, :] = 0  # Zero out high frequencies
        
        # Also create loss mask (inverse - where we want to generate)
        loss_mask = torch.zeros_like(spec)
        loss_mask[:, cutoff_bin:, :] = 1
        
        corrupted = spec * mask
        # Add small noise to masked region
        noise = torch.randn_like(spec) * 0.5
        corrupted = corrupted + noise * loss_mask
        
        return corrupted, loss_mask
    
    @torch.no_grad()
    def sample_ddpm(self, x_corrupted, loss_mask, n_steps=50):
        """Run DDPM sampling to restore audio"""
        t_steps = torch.linspace(1, 0.05, n_steps).to(self.device)
        
        x_t = x_corrupted.clone()
        
        for t_idx in tqdm(range(len(t_steps) - 1), desc="Sampling", leave=False):
            t = t_steps[t_idx]
            t_prev = t_steps[t_idx + 1]
            
            t_emb = self.t_to_emb(t.unsqueeze(0)).repeat(x_t.shape[0], 1)
            
            model = self.get_model_for_t(t.item())
            vf_output = model(x_t, t_emb)
            
            pred_x0 = self.ddpm.get_pred_x0(t.unsqueeze(0), x_t, vf_output)
            
            # Apply mask to pred_x0
            pred_x0 = pred_x0 * loss_mask + (1 - loss_mask) * x_corrupted
            
            x_t = self.ddpm.p_posterior(t_prev.unsqueeze(0), t.unsqueeze(0), x_t, pred_x0, ot_ode=False)
            
            # Apply mask to x_t
            xt_true = x_corrupted
            std_sb = self.ddpm.get_std_t(t_prev.unsqueeze(0))
            xt_true = xt_true + std_sb * torch.randn_like(xt_true)
            x_t = (1 - loss_mask) * xt_true + loss_mask * x_t
        
        return x_t
    
    def pad_to_multiple(self, spec, multiple=32):
        """Pad spectrogram to be multiple of given value (needed for UNet)"""
        _, h, w = spec.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            spec = torch.nn.functional.pad(spec, (0, pad_w, 0, pad_h), mode='reflect')
        return spec, h, w
    
    def unpad(self, spec, orig_h, orig_w):
        """Remove padding from spectrogram"""
        return spec[:, :orig_h, :orig_w]
    
    def process_chunk(self, audio_chunk, cutoff_freq, n_steps=50):
        """Process a single audio chunk"""
        # Convert to tensor if needed (keep on CPU for spectrogram)
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk).float()
        
        # Save original length
        original_length = len(audio_chunk)
        
        # Audio to spectrogram (on CPU)
        spec = self.audio_to_spectrogram(audio_chunk.cpu())
        
        # Pad to multiple of 32 for UNet (5 levels of downsampling = 2^5)
        spec_padded, orig_h, orig_w = self.pad_to_multiple(spec, multiple=32)
        spec_padded = spec_padded.unsqueeze(0)  # Add batch dimension
        
        # Move to GPU for model processing
        spec_padded = spec_padded.to(self.device)
        
        # Apply corruption (lowpass)
        corrupted, loss_mask = self.apply_lowpass_corruption(spec_padded, cutoff_freq)
        
        # Sample (on GPU)
        restored = self.sample_ddpm(corrupted, loss_mask, n_steps)
        
        # Move back to CPU for inverse spectrogram
        restored_cpu = restored[0].cpu()
        
        # Unpad to original size
        restored_cpu = self.unpad(restored_cpu, orig_h, orig_w)
        
        # Convert back to audio (on CPU)
        restored_audio = self.spectrogram_to_audio(restored_cpu)
        restored_audio = restored_audio.numpy()
        
        # Ensure output length matches input length
        if len(restored_audio) > original_length:
            restored_audio = restored_audio[:original_length]
        elif len(restored_audio) < original_length:
            restored_audio = np.pad(restored_audio, (0, original_length - len(restored_audio)), mode='constant')
        
        return restored_audio
    
    def process_channel(self, audio, sr, cutoff_freq, n_steps, chunk_duration, overlap_duration, channel_name=""):
        """Process a single audio channel with chunking and overlap"""
        
        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        hop_samples = chunk_samples - overlap_samples
        
        # Process in chunks
        total_samples = len(audio)
        output_audio = np.zeros(total_samples, dtype=np.float32)
        weight_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Create crossfade window
        fade_samples = overlap_samples // 2
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        start = 0
        chunk_idx = 0
        total_chunks = (total_samples - overlap_samples) // hop_samples + 1
        
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]
            
            # Pad if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            print(f"\n{channel_name} chunk {chunk_idx + 1}/{total_chunks}: {start/sr:.1f}s - {end/sr:.1f}s")
            
            # Process chunk
            restored_chunk = self.process_chunk(chunk, cutoff_freq, n_steps)
            
            # Trim to original length
            restored_chunk = restored_chunk[:end - start]
            
            # Apply crossfade weights
            chunk_weight = np.ones(len(restored_chunk))
            
            # Fade in at start (except first chunk)
            if start > 0 and fade_samples <= len(chunk_weight):
                chunk_weight[:fade_samples] = fade_in[:min(fade_samples, len(chunk_weight))]
            
            # Fade out at end (except last chunk)
            if end < total_samples and fade_samples <= len(chunk_weight):
                chunk_weight[-fade_samples:] = fade_out[-min(fade_samples, len(chunk_weight)):]
            
            # Add to output with weights
            output_audio[start:end] += restored_chunk * chunk_weight
            weight_audio[start:end] += chunk_weight
            
            start += hop_samples
            chunk_idx += 1
        
        # Normalize by weights
        weight_audio = np.maximum(weight_audio, 1e-8)
        output_audio = output_audio / weight_audio
        
        return output_audio
    
    def process_file(self, input_path, output_path, n_steps=50, 
                     chunk_duration=30.0, overlap_duration=2.0):
        """Process an audio file with chunking and overlap (supports stereo)"""
        print(f"Loading audio: {input_path}")
        
        # Load audio without mono conversion to preserve stereo
        audio, sr = librosa.load(input_path, sr=self.sampling_rate, mono=False)
        
        # Handle mono vs stereo
        if audio.ndim == 1:
            # Mono input
            n_channels = 1
            audio = audio.reshape(1, -1)
        else:
            n_channels = audio.shape[0]
        
        print(f"Audio loaded: {audio.shape[1]/sr:.1f}s @ {sr}Hz, {n_channels} channel(s)")
        
        # Compute rolloff frequency (using first channel or mono)
        cutoff_freq = compute_rolloff_freq(audio[0], sr)
        print(f"Detected rolloff frequency: {cutoff_freq} Hz")
        
        # Process each channel
        output_channels = []
        for ch in range(n_channels):
            channel_name = f"[Ch{ch+1}/{n_channels}]" if n_channels > 1 else ""
            if n_channels > 1:
                print(f"\n{'='*50}")
                print(f"Processing channel {ch+1}/{n_channels}")
                print(f"{'='*50}")
            
            output_ch = self.process_channel(
                audio[ch], sr, cutoff_freq, n_steps, 
                chunk_duration, overlap_duration, channel_name
            )
            output_channels.append(output_ch)
        
        # Stack channels
        if n_channels == 1:
            output_audio = output_channels[0]
        else:
            output_audio = np.stack(output_channels, axis=0)
            # Transpose for soundfile (samples, channels)
            output_audio = output_audio.T
        
        # Save
        print(f"\nSaving output: {output_path}")
        sf.write(output_path, output_audio, sr)
        
        return output_audio


def main():
    parser = argparse.ArgumentParser(description='A2SB Chunked Audio Upsampling')
    parser.add_argument('-f', '--input_file', type=str, required=True, help='Input audio file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output audio file')
    parser.add_argument('-n', '--n_steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('-c', '--chunk_duration', type=float, default=30.0, help='Chunk duration in seconds')
    parser.add_argument('--overlap', type=float, default=2.0, help='Overlap duration in seconds')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints/ckpt', help='Directory with checkpoints')
    args = parser.parse_args()
    
    # Checkpoint paths
    ckpt_dir = args.checkpoint_dir
    checkpoint_paths = [
        os.path.join(ckpt_dir, 'A2SB_twosplit_0.0_0.5_release.ckpt'),
        os.path.join(ckpt_dir, 'A2SB_twosplit_0.5_1.0_release.ckpt')
    ]
    
    # Check if checkpoints exist
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"Error: Checkpoint not found: {path}")
            return
    
    # Create inference engine
    print("Initializing A2SB model...")
    engine = A2SBChunkedInference(
        checkpoint_paths=checkpoint_paths,
        t_cutoffs=[0.5],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process file
    engine.process_file(
        input_path=args.input_file,
        output_path=args.output_file,
        n_steps=args.n_steps,
        chunk_duration=args.chunk_duration,
        overlap_duration=args.overlap
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Coqui XTTS v2 on Bengali language data
Uses Common Voice Bengali dataset for training
"""

import os
import sys
import gc
import re
import json
import shutil
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Fix dependency issues before importing numpy and other libraries
def setup_environment():
    """Set up the environment with correct dependencies."""
    import subprocess
    import sys
    
    # Install/reinstall packages with compatible versions
    packages = [
        "numpy==1.24.3",  # Specific version to avoid compatibility issues
        "pandas==2.0.3",
        "torch==2.0.1",
        "torchaudio==2.0.2",
        "librosa==0.10.1",
        "datasets==2.14.5",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "TTS==0.17.5"      # Coqui TTS library
    ]
    
    print("Setting up environment with compatible packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Environment setup complete.")

# Run setup before importing problematic packages
setup_environment()

# Now import the packages after setting up the environment
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import librosa.display
from datasets import load_dataset, Audio
from sklearn.model_selection import train_test_split
from IPython.display import Audio as IPythonAudio

# Import TTS after ensuring proper environment setup
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets.dataset import TTSDataset
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
class Config:
    # General settings
    output_dir = "./xtts_bengali_finetuned"
    original_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset settings
    dataset_name = "mozilla-foundation/common_voice_13_0"
    dataset_config = "bn"
    dataset_split = "train"
    streaming = True
    max_samples = 5000  # Max number of samples to use for fine-tuning
    test_size = 0.1  # 10% of data used for testing
    val_size = 0.1  # 10% of data used for validation
    min_audio_length = 1.0  # Minimum audio length in seconds
    max_audio_length = 20.0  # Maximum audio length in seconds
    target_sample_rate = 24000  # XTTS requires 24kHz
    
    # Data preparation settings
    preprocessed_dir = "./preprocessed_data"
    metadata_file = "metadata.csv"
    
    # Training settings
    batch_size = 4
    eval_batch_size = 4
    num_epochs = 10
    learning_rate = 5e-5
    save_checkpoints = True
    checkpoint_dir = "./checkpoints"
    checkpoint_interval = 1000  # Save checkpoint every N steps
    
    # Evaluation settings
    num_test_samples = 5  # Number of samples to generate for evaluation


def download_model():
    """Download the pre-trained XTTS v2 model"""
    logger.info("Downloading pre-trained XTTS v2 model...")
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(Config.original_model_name)
    return model_path, config_path


def prepare_dataset():
    """Prepare the Bengali Common Voice dataset for fine-tuning"""
    logger.info("Loading Common Voice Bengali dataset...")
    
    # Load the dataset
    cv_dataset = load_dataset(
        Config.dataset_name,
        Config.dataset_config,
        split=Config.dataset_split,
        streaming=Config.streaming
    )
    
    # Convert to audio format
    cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=Config.target_sample_rate))
    
    # Create preprocessed directory
    os.makedirs(Config.preprocessed_dir, exist_ok=True)
    
    logger.info("Preprocessing dataset...")
    metadata = []
    count = 0
    
    # Process samples
    for sample in tqdm(cv_dataset.take(Config.max_samples)):
        # Skip if no audio or sentence
        if not sample["audio"] or not sample["sentence"]:
            continue
            
        try:
            # Load and resample audio
            audio_array = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            
            # Check audio length
            duration = len(audio_array) / sample_rate
            if duration < Config.min_audio_length or duration > Config.max_audio_length:
                continue
                
            # Create a unique filename
            filename = f"sample_{count:05d}.wav"
            filepath = os.path.join(Config.preprocessed_dir, filename)
            
            # Save audio file
            if sample_rate != Config.target_sample_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=Config.target_sample_rate
                )
            
            # Convert to 16-bit PCM
            audio_array = (audio_array * 32767).astype(np.int16)
            
            # Save as WAV
            torchaudio.save(
                filepath,
                torch.tensor(audio_array).unsqueeze(0),
                Config.target_sample_rate,
                bits_per_sample=16
            )
            
            # Add to metadata
            metadata.append({
                "audio_file": filename,
                "text": sample["sentence"],
                "duration": duration
            })
            
            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count} samples")
                
        except Exception as e:
            logger.warning(f"Error processing sample: {str(e)}")
            continue
    
    logger.info(f"Total processed samples: {len(metadata)}")
    
    # Create train-val-test split
    train_meta, test_meta = train_test_split(
        metadata, 
        test_size=Config.test_size, 
        random_state=SEED
    )
    train_meta, val_meta = train_test_split(
        train_meta, 
        test_size=Config.val_size / (1 - Config.test_size), 
        random_state=SEED
    )
    
    logger.info(f"Train samples: {len(train_meta)}")
    logger.info(f"Validation samples: {len(val_meta)}")
    logger.info(f"Test samples: {len(test_meta)}")
    
    # Save metadata
    os.makedirs(os.path.join(Config.preprocessed_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(Config.preprocessed_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(Config.preprocessed_dir, "test"), exist_ok=True)
    
    # Create metadata files
    for split_name, split_data in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        split_df = pd.DataFrame(split_data)
        
        # Copy files to split directories
        for idx, row in split_df.iterrows():
            src_path = os.path.join(Config.preprocessed_dir, row["audio_file"])
            dst_path = os.path.join(Config.preprocessed_dir, split_name, row["audio_file"])
            shutil.copy(src_path, dst_path)
        
        # Save metadata
        metadata_path = os.path.join(Config.preprocessed_dir, f"{split_name}_{Config.metadata_file}")
        split_df["audio_file"] = split_df["audio_file"].apply(
            lambda x: os.path.join(split_name, x)
        )
        split_df.to_csv(metadata_path, index=False)
    
    # Remove original files
    for file in os.listdir(Config.preprocessed_dir):
        if file.endswith(".wav"):
            os.remove(os.path.join(Config.preprocessed_dir, file))
    
    return os.path.join(Config.preprocessed_dir, f"train_{Config.metadata_file}"), \
           os.path.join(Config.preprocessed_dir, f"val_{Config.metadata_file}"), \
           os.path.join(Config.preprocessed_dir, f"test_{Config.metadata_file}")


def modify_config(config_path, train_meta_path, val_meta_path):
    """Modify the XTTS config for fine-tuning"""
    logger.info("Modifying XTTS config for fine-tuning...")
    
    # Load the original config
    config = XttsConfig()
    config.load_json(config_path)
    
    # Modify config for fine-tuning
    config.audio.resample = True
    config.audio.sample_rate = Config.target_sample_rate
    
    # Dataset config
    config.datasets = [
        {
            "name": "bengali_cv",
            "path": Config.preprocessed_dir,
            "meta_file_train": train_meta_path,
            "meta_file_val": val_meta_path,
            "language": "bn"  # Bengali language code
        }
    ]
    
    # Training config
    config.trainer_params.max_epochs = Config.num_epochs
    config.trainer_params.batch_size = Config.batch_size
    config.trainer_params.eval_batch_size = Config.eval_batch_size
    config.trainer_params.gradient_clip = 1.0
    config.trainer_params.scheduler_after_epoch = True
    
    # Optimizer config
    config.optimizer_params.lr = Config.learning_rate
    config.optimizer_params.weight_decay = 0.01
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Save modified config
    modified_config_path = os.path.join(Config.output_dir, "config.json")
    config.save_json(modified_config_path)
    
    return modified_config_path, config


def fine_tune_model(model_path, config_path, config_obj):
    """Fine-tune the XTTS model on Bengali data"""
    logger.info("Starting fine-tuning...")
    
    # Create checkpoint directory
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model = Xtts.init_from_config(config_obj)
    model.load_checkpoint(config_obj, checkpoint_path=model_path, eval=False)
    model.to(Config.device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=0.01
    )
    
    # Train loop
    try:
        model.train()
        for epoch in range(Config.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{Config.num_epochs}")
            
            # Create dataset and dataloader for this epoch
            train_dataset = TTSDataset.init_from_config(config_obj, "train")
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=train_dataset.collate_fn
            )
            
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(Config.device)
                
                # Forward pass
                outputs = model.train_step(batch)
                loss = outputs["loss"]
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Save checkpoint
                if batch_idx % Config.checkpoint_interval == 0 and batch_idx > 0:
                    checkpoint_path = os.path.join(
                        Config.checkpoint_dir,
                        f"checkpoint_epoch{epoch+1}_batch{batch_idx}.pth"
                    )
                    model.save_checkpoint(config_obj, checkpoint_path)
                    logger.info(f"Checkpoint saved at {checkpoint_path}")
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_path = os.path.join(
                Config.checkpoint_dir,
                f"checkpoint_epoch{epoch+1}.pth"
            )
            model.save_checkpoint(config_obj, checkpoint_path)
            logger.info(f"Epoch checkpoint saved at {checkpoint_path}")
            
            # Evaluate on validation set
            model.eval()
            val_dataset = TTSDataset.init_from_config(config_obj, "val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.eval_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=val_dataset.collate_fn
            )
            
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_batch in tqdm(val_loader):
                    # Move batch to device
                    for k, v in val_batch.items():
                        if isinstance(v, torch.Tensor):
                            val_batch[k] = v.to(Config.device)
                    
                    # Forward pass
                    val_outputs = model.train_step(val_batch)
                    val_loss += val_outputs["loss"].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
            
            # Set back to training mode
            model.train()
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        # Save final model
        final_model_path = os.path.join(Config.output_dir, "model.pth")
        model.save_checkpoint(config_obj, final_model_path)
        logger.info(f"Final model saved at {final_model_path}")
        
        return final_model_path
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        # Try to save checkpoint on error
        error_checkpoint_path = os.path.join(Config.checkpoint_dir, "error_checkpoint.pth")
        model.save_checkpoint(config_obj, error_checkpoint_path)
        logger.info(f"Emergency checkpoint saved at {error_checkpoint_path}")
        raise


def evaluate_model(model_path, config_path, test_meta_path):
    """Evaluate the fine-tuned model on test samples"""
    logger.info("Evaluating fine-tuned model...")
    
    # Load config and model
    config = XttsConfig()
    config.load_json(config_path)
    
    # Initialize model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=model_path, eval=True)
    model.to(Config.device)
    
    # Load test metadata
    test_df = pd.read_csv(test_meta_path)
    test_samples = test_df.sample(min(Config.num_test_samples, len(test_df))).to_dict('records')
    
    # Load original model for comparison
    logger.info("Loading original model for comparison...")
    original_model_path, original_config_path, _ = ModelManager().download_model(Config.original_model_name)
    original_config = XttsConfig()
    original_config.load_json(original_config_path)
    original_model = Xtts.init_from_config(original_config)
    original_model.load_checkpoint(original_config, checkpoint_path=original_model_path, eval=True)
    original_model.to(Config.device)
    
    # Create evaluation directory
    eval_dir = os.path.join(Config.output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Set up evaluation metrics
    metrics = {
        "original": {
            "spectrograms": [],
            "audios": [],
            "mcd": []  # Mel Cepstral Distortion
        },
        "finetuned": {
            "spectrograms": [],
            "audios": [],
            "mcd": []
        },
        "reference": {
            "spectrograms": [],
            "audios": []
        }
    }
    
    # Process test samples
    for i, sample in enumerate(test_samples):
        logger.info(f"Processing test sample {i+1}/{len(test_samples)}")
        
        # Load reference audio
        ref_path = os.path.join(Config.preprocessed_dir, sample["audio_file"])
        ref_audio, sr = librosa.load(ref_path, sr=Config.target_sample_rate)
        ref_file_path = os.path.join(eval_dir, f"reference_{i}.wav")
        
        # Write reference audio (using scipy instead of deprecated librosa.output)
        from scipy.io import wavfile
        wavfile.write(ref_file_path, sr, (ref_audio * 32767).astype(np.int16))
        
        # Get reference spectrogram
        ref_mel = librosa.feature.melspectrogram(
            y=ref_audio, 
            sr=sr, 
            n_mels=80,
            fmin=0, 
            fmax=8000
        )
        ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)
        
        # Store reference data
        metrics["reference"]["spectrograms"].append(ref_mel_db)
        metrics["reference"]["audios"].append(ref_audio)
        
        # Generate with original model
        speaker_wav = ref_path
        text = sample["text"]
        
        try:
            # Original model synthesis
            with torch.no_grad():
                original_output = original_model.synthesize(
                    text,
                    speaker_wav,
                    language="bn",
                    gpt_cond_len=3,
                    speed=1.0
                )
            
            original_audio = original_output["wav"].squeeze().cpu().numpy()
            original_file_path = os.path.join(eval_dir, f"original_{i}.wav")
            
            # Write original audio
            wavfile.write(original_file_path, Config.target_sample_rate, 
                         (original_audio * 32767).astype(np.int16))
            
            # Get original spectrogram
            original_mel = librosa.feature.melspectrogram(
                y=original_audio, 
                sr=Config.target_sample_rate, 
                n_mels=80,
                fmin=0, 
                fmax=8000
            )
            original_mel_db = librosa.power_to_db(original_mel, ref=np.max)
            
            # Calculate MCD for original vs reference
            original_mcd = calculate_mcd(ref_mel, original_mel)
            
            # Store original data
            metrics["original"]["spectrograms"].append(original_mel_db)
            metrics["original"]["audios"].append(original_audio)
            metrics["original"]["mcd"].append(original_mcd)
            
            # Fine-tuned model synthesis
            with torch.no_grad():
                finetuned_output = model.synthesize(
                    text,
                    speaker_wav,
                    language="bn",
                    gpt_cond_len=3,
                    speed=1.0
                )
            
            finetuned_audio = finetuned_output["wav"].squeeze().cpu().numpy()
            finetuned_file_path = os.path.join(eval_dir, f"finetuned_{i}.wav")
            
            # Write fine-tuned audio
            wavfile.write(finetuned_file_path, Config.target_sample_rate, 
                         (finetuned_audio * 32767).astype(np.int16))
            
            # Get fine-tuned spectrogram
            finetuned_mel = librosa.feature.melspectrogram(
                y=finetuned_audio, 
                sr=Config.target_sample_rate, 
                n_mels=80,
                fmin=0, 
                fmax=8000
            )
            finetuned_mel_db = librosa.power_to_db(finetuned_mel, ref=np.max)
            
            # Calculate MCD for fine-tuned vs reference
            finetuned_mcd = calculate_mcd(ref_mel, finetuned_mel)
            
            # Store fine-tuned data
            metrics["finetuned"]["spectrograms"].append(finetuned_mel_db)
            metrics["finetuned"]["audios"].append(finetuned_audio)
            metrics["finetuned"]["mcd"].append(finetuned_mcd)
            
            # Save sample info
            with open(os.path.join(eval_dir, f"sample_{i}_info.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "text": text,
                    "original_mcd": float(original_mcd),
                    "finetuned_mcd": float(finetuned_mcd),
                    "reference_path": ref_file_path,
                    "original_path": original_file_path,
                    "finetuned_path": finetuned_file_path
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error generating sample {i}: {str(e)}")
    
    # Generate comparison plots
    generate_comparison_plots(metrics, eval_dir)
    
    # Calculate average MCD scores
    avg_original_mcd = np.mean(metrics["original"]["mcd"])
    avg_finetuned_mcd = np.mean(metrics["finetuned"]["mcd"])
    
    logger.info(f"Average MCD for original model: {avg_original_mcd:.4f}")
    logger.info(f"Average MCD for fine-tuned model: {avg_finetuned_mcd:.4f}")
    logger.info(f"MCD improvement: {avg_original_mcd - avg_finetuned_mcd:.4f}")
    
    # Save evaluation summary
    summary = {
        "avg_original_mcd": float(avg_original_mcd),
        "avg_finetuned_mcd": float(avg_finetuned_mcd),
        "mcd_improvement": float(avg_original_mcd - avg_finetuned_mcd),
        "num_samples": len(test_samples)
    }
    
    with open(os.path.join(eval_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def calculate_mcd(ref_mel, test_mel):
    """Calculate Mel Cepstral Distortion between two mel spectrograms"""
    # Convert to MFCCs (Mel Frequency Cepstral Coefficients)
    ref_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(ref_mel), n_mfcc=13)
    test_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(test_mel), n_mfcc=13)
    
    # Make sure they have the same length
    min_len = min(ref_mfcc.shape[1], test_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_len]
    test_mfcc = test_mfcc[:, :min_len]
    
    # Calculate Euclidean distance
    diff = ref_mfcc - test_mfcc
    mcd = np.sqrt(np.sum(diff**2, axis=0)).mean()
    
    return mcd


def generate_comparison_plots(metrics, eval_dir):
    """Generate comparison plots between original and fine-tuned models"""
    logger.info("Generating comparison plots...")
    
    # Create plots directory
    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate spectrogram comparisons
    for i in range(len(metrics["reference"]["spectrograms"])):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        librosa.display.specshow(
            metrics["reference"]["spectrograms"][i],
            sr=Config.target_sample_rate,
            x_axis='time',
            y_axis='mel',
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Reference Audio Spectrogram')
        
        plt.subplot(3, 1, 2)
        librosa.display.specshow(
            metrics["original"]["spectrograms"][i],
            sr=Config.target_sample_rate,
            x_axis='time',
            y_axis='mel',
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Original Model Spectrogram (MCD: {metrics["original"]["mcd"][i]:.4f})')
        
        plt.subplot(3, 1, 3)
        librosa.display.specshow(
            metrics["finetuned"]["spectrograms"][i],
            sr=Config.target_sample_rate,
            x_axis='time',
            y_axis='mel',
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Fine-tuned Model Spectrogram (MCD: {metrics["finetuned"]["mcd"][i]:.4f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"spectrogram_comparison_{i}.png"))
        plt.close()
    
    # Generate MCD comparison plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(metrics["original"]["mcd"]))
    width = 0.35
    
    plt.bar(x - width/2, metrics["original"]["mcd"], width, label='Original Model')
    plt.bar(x + width/2, metrics["finetuned"]["mcd"], width, label='Fine-tuned Model')
    
    plt.xlabel('Sample Index')
    plt.ylabel('MCD (lower is better)')
    plt.title('Mel Cepstral Distortion Comparison')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mcd_comparison.png"))
    plt.close()
    
    # Generate average MCD comparison
    avg_original_mcd = np.mean(metrics["original"]["mcd"])
    avg_finetuned_mcd = np.mean(metrics["finetuned"]["mcd"])
    
    plt.figure(figsize=(8, 6))
    
    plt.bar(["Original Model", "Fine-tuned Model"], 
            [avg_original_mcd, avg_finetuned_mcd])
    
    plt.ylabel('Average MCD (lower is better)')
    plt.title('Average Mel Cepstral Distortion')
    
    for i, v in enumerate([avg_original_mcd, avg_finetuned_mcd]):
        plt.text(i, v + 0.1, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "avg_mcd_comparison.png"))
    plt.close()


def main():
    """Main function to run the fine-tuning workflow"""
    try:
        # Setup
        logger.info("Starting XTTS v2 fine-tuning for Bengali")
        logger.info(f"Using device: {Config.device}")
        
        # Download model
        model_path, config_path = download_model()
        logger.info(f"Original model downloaded to {model_path}")
        
        # Prepare dataset
        train_meta_path, val_meta_path, test_meta_path = prepare_dataset()
        logger.info(f"Dataset prepared and split: {Config.preprocessed_dir}")
        
        # Modify config
        modified_config_path, config_obj = modify_config(config_path, train_meta_path, val_meta_path)
        logger.info(f"Modified config saved to {modified_config_path}")
        
        # Fine-tune model
        finetuned_model_path = fine_tune_model(model_path, modified_config_path, config_obj)
        logger.info(f"Fine-tuning completed. Model saved to {finetuned_model_path}")
        
        # Evaluate model
        evaluation_results = evaluate_model(finetuned_model_path, modified_config_path, test_meta_path)
        logger.info("Evaluation completed.")
        
        # Final report
        if evaluation_results["mcd_improvement"] > 0:
            logger.info("🎉 Fine-tuning was successful! The model shows improvement on Bengali data.")
        else:
            logger.info("⚠️ The fine-tuned model did not show improvement according to MCD metrics.")
            
        logger.info(f"All outputs saved to {Config.output_dir}")
        logger.info("Process completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
      

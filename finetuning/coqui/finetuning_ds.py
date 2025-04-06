"""
Bengali XTTS v2 Fine-tuning Script
Requires: Python 3.9+, torch 2.0+, CUDA 11.8+
"""

# Install requirements
# !pip install TTS torchaudio datasets evaluate soundfile matplotlib

import os
import random
import torch
import torchaudio
import soundfile as sf
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.generic_utils import get_user_data_dir
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from datasets import load_dataset, Audio
import evaluate
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RANDOM_SEED = 42
OUTPUT_PATH = "bengali_xtts"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set seed
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# 1. Dataset Preparation
# --------------------------------------------------

def prepare_dataset():
    """Load and preprocess Common Voice Bengali dataset"""
    
    print("Loading dataset...")
    dataset = load_dataset("mozilla-foundation/common_voice_16_1", "bn", split="train+validation+test", streaming=True)
    
    # Convert to local files for processing
    dataset = dataset.cast_column("audio", Audio(sampling_rate=22050))
    
    def process_example(example):
        # Text preprocessing for Bengali
        text = example["sentence"].replace("\u200c", "")  # Remove zero-width non-joiner
        text = " ".join(text.split())  # Normalize whitespace
        
        # Audio preprocessing
        audio = example["audio"]["array"]
        if audio.ndim > 1:  # Convert to mono
            audio = torch.mean(torch.tensor(audio), dim=0).numpy()
        
        # Save as WAV file
        audio_path = f"{OUTPUT_PATH}/audio_{example['id']}.wav"
        sf.write(audio_path, audio, 22050)
        
        return {"audio_path": audio_path, "text": text}

    print("Processing dataset...")
    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        batched=False
    )
    
    # Split dataset (adjust proportions as needed)
    train_dataset = processed_dataset.take(10000)
    eval_dataset = processed_dataset.skip(10000).take(1000)
    
    return train_dataset, eval_dataset

# --------------------------------------------------
# 2. Fine-tuning Setup
# --------------------------------------------------

def initialize_model():
    """Initialize XTTS model for fine-tuning"""
    
    # Download XTTS v2 files
    model_manager = ModelManager()
    model_path, config_path, _ = model_manager.download_model("xtts_v2")
    
    # Initialize config
    config = XttsConfig()
    config.load_json(config_path)
    config.optimize_wav2vec = True
    config.max_conditioning_length = 20
    
    # Initialize model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=model_path)
    model.to(device)
    
    return model

def finetune_model(model, train_dataset, eval_dataset):
    """Run fine-tuning process"""
    
    # Training parameters
    training_args = {
        "num_epochs": 10,
        "batch_size": 16,
        "eval_batch_size": 8,
        "mixed_precision": True,
        "use_wandb": False,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "eval_max_length": 60,
        "output_path": OUTPUT_PATH,
        "language": "bn",
        "run_eval_steps": 500,
        "save_step_steps": 1000,
    }
    
    # Convert HF dataset to TTS format
    train_samples = [{"audio_file": x["audio_path"], "text": x["text"], "speaker": "bengali"} for x in train_dataset]
    eval_samples = [{"audio_file": x["audio_path"], "text": x["text"], "speaker": "bengali"} for x in eval_dataset]

    # Start fine-tuning
    print("Starting fine-tuning...")
    model.finetune(
        train_samples=train_samples,
        eval_samples=eval_samples,
        config=training_args,
    )
    
    return model

# --------------------------------------------------
# 3. Evaluation
# --------------------------------------------------

def generate_samples(original_model, finetuned_model, test_samples):
    """Generate comparison samples"""
    
    results = []
    for sample in test_samples[:3]:  # Compare first 3 samples
        # Original model
        original_output = original_model.synthesize(
            sample["text"],
            speaker_wav=sample["audio_path"],
            language="bn",
        )
        
        # Finetuned model
        finetuned_output = finetuned_model.synthesize(
            sample["text"],
            speaker_wav=sample["audio_path"],
            language="bn",
        )
        
        results.append({
            "original_audio": original_output["wav"],
            "finetuned_audio": finetuned_output["wav"],
            "text": sample["text"],
            "sample_rate": original_output["sample_rate"]
        })
    
    return results

def evaluate_models(original_model, finetuned_model, eval_dataset):
    """Run comprehensive evaluation"""
    
    # Get test samples
    test_samples = [{"audio_path": x["audio_path"], "text": x["text"]} for x in eval_dataset.take(5)]
    
    # Generate comparison samples
    comparisons = generate_samples(original_model, finetuned_model, test_samples)
    
    # Save audio files
    for idx, comp in enumerate(comparisons):
        sf.write(f"{OUTPUT_PATH}/original_{idx}.wav", comp["original_audio"], comp["sample_rate"])
        sf.write(f"{OUTPUT_PATH}/finetuned_{idx}.wav", comp["finetuned_audio"], comp["sample_rate"])
    
    # Generate evaluation metrics
    asr_metric = evaluate.load("wer")
    wers = []
    for comp in comparisons:
        # Calculate Word Error Rate using ASR
        transcription = original_model.synthesize(comp["text"], speaker_wav=comp["original_audio"], language="bn")
        wer = asr_metric.compute(predictions=[transcription["text"]], references=[comp["text"]])
        wers.append(wer)
    
    print(f"Average WER: {np.mean(wers):.2f}")
    
    # Generate comparison plots
    plot_comparison(comparisons)

def plot_comparison(comparisons):
    """Generate waveform and spectrogram comparisons"""
    
    for idx, comp in enumerate(comparisons):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original waveform
        axes[0,0].set_title("Original Waveform")
        axes[0,0].plot(comp["original_audio"])
        
        # Finetuned waveform
        axes[0,1].set_title("Finetuned Waveform")
        axes[0,1].plot(comp["finetuned_audio"])
        
        # Original spectrogram
        S = np.abs(librosa.stft(comp["original_audio"]))
        axes[1,0].set_title("Original Spectrogram")
        axes[1,0].imshow(librosa.amplitude_to_db(S, ref=np.max), aspect='auto', origin='lower')
        
        # Finetuned spectrogram
        S = np.abs(librosa.stft(comp["finetuned_audio"]))
        axes[1,1].set_title("Finetuned Spectrogram")
        axes[1,1].imshow(librosa.amplitude_to_db(S, ref=np.max), aspect='auto', origin='lower')
        
        plt.savefig(f"{OUTPUT_PATH}/comparison_{idx}.png")
        plt.close()

# --------------------------------------------------
# Main Workflow
# --------------------------------------------------

if __name__ == "__main__":
    # Prepare data
    train_dataset, eval_dataset = prepare_dataset()
    
    # Initialize models
    original_model = initialize_model()
    finetuned_model = initialize_model()  # Fresh copy for fine-tuning
    
    # Fine-tune
    finetuned_model = finetune_model(finetuned_model, train_dataset, eval_dataset)
    
    # Save model
    finetuned_model.save_checkpoint(f"{OUTPUT_PATH}/final_model.pth")
    
    # Evaluate
    evaluate_models(original_model, finetuned_model, eval_dataset)
    
    print("Training and evaluation complete!")
#!/usr/bin/env python3
"""
Compute pseudo-log-likelihood scores for protein sequences using ESM2 models and finetuned models.
PLL scores indicate how well different models can predict masked amino acids in the sequence.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
import time
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import gc

# Import project utilities
from pathlib import Path

def get_pll_project_root() -> Path:
    """Get the actual project root directory for PLL module."""
    # From CART/src/modeling/pll.py, go up 3 levels to reach CART-Project root
    return Path(__file__).parent.parent.parent.parent.resolve()


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Compute pseudo-log-likelihoods for CAR-T sequences")
    
    # Required arguments with default paths
    parser.add_argument(
        "--mutant_fasta", 
        type=Path, 
        default=Path('output/mutants/CAR_mutants.fasta'),
        help="Path to mutant FASTA file, relative to project root"
    )
    parser.add_argument(
        "--pretrained", 
        type=str, 
        default="facebook/esm2_t6_8M_UR50D", 
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--finetuned_high", 
        type=Path, 
        default=Path('output/models/high'), 
        help="Path to high-diversity fine-tuned model"
    )
    parser.add_argument(
        "--finetuned_low", 
        type=Path, 
        default=Path('output/models/low'), 
        help="Path to low-diversity fine-tuned model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cuda", "mps", "cpu"], 
        default="auto", 
        help="Compute device to use"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=512, 
        help="Maximum number of tokens per sequence"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path('output/results'), 
        help="Directory to save results"
    )
    parser.add_argument(
        "--use_subset", 
        action="store_true", 
        help="Use a subset of sequences for testing"
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=50, 
        help="Number of sequences to use for testing"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from previous checkpoint if available"
    )
    
    # Only parse command line arguments if this module is run directly
    if args_list is None and __name__ == "__main__":
        return parser.parse_args()
    else:
        # When imported, use the provided args_list or an empty list
        return parser.parse_args(args_list or [])


def select_device(choice: str) -> torch.device:
    """Select the best available device for computation with enhanced MPS support."""
    if choice == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🍎 Using Apple Metal Performance Shaders (MPS)")
            print("   Optimized for Apple Silicon (M1/M2/M3)")
            # Set MPS memory fraction to avoid OOM
            try:
                # Enable memory efficient attention for MPS
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                print("   MPS memory optimization enabled")
            except:
                pass
            return device
        else:
            device = torch.device("cpu")
            print("💻 Using CPU (consider using MPS on Apple Silicon)")
            return device
    else:
        device = torch.device(choice)
        if choice == "mps":
            print("🍎 Using Apple Metal Performance Shaders (MPS)")
            print("   Manually selected MPS device")
            # Set MPS optimizations
            try:
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                print("   MPS memory optimization enabled")
            except:
                pass
        elif choice == "cuda":
            print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'Unknown'}")
        else:
            print(f"💻 Using device: {choice}")
        return device


def compute_pll_optimized(sequence, model, tokenizer, device, max_tokens):
    """Compute pseudo-log-likelihood for a single sequence with MPS optimizations."""
    # Tokenize the sequence
    enc = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_tokens
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    L = input_ids.size(1)
    
    # Skip special tokens (CLS, SEP, etc.)
    start_idx = 1 if tokenizer.cls_token_id is not None else 0
    end_idx = L - 1 if tokenizer.sep_token_id is not None else L
    
    log_probs = []
    
    # MPS-specific memory management
    if device.type == "mps":
        # Process in smaller chunks for MPS to avoid memory issues
        chunk_size = min(50, end_idx - start_idx)  # Process max 50 positions at once
    else:
        chunk_size = end_idx - start_idx
    
    # Batch process multiple masked positions for efficiency
    with torch.no_grad():
        for chunk_start in range(start_idx, end_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_idx)
            
            for i in range(chunk_start, chunk_end):
                # Skip if it's already a special token
                if input_ids[0, i] in [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]:
                    continue
                    
                masked = input_ids.clone()
                masked[0, i] = tokenizer.mask_token_id
                
                try:
                    outputs = model(input_ids=masked, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    log_soft = torch.log_softmax(logits[0, i], dim=-1)
                    true_id = input_ids[0, i]
                    log_p = log_soft[true_id].item()
                    log_probs.append(log_p)
                    
                except RuntimeError as e:
                    if "MPS" in str(e) or "out of memory" in str(e).lower():
                        print(f"⚠️  MPS memory issue at position {i}, skipping...")
                        continue
                    else:
                        raise e
            
            # Clear MPS cache after each chunk
            if device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except:
                    pass
    
    return float(np.mean(log_probs)) if log_probs else 0.0


def load_model_and_tokenizer(model_path, device):
    """Load model and tokenizer with MPS optimizations for Apple Silicon."""
    try:
        print(f"Loading model from {model_path}")
        
        # MPS-specific loading optimizations
        if device.type == "mps":
            print("🍎 Applying MPS optimizations...")
            # Set environment variables for better MPS performance
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
        if os.path.exists(str(model_path)):
            # Local model path
            model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32 if device.type == "mps" else torch.float16,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Hugging Face model
            model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32 if device.type == "mps" else torch.float16,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Move model to device with MPS considerations
        if device.type == "mps":
            print("   Moving model to MPS device...")
            # For MPS, we need to be careful about memory
            model = model.to(device)
            # Force garbage collection after moving to MPS
            gc.collect()
        else:
            model = model.to(device)
            
        model.eval()
        
        # Optimize model configuration
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            
        # MPS-specific optimizations
        if device.type == "mps":
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            print("   MPS model optimizations applied")
            
        print(f"✅ Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        if device.type == "mps":
            print("💡 Try using --device cpu if MPS causes issues")
        return None, None


def save_checkpoint(results, model_name, completed_idx, total_count, output_dir):
    """Save progress checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_name": model_name,
        "results": results,
        "completed": completed_idx,
        "total": total_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    checkpoint_file = output_dir / f"checkpoint_{model_name}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"[CHECKPOINT] Saved progress: {completed_idx}/{total_count} sequences")


def load_checkpoint(model_name, output_dir):
    """Load previous checkpoint if available."""
    checkpoint_file = Path(output_dir) / f"checkpoint_{model_name}.json"
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            
            print(f"[RESUME] Found checkpoint: {checkpoint['completed']}/{checkpoint['total']} sequences completed")
            return checkpoint["results"], checkpoint["completed"]
        except Exception as e:
            print(f"[WARNING] Error loading checkpoint: {e}")
    
    return [], 0


def plot_pll_results(results_dict, output_dir):
    """Create comprehensive PLL analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(results_dict.keys())
    avg_plls = [np.nanmean(results_dict[model]) for model in models]
    perplexities = [np.exp(-pll) for pll in avg_plls]
    
    # Save detailed results
    seq_data = {}
    max_len = max(len(scores) for scores in results_dict.values())
    
    for model, scores in results_dict.items():
        # Pad with NaN to ensure equal length
        padded_scores = scores + [np.nan] * (max_len - len(scores))
        seq_data[model] = padded_scores
    
    # Save sequence-level scores
    seq_df = pd.DataFrame(seq_data)
    seq_df.to_csv(output_dir / "pll_scores.csv", index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame({
        'model': models,
        'avg_pll': avg_plls,
        'perplexity': perplexities,
        'min_pll': [np.nanmin(results_dict[model]) for model in models],
        'max_pll': [np.nanmax(results_dict[model]) for model in models],
        'std_pll': [np.nanstd(results_dict[model]) for model in models],
        'count': [len([x for x in results_dict[model] if not np.isnan(x)]) for model in models]
    })
    summary_df.to_csv(output_dir / "pll_summary.csv", index=False)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PLL scores
    bars1 = ax1.bar(models, avg_plls, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_title('Average Pseudo-Log-Likelihood Scores')
    ax1.set_ylabel('Average PLL Score')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Perplexity
    bars2 = ax2.bar(models, perplexities, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_title('Perplexity (lower is better)')
    ax2.set_ylabel('Perplexity')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pll_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Results saved to {output_dir}")


def run_pll(args):
    """Run optimized PLL computation pipeline."""
    # Resolve paths using correct project root
    root = get_pll_project_root()
    mutant_fasta = args.mutant_fasta if args.mutant_fasta.is_absolute() else root / args.mutant_fasta
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    
    # Select device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Load sequences
    sequences = []
    sequence_ids = []
    for record in SeqIO.parse(mutant_fasta, "fasta"):
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)
    
    # Use subset if requested
    if args.use_subset:
        sequences = sequences[:args.subset_size]
        sequence_ids = sequence_ids[:args.subset_size]
        print(f"Using subset of {len(sequences)} sequences for testing")
    else:
        print(f"Processing {len(sequences)} sequences")
    
    # Model configurations
    model_configs = {
        'pretrained': args.pretrained,
        'finetuned_high': args.finetuned_high,
        'finetuned_low': args.finetuned_low
    }
    
    pll_results = {}
    
    for model_name, model_path in model_configs.items():
        print(f"\n[INFO] Processing with {model_name} model")
        
        # Load checkpoint if resuming
        results, start_idx = ([], 0)
        if args.resume:
            results, start_idx = load_checkpoint(model_name, output_dir)
        
        if start_idx >= len(sequences):
            print(f"[INFO] {model_name} already completed")
            pll_results[model_name] = results
            continue
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        if model is None:
            print(f"[ERROR] Failed to load {model_name}, skipping...")
            continue
        
        # Process sequences
        try:
            for i in tqdm(range(start_idx, len(sequences)), 
                         desc=f"Computing PLL for {model_name}",
                         initial=start_idx, total=len(sequences)):
                
                pll = compute_pll_optimized(
                    sequences[i], model, tokenizer, device, args.max_tokens
                )
                results.append(pll)
                
                # Save checkpoint every 50 sequences
                if (i + 1) % 50 == 0:
                    save_checkpoint(results, model_name, i + 1, len(sequences), output_dir)
            
            pll_results[model_name] = results
            
            # Save final results
            output_path = output_dir / f"{model_name}_pll.npy"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_path, results)
            
            # Save as CSV
            csv_path = output_dir / f"{model_name}_pll.csv"
            pll_df = pd.DataFrame({
                'sequence_id': sequence_ids[:len(results)],
                'pll_score': results
            })
            pll_df.to_csv(csv_path, index=False)
            
            print(f"[INFO] Saved {model_name} results: {len(results)} sequences")
            
        except Exception as e:
            print(f"[ERROR] Error processing {model_name}: {e}")
            continue
        finally:
            # Enhanced memory cleanup with MPS support
            print(f"🧹 Cleaning up memory for {model_name}...")
            del model, tokenizer
            
            # Device-specific memory cleanup
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   CUDA cache cleared")
            elif device.type == "mps":
                try:
                    torch.mps.empty_cache()
                    print("   MPS cache cleared")
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
            print("   Garbage collection completed")
    
    # Generate comparison plots
    if pll_results:
        plot_pll_results(pll_results, output_dir)
        print(f"\n[SUCCESS] PLL analysis completed. Results saved to {output_dir}")
    else:
        print("[ERROR] No results generated")


def main():
    args = parse_args()
    run_pll(args)


if __name__ == "__main__":
    main()
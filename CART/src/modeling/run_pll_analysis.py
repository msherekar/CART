#!/usr/bin/env python3
"""
Standalone script to run PLL (Pseudo-Log-Likelihood) analysis.

This script runs PLL analysis separately from the main pipeline since it's computationally expensive.
Use this when you want to analyze model performance on sequence prediction tasks.

Usage:
    python run_pll_analysis.py --help
    python run_pll_analysis.py --use_subset  # Quick test with 50 sequences
    python run_pll_analysis.py --resume      # Resume from checkpoint
"""

import sys
from CART.src.modeling.pll import parse_args, run_pll

def main():
    """Run PLL analysis with command line arguments."""
    print("=" * 60)
    print("CAR-T PLL (Pseudo-Log-Likelihood) Analysis")
    print("=" * 60)
    print()
    print("This analysis computes how well different models predict")
    print("masked amino acids in CAR-T sequences.")
    print()
    print("⚠️  WARNING: This is computationally intensive!")
    print("   - Use --use_subset for quick testing")
    print("   - Use --resume to continue from checkpoints")
    print("   - Full analysis may take several hours")
    print()
    
    # Check for Apple Silicon and MPS availability
    import torch
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected with MPS support!")
        print("   - MPS will significantly speed up computation")
        print("   - Optimized memory management for M1/M2/M3")
        print("   - Use --device mps to force MPS usage")
        print()
    elif "arm64" in str(torch.version.__version__) or "darwin" in str(torch.version.__version__):
        print("🍎 Apple Silicon detected but MPS not available")
        print("   - Consider updating PyTorch for MPS support")
        print("   - pip install torch torchvision torchaudio")
        print()
    
    # Parse arguments - pass sys.argv[1:] to get command line arguments
    args = parse_args(sys.argv[1:])
    
    # Show configuration
    print("Configuration:")
    print(f"  Input FASTA: {args.mutant_fasta}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Use subset: {args.use_subset}")
    if args.use_subset:
        print(f"  Subset size: {args.subset_size}")
    print(f"  Resume from checkpoint: {args.resume}")
    print()
    
    # Performance estimates for different devices
    if args.use_subset:
        print("⏱️  Estimated time for subset:")
        print(f"   - MPS (Apple Silicon): ~5-15 minutes")
        print(f"   - CUDA GPU: ~3-10 minutes") 
        print(f"   - CPU: ~15-45 minutes")
    else:
        print("⏱️  Estimated time for full analysis:")
        print(f"   - MPS (Apple Silicon): ~1-3 hours")
        print(f"   - CUDA GPU: ~30min-2 hours")
        print(f"   - CPU: ~4-12 hours")
    print()
    
    # Confirm before running
    if not args.use_subset:
        response = input("Run full PLL analysis? This may take hours. (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Run PLL analysis
    try:
        run_pll(args)
        print("\n✅ PLL analysis completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
        print("💾 Progress has been saved. Use --resume to continue.")
        
    except Exception as e:
        print(f"\n❌ Error during PLL analysis: {e}")
        if "MPS" in str(e):
            print("💡 MPS error detected. Try:")
            print("   - python run_pll_analysis.py --device cpu")
            print("   - Reduce --subset_size if using subset")
        print("💾 Check if partial results were saved for resuming.")

if __name__ == "__main__":
    main() 
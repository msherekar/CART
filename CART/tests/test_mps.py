#!/usr/bin/env python3
"""
Test script to check MPS (Metal Performance Shaders) availability and performance on Apple Silicon.
Run this before using PLL analysis to ensure MPS is working correctly.
"""

import torch
import time
import sys

def test_mps_availability():
    """Test if MPS is available and working."""
    print("🍎 Apple Silicon MPS Test")
    print("=" * 40)
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        if sys.platform == "darwin":
            print("💡 You're on macOS but MPS is not available.")
            print("   Try updating PyTorch:")
            print("   pip install torch torchvision torchaudio")
        else:
            print("💡 MPS is only available on Apple Silicon Macs")
        return False
    
    print("✅ MPS is available!")
    
    # Test basic MPS operations
    try:
        print("\n🧪 Testing basic MPS operations...")
        
        # Create tensors on MPS
        device = torch.device("mps")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Test matrix multiplication
        start_time = time.time()
        z = torch.mm(x, y)
        mps_time = time.time() - start_time
        
        print(f"✅ Matrix multiplication successful: {mps_time:.4f}s")
        
        # Compare with CPU
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        speedup = cpu_time / mps_time
        print(f"🚀 MPS speedup: {speedup:.2f}x faster than CPU")
        
        # Test memory cleanup
        del x, y, z, x_cpu, y_cpu, z_cpu
        torch.mps.empty_cache()
        print("✅ Memory cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        print("💡 Try using CPU instead: --device cpu")
        return False

def test_transformers_mps():
    """Test if transformers work with MPS."""
    try:
        print("\n🤖 Testing transformers with MPS...")
        from transformers import AutoTokenizer, AutoModel
        
        # Load a small model for testing
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move to MPS
        device = torch.device("mps")
        model = model.to(device)
        
        # Test inference
        text = "This is a test sequence for MPS."
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Transformers work with MPS!")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        torch.mps.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers MPS test failed: {e}")
        print("💡 This might affect PLL analysis performance")
        return False

def main():
    """Run all MPS tests."""
    print("Testing MPS setup for PLL analysis...\n")
    
    # Test basic MPS
    mps_basic = test_mps_availability()
    
    if mps_basic:
        # Test transformers with MPS
        mps_transformers = test_transformers_mps()
        
        print("\n" + "=" * 40)
        print("📊 Test Results:")
        print(f"   Basic MPS: {'✅ PASS' if mps_basic else '❌ FAIL'}")
        print(f"   Transformers MPS: {'✅ PASS' if mps_transformers else '❌ FAIL'}")
        
        if mps_basic and mps_transformers:
            print("\n🎉 MPS is ready for PLL analysis!")
            print("   Recommended command:")
            print("   python run_pll_analysis.py --device mps --use_subset")
        elif mps_basic:
            print("\n⚠️  Basic MPS works but transformers may have issues")
            print("   Try PLL analysis with MPS, fallback to CPU if needed")
        else:
            print("\n❌ MPS has issues, use CPU instead")
            print("   python run_pll_analysis.py --device cpu --use_subset")
    else:
        print("\n❌ MPS not available, use CPU")
        print("   python run_pll_analysis.py --device cpu --use_subset")

if __name__ == "__main__":
    main() 
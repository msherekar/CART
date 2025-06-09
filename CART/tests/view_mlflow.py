#!/usr/bin/env python3
"""
Simple script to launch the MLflow UI.
Run with:
  python view_mlflow.py
"""
import subprocess
import os

def main():
    port = 5000
    print(f"[INFO] Starting MLflow UI. View the results at http://localhost:{port}")
    print("[INFO] Press Ctrl+C to stop the MLflow UI")
    try:
        # Find the mlruns directory
        mlruns_dir = "./mlruns"
        if not os.path.exists(mlruns_dir):
            # Check in the parent directory
            parent_mlruns = "../mlruns"
            if os.path.exists(parent_mlruns):
                mlruns_dir = parent_mlruns
        
        if os.path.exists(mlruns_dir):
            print(f"[INFO] Using MLflow data from: {os.path.abspath(mlruns_dir)}")
        else:
            print("[WARNING] Could not find mlruns directory. MLflow UI may not show any data.")
        
        subprocess.run(["mlflow", "ui", "--port", str(port)], check=True)
    except KeyboardInterrupt:
        print("\n[INFO] MLflow UI stopped")
    except FileNotFoundError:
        print("\n[ERROR] MLflow not found. Please install with: pip install mlflow")

if __name__ == "__main__":
    main() 
"""
Isolated benchmark runner that bypasses Intel Fortran Control-C issues
Runs the optimization in a separate process with proper signal isolation
"""
import subprocess
import sys
from pathlib import Path
import time

# Get the script directory
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

def run_benchmark_isolated(test_mode=False):
    """Run the benchmark in an isolated subprocess"""
    
    # Build the command
    cmd = [sys.executable, str(SCRIPT_DIR / "optimize_best_model.py"), "--device", "auto"]
    
    if test_mode:
        cmd.extend(["--test"])
    
    print("="*80)
    print("ISOLATED BENCHMARK RUNNER")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print(f"Test mode: {test_mode}")
    print("\nStarting benchmark in isolated process...")
    print("="*80)
    print()
    
    # Run with creationflags to create a new process group on Windows
    # This isolates it from parent process signal handling
    import os
    creation_flags = 0
    if os.name == 'nt':  # Windows
        # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=creation_flags,
            cwd=str(ROOT_DIR)
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='', flush=True)
        
        # Wait for completion
        return_code = process.wait()
        
        print()
        print("="*80)
        if return_code == 0:
            print("✓ Benchmark completed successfully!")
        else:
            print(f"✗ Benchmark exited with code {return_code}")
        print("="*80)
        
        return return_code
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating benchmark...")
        process.terminate()
        process.wait()
        return 1
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Isolated benchmark runner")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (50 epochs, 2 runs)")
    args = parser.parse_args()
    
    exit_code = run_benchmark_isolated(test_mode=args.test)
    sys.exit(exit_code)

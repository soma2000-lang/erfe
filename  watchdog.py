import subprocess
import time
import os
import signal

# Training process name or keyword to look for
TRAIN_KEYWORD = "python"  # Or "train.py" depending on how you launch training
TIMEOUT_MINUTES = 15  # Kill if no GPU activity for this many minutes

def get_gpu_usage():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        lines = output.decode().strip().split('\n')
        usage = {}
        for line in lines:
            pid, mem = line.split(',')
            usage[int(pid.strip())] = int(mem.strip())
        return usage
    except Exception as e:
        print(f"Error checking GPU usage: {e}")
        return {}

if __name__ == "__main__":
    print("Starting GPU Watchdog...")
    last_active_time = time.time()
    while True:
        gpu_usage = get_gpu_usage()
        
        if any(mem > 100 for mem in gpu_usage.values()):  # If any GPU memory usage > 100MB
            last_active_time = time.time()
        
        idle_time = (time.time() - last_active_time) / 60  # in minutes

        if idle_time > TIMEOUT_MINUTES:
            print(f"\n⚠️ GPU idle for {idle_time:.2f} minutes. Killing training process... ⚠️")

            # Kill all processes matching TRAIN_KEYWORD
            try:
                output = subprocess.check_output(["ps", "aux"])
                lines = output.decode().split('\n')
                for line in lines:
                    if TRAIN_KEYWORD in line and 'watchdog' not in line:
                        parts = line.split()
                        pid = int(parts[1])
                        print(f"Killing PID: {pid}")
                        os.kill(pid, signal.SIGKILL)
            except Exception as e:
                print(f"Error killing process: {e}")

            break

        time.sleep(60)  
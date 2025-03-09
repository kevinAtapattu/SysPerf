import time
import psutil
from datetime import datetime

# Try to import NVIDIA's NVML library for GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    has_gpu = True
except Exception as e:
    print("GPU monitoring not available:", e)
    has_gpu = False

def get_cpu_usage():
    """Return the current CPU usage percentage."""
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    """Return the current Memory usage percentage."""
    mem = psutil.virtual_memory()
    return mem.percent

def get_gpu_usage():
    """Return the current GPU usage percentage if available."""
    if has_gpu:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    return None

def log_metrics():
    """Collect and log system metrics."""
    cpu = get_cpu_usage()
    mem = get_memory_usage()
    gpu = get_gpu_usage()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{now} | CPU: {cpu}% | Memory: {mem}% | GPU: {gpu if gpu is not None else 'N/A'}%"
    print(log_entry)

if __name__ == "__main__":
    # Log system metrics every 2 seconds
    while True:
        log_metrics()
        time.sleep(2)

# An impractical solution for a slightly dumb problem that is hong kong's summer heat
import os
import time
import subprocess
import json
from datetime import datetime


def get_gpu_temperature():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)
        return float(result.stdout.decode().strip())
    except Exception as e:
        print(f"Error getting GPU temperature: {e}")
        return None


def wait_until_gpu_drops_below_temp(temp: float = 42., interval: float = 5.0):
    """Wait until the GPU temperature drops below a specified threshold."""
    if temp >= 100:
        return
    while True:
        gpu_temp = get_gpu_temperature()
        if gpu_temp is None or gpu_temp < temp:
            break
        time.sleep(interval)
    return

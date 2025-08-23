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


def wait_until_gpu_drops_below_temp(temp: int | float | str = 42., interval: float = 5.0):
    """Wait until the GPU temperature drops below a specified threshold."""
    if isinstance(temp, str):
        try:
            with open(temp, 'r') as f:
                data = json.load(f)
                temp = data.get('max_temp', 100)
        except Exception as e:
            print(f"Error reading temperature from {temp}: {e}")
            return
    assert isinstance(temp, (int, float)), f"Temperature must be an int or float, got {temp}"
    if temp >= 100:
        return
    while True:
        gpu_temp = get_gpu_temperature()
        if gpu_temp is None or gpu_temp < temp:
            break
        time.sleep(interval)
    return

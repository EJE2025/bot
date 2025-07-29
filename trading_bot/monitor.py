import os
import time
import psutil
from . import notify
from . import config

CPU_THRESHOLD = config.CPU_THRESHOLD
MEMORY_THRESHOLD_MB = config.MEMORY_THRESHOLD_MB


def monitor_system() -> None:
    """Monitor CPU and memory usage and send alerts on high load."""
    while True:
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().used / (1024 * 1024)
        if cpu_usage / 100 > CPU_THRESHOLD:
            notify.send_telegram(f"⚠️ Alto uso de CPU: {cpu_usage:.1f}%")
            notify.send_discord(f"⚠️ Alto uso de CPU: {cpu_usage:.1f}%")
        if mem_usage > MEMORY_THRESHOLD_MB:
            notify.send_telegram(f"⚠️ Alto uso de memoria: {mem_usage:.0f} MB")
            notify.send_discord(f"⚠️ Alto uso de memoria: {mem_usage:.0f} MB")
        time.sleep(60)


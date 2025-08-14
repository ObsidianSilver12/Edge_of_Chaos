# health.py

### START FILE ###

# File tool to Monitor own Health

import psutil  # Safe system monitoring


    
def capture_safe_system_metrics():
    """Safely capture actual system state without CLI access"""
    return {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'gpu_usage': get_gpu_usage_safe(),  # If available
        'temperature_sensors': get_safe_temperature_readings(),
        'power_consumption': estimate_power_consumption_safe(),
        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
        'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
    }
### END FILE ###
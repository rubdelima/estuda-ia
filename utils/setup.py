import GPUtil
import shutil
import psutil
import os

def get_hardware_info():
    """Obtém informações sobre CPU, GPU e memória disponível."""
    cpu_cores  = psutil.cpu_count(logical=False)
    total_threads = psutil.cpu_count(logical=True)
    total_disk = shutil.disk_usage("/").total / 1e9
    ram_total  = psutil.virtual_memory().total / 1e9

    try:
        gpu = GPUtil.getGPUs()[0]
        gpu_name = gpu.name
        gpu_memory = gpu.memoryTotal
        gpu_available = True
    except IndexError:
        gpu_name, gpu_memory, gpu_available = None, None, False

    return {
        "cpu_cores": cpu_cores,
        "cpu_threads" : total_threads,
        "ram_total": ram_total,
        "total_disk": total_disk,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
    }

def show_hardware_info():
    hardware_info  = get_hardware_info()
    print("\n🔍 Informações do Hardware:")
    for key, value in hardware_info.items():
        print(f"{key}: {value}")

def set_execution_mode():
    """Configura a execução do sistema para usar a GPU, se disponível."""
    hardware_info = get_hardware_info()
    
    if hardware_info["gpu_name"] and "NVIDIA" in hardware_info["gpu_name"].upper():
        print("\n✅ GPU NVIDIA detectada! Configurando execução para usar CUDA...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
        print("🔧 CUDA ativado para GPU NVIDIA")
    else:
        print("\n⚠️ Nenhuma GPU NVIDIA detectada. O sistema usará CPU.")
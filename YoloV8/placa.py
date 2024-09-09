import torch

print(torch.__version__)

print(f"GPU Configurada: {torch.cuda.is_available()}")
print(f"Total de GPUs: {torch.cuda.device_count()}")

if(torch.cuda.is_available()):
    print(f"GPU Atual: {torch.cuda.current_device()}")
    print(f"Device: {torch.cuda.device()}")
    print(f"Device Name: {torch.cuda.get_device_name()}")
else:
    print("Nenhuma GPU configurada")
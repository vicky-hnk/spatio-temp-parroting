"""
Check CUDA availability.
"""
import torch


def check_cuda():
    cuda_available = torch.cuda.is_available()
    print("CUDA is available: ", cuda_available)
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print("Number of GPUs: ", num_gpus)
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    check_cuda()

import os

try:
    import pynvml  # provides utility for NVIDIA management

    HAS_NVML = True
except:
    HAS_NVML = False


def auto_select_gpu():
    """ select gpu which has the largest free memory """
    if HAS_NVML:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        print(f'Found {deviceCount} GPUs')
        largest_free_mem = 0
        largest_free_idx = 0
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = info.free / 1024. / 1024.  # convert to MB
            total_mem = info.total / 1024. / 1024.
            print(f'GPU {i} memory: {free_mem:.0f}MB / {total_mem:.0f}MB')
            if free_mem > largest_free_mem:
                largest_free_mem = free_mem
                largest_free_idx = i
        pynvml.nvmlShutdown()
        print(f'Using largest free memory GPU {largest_free_idx} with free memory {largest_free_mem:.0f}MB')
        return str(largest_free_idx)
    else:
        print('pynvml is not installed, gpu auto-selection is disabled!')
        return None


def main(gpu_idx = None):
    if gpu_idx is None:
        gpu_idx = auto_select_gpu()
    else:
        print(f'Using GPU {gpu_idx}')
    if gpu_idx is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
        print(f'TORCH CUDA SET_DEVICE: {gpu_idx}')


if __name__ == '__main__':
    main()

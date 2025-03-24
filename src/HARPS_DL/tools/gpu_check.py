
import nvidia_smi # pip install nvidia-ml-py3
from pdb import set_trace

def get_empty_gpu():
    nvidia_smi.nvmlInit()
    empty_gpu_index = -1

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        if gpu_is_empty(handle, nvidia_smi, based_on='memory'):
            empty_gpu_index = i
            break
    nvidia_smi.nvmlShutdown()
    assert(empty_gpu_index != -1) # no device is empty -> cheaply thrown error
    return empty_gpu_index


def gpu_is_empty(handle, nvidia_smi, based_on='utilization'):
    if based_on == 'utilization':
        return nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu == 0 # unused
    elif based_on == 'memory':
        return nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used < 5145728 # less than 5 mb

def get_gpus_info():
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

    nvidia_smi.nvmlShutdown()

#print(get_empty_gpu())
#get_gpus_info()

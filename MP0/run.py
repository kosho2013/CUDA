import pycuda.driver as cuda
import pycuda.autoinit

def print_gpu_architecture_info():
    # Get device (assuming single GPU)
    device = cuda.Device(0)
    
    print(f"GPU Architecture Information for: {device.name()}")
    print("-" * 50)

    # Basic information
    num_sm = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    max_threads_per_sm = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
    max_blocks_per_sm = device.get_attribute(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR)
    max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    shared_memory_per_block = device.get_attribute(cuda.device_attribute.SHARED_MEMORY_PER_BLOCK)
    total_global_memory = device.total_memory() // (1024 ** 2)  # In MB

    # L1 Cache and L2 Cache
    l1_cache_size = device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)  # Approximate L1 cache size
    l2_cache_size = device.get_attribute(cuda.device_attribute.L2_CACHE_SIZE)  # In bytes

    # Register File Size
    registers_per_block = device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_BLOCK)
    registers_per_sm = device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR)

    # FLOPs (based on CUDA cores and clock rate)
    cuda_cores_per_sm = 64  # Typical value for many GPUs, but can vary
    clock_rate_khz = device.get_attribute(cuda.device_attribute.CLOCK_RATE)  # in kHz
    num_cuda_cores = num_sm * cuda_cores_per_sm
    clock_rate_ghz = clock_rate_khz / 1e6
    flops = 2 * num_cuda_cores * clock_rate_ghz * 1e9  # Approximate FLOPs

    # Memory bandwidth
    memory_bus_width_bits = device.get_attribute(cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH)  # in bits
    memory_clock_rate_khz = device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE)  # in kHz
    memory_bandwidth_gbps = (memory_clock_rate_khz * memory_bus_width_bits * 2) / (8 * 1e6)  # in GB/s

    # Printing GPU information
    print(f"Number of Streaming Multiprocessors (SM): {num_sm}")
    print(f"Max Threads per SM: {max_threads_per_sm}")
    print(f"Max Blocks per SM: {max_blocks_per_sm}")
    print(f"Max Threads per Block: {max_threads_per_block}")
    print(f"Shared Memory per Block: {shared_memory_per_block} bytes")
    print(f"Total Global Memory: {total_global_memory} MB")
    print(f"L1 Cache Size (approx.): {l1_cache_size} bytes")
    print(f"L2 Cache Size: {l2_cache_size} bytes")
    print(f"Registers per Block: {registers_per_block}")
    print(f"Registers per SM: {registers_per_sm}")
    print(f"Approximate FLOPs: {flops:.2e} FLOPs")
    print(f"Memory Bandwidth: {memory_bandwidth_gbps:.2f} GB/s")

if __name__ == "__main__":
    print_gpu_architecture_info()

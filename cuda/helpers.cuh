#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <iostream>
#include <cstdint>
#define CEIL(a, b) ((a + b - 1) / b)

// source: https://stackoverflow.com/a/14038590
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line
                  << "\n";
        if (abort) {
            std::abort();
        }
    }
}

inline void *safeCudaMalloc(uint64_t size)
{
    void *dest;
    gpuErrchk(cudaMalloc(&dest, size));
    static uint64_t total = 0;
    total += size;
    std::cerr << "GPU allocate ptr: " << dest << ". Size: " << size << " bytes. Total: " << total
              << " bytes" << std::endl;
    return dest;
}

#endif // CUDA_HELPERS_CUH
#ifndef TESTTEXTURECUDA_CUDA_HELPER_H
#define TESTTEXTURECUDA_CUDA_HELPER_H

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //TESTTEXTURECUDA_CUDA_HELPER_H

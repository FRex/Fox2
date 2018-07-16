#pragma once
#include <cstdio>
#include "cuda_runtime_api.h"
#include <cstdlib>

inline void impl_checkCudaCall(cudaError_t ret, const char * file, int line)
{
    if(ret != cudaSuccess)
    {
        std::fprintf(stderr, "%s:%d: cuda error %d\n", file, line, static_cast<int>(ret));
        std::exit(1);
    }
}

#define checkCudaCall(r) impl_checkCudaCall(r, __FILE__, __LINE__)

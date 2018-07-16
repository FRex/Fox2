#pragma once
#include <cstdio>
#include "cuda_runtime_api.h"

inline void impl_checkCudaCall(cudaError_t ret, const char * file, int line)
{
    if(ret != cudaSuccess)
        std::fprintf(stderr, "%s:%d: cuda error %d\n", file, line, static_cast<int>(ret));
}

#define checkCudaCall(r) impl_checkCudaCall(r, __FILE__, __LINE__)

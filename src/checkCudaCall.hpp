#pragma once
#include <cstdio>
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <string>
#include <iostream>

inline void impl_checkCudaCall(cudaError_t ret, const char * file, int line)
{
    if(ret != cudaSuccess)
    {
        std::fprintf(
            stderr,
            "%s:%d: cuda error %d: %s - %s\n",
            file,
            line,
            static_cast<int>(ret),
            cudaGetErrorName(ret),
            cudaGetErrorString(ret)
        );
        std::cerr << "Press enter to quit..." << std::endl;
        std::string line;
        std::getline(std::cin, line);
        std::exit(1);
    }
}

#define checkCudaCall(r) impl_checkCudaCall(r, __FILE__, __LINE__)

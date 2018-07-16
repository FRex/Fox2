#include "CudaAutoBuffer.hpp"
#include "cuda_runtime_api.h"
#include "checkCudaCall.hpp"
#include <algorithm>

CudaUntypedAutoBuffer::~CudaUntypedAutoBuffer()
{
    if(m_cudaptr)
        checkCudaCall(cudaFree(m_cudaptr));
}

void CudaUntypedAutoBuffer::resize(unsigned bytes)
{
    if(bytes == 0u)
    {
        if(m_cudaptr)
            checkCudaCall(cudaFree(m_cudaptr));

        m_cudaptr = 0x0;
        m_size = 0u;
        return;
    }

    void * old = m_cudaptr;
    const unsigned oldsize = m_size;
    checkCudaCall(cudaMalloc(&m_cudaptr, bytes));
    checkCudaCall(cudaMemset(m_cudaptr, 0x0, bytes));
    m_size = bytes;
    if(!old)
        return;

    checkCudaCall(cudaMemcpy(m_cudaptr, old, std::min(m_size, oldsize), cudaMemcpyDeviceToDevice));
    checkCudaCall(cudaFree(old));
}

unsigned CudaUntypedAutoBuffer::size() const
{
    return m_size;
}

void * CudaUntypedAutoBuffer::ptr() const
{
    return m_cudaptr;
}

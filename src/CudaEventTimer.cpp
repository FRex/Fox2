#include "CudaEventTimer.hpp"
#include "checkCudaCall.hpp"

#define startevent (static_cast<cudaEvent_t*>(m_events)[0])
#define stopevent (static_cast<cudaEvent_t*>(m_events)[1])

CudaEventTimer::CudaEventTimer()
{
    m_events = std::malloc(sizeof(cudaEvent_t) * 2);
    checkCudaCall(cudaEventCreate(&startevent));
    checkCudaCall(cudaEventCreate(&stopevent));
}

CudaEventTimer::~CudaEventTimer()
{
    checkCudaCall(cudaEventDestroy(startevent));
    checkCudaCall(cudaEventDestroy(stopevent));
    std::free(m_events);
}

void CudaEventTimer::start()
{
    checkCudaCall(cudaEventRecord(startevent));
}

void CudaEventTimer::stop()
{
    checkCudaCall(cudaEventRecord(stopevent));
    m_doneonce = true;
}

float CudaEventTimer::sync()
{
    if(!m_doneonce)
        return 0.f;

    float ret = 0.f;
    checkCudaCall(cudaEventSynchronize(stopevent));
    checkCudaCall(cudaEventElapsedTime(&ret, startevent, stopevent));
    return ret;
}

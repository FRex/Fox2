#pragma once
#include <SFML/System/NonCopyable.hpp>

class CudaEventTimer : sf::NonCopyable
{
public:
    CudaEventTimer();
    ~CudaEventTimer();
    void start();
    void stop();
    float sync();

private:
    bool m_doneonce = false;
    void * m_events = 0x0;

};

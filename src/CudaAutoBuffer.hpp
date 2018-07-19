#pragma once
#include <SFML/System/NonCopyable.hpp>

class CudaUntypedAutoBuffer : sf::NonCopyable
{
public:
    ~CudaUntypedAutoBuffer();
    void resize(unsigned bytes);
    unsigned size() const;
    void * ptr() const;

private:
    void * m_cudaptr = 0x0;
    unsigned m_size = 0u;

};


template <class T>
class CudaAutoBuffer
{
public:
    T * ptr() const
    {
        return static_cast<T*>(m_buff.ptr());
    }

    void resize(unsigned count)
    {
        m_buff.resize(sizeof(T) * count);
    }

    unsigned size() const
    {
        return m_buff.size() / sizeof(T);
    }

    unsigned bytesize() const
    {
        return m_buff.size();
    }

private:
    CudaUntypedAutoBuffer m_buff;

};

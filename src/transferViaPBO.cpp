#include "transferViaPBO.hpp"
#include "gl_core21.h"
#include <SFML/Graphics/Texture.hpp>
#include <cstring>
#include "checkCudaCall.hpp"
#include "cuda_gl_interop.h"

void transferViaPBO(unsigned * cudascreen, sf::Texture& texture, unsigned pbo)
{
    const int x = static_cast<int>(texture.getSize().x);
    const int y = static_cast<int>(texture.getSize().y);
    const unsigned size = texture.getSize().x * texture.getSize().y * 4;

    cudaGraphicsResource * res;
    checkCudaCall(cudaGraphicsGLRegisterBuffer(&res, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    void * ptr = 0x0;
    size_t buffsize = 0u;
    checkCudaCall(cudaGraphicsMapResources(1, &res));
    checkCudaCall(cudaGraphicsResourceGetMappedPointer(&ptr, &buffsize, res));
    checkCudaCall(cudaMemcpy(ptr, cudascreen, buffsize, cudaMemcpyDeviceToDevice));
    checkCudaCall(cudaGraphicsUnmapResources(1, &res));
    checkCudaCall(cudaGraphicsUnregisterResource(res));

    sf::Texture::bind(&texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        x, y, 0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        0x0
    );
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

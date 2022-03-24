#include "shader.h"
#include "cuda_helper.h"
#include <vector>
#include <iostream>

__global__ void global_testRGBA(cudaTextureObject_t tex, size_t width)
{
    const int count = 20;
    for(int i = 0; i < count; i++)
    {
        const float x = static_cast<float>(i) / count;
        const float4 c = tex2D<float4>(tex, x, 0.0f);

        printf("fetch(x=%f) = (r=%5f, g=%5f, b=%5f, a=%5f)\n", x, c.x, c.y, c.z, c.w);
    }
}

void kernelTestRGBA()
{
    printf("=== RGBA\n");

    const int width = 3;
    const int nBitsPerTexel = 32;
    const cudaChannelFormatKind format = cudaChannelFormatKindFloat;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc(nBitsPerTexel, nBitsPerTexel, nBitsPerTexel, nBitsPerTexel, format);

    // Fill the i-th texel with some random colors different for R, G, B and A
    std::vector<float4> arr(width);
    for(size_t i = 0; i < arr.size(); i++) {
        const float f = static_cast<float>(i);
        arr[i] = make_float4(f, f * 2.0f, f * 3.0f,  f * 4.0f);
    }

    cudaArray_t d_array;
    CUDA_CHECK(cudaMallocArray(&d_array, &desc, width, 1));

    CUDA_CHECK(cudaMemcpy2DToArray(d_array, 0, 0, arr.data(),
                                   sizeof(float4) * width, sizeof(float4) * width, 1,
                                   cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    const cudaTextureAddressMode mode = cudaAddressModeClamp;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = mode;
    texDesc.addressMode[1] = mode;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t textureObj = {};
    CUDA_CHECK(cudaCreateTextureObject(&textureObj, &resDesc, &texDesc, nullptr));

    global_testRGBA<<<1, 1>>>(textureObj, width);

    CUDA_CHECK(cudaDestroyTextureObject(textureObj));
    CUDA_CHECK(cudaFreeArray(d_array));

    std::cout << "end" << std::endl;
}
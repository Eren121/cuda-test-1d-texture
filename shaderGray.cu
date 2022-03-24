#include "shader.h"
#include "cuda_helper.h"
#include <vector>
#include <iostream>

__global__ void global_testGray(cudaTextureObject_t tex, size_t width)
{
    // We print some values of the textures in [0.0;1.0],
    // linearly interpolated
    const int count = 20;
    for(int i = 0; i < count; i++)
    {
        const float x = static_cast<float>(i) / count;
        const float color = tex2D<float>(tex, x, 0.0f);

        printf("fetch(x=%f) = %f\n", x, color);
    }
}

void kernelTestGray()
{
    printf("=== Gray\n");

    // You can use any value as the width,
    // We are creating a 1D texture of "width" texels with the i-th texel containing the value float(i).
    const int width = 3;

    // You can not use any value for nBitsPerTexel,
    // it should match the format.
    const int nBitsPerTexel = 32;

    const cudaChannelFormatKind format = cudaChannelFormatKindFloat;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc(nBitsPerTexel, 0, 0, 0, format);

    std::vector<float> arr(width);
    for(size_t i = 0; i < arr.size(); i++) {
        arr[i] = static_cast<float>(i);
    }

    cudaArray_t d_array;
    CUDA_CHECK(cudaMallocArray(&d_array, &desc, width, 1));

    CUDA_CHECK(cudaMemcpy2DToArray(d_array, 0, 0, arr.data(),
                        sizeof(float) * width, sizeof(float) * width, 1,
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

    global_testGray<<<1, 1>>>(textureObj, width);

    CUDA_CHECK(cudaDestroyTextureObject(textureObj));
    CUDA_CHECK(cudaFreeArray(d_array));

    std::cout << "end" << std::endl;
}
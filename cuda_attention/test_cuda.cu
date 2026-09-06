#include <iostream>
#include <cuda_runtime.h>

__global__ void helloGPU() {
    printf(
        "Hello from GPU! Block %d, Thread %d\n",
        blockIdx.x,
        threadIdx.x
    );
}

int main() {

    helloGPU<<<1, 4>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(err)
                  << std::endl;
        return 1;
    }

    std::cout
        << "CUDA program completed successfully!"
        << std::endl;

    return 0;
}
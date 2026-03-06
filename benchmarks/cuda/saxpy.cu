#include <stdio.h>
#include <stdlib.h>

__device__ float saxpy(float a, float b)
{
    return (((2 * a) + b));
}

extern "C" __global__ void map_2kernel(float *a1, float *a2, float *a3, int size)
{
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int stride = (blockDim.x * gridDim.x);
    for (int i = id; i < size; i += stride)
    {
        if ((id < size))
        {
            a3[id] = saxpy(a1[id], a2[id]);
        }
    }
}

int main(int argc, char const *argv[])
{
    int size = atoi(argv[1]);
    size_t bytes = sizeof(float) * size;

    float *host_a, *host_b, *host_result;
    host_a = (float *)malloc(bytes);
    host_b = (float *)malloc(bytes);
    host_result = (float *)malloc(bytes);

    // Filling a and b arrays with 1s (the same thing as in PolyHok)
    for (int i = 0; i < size; i++)
    {
        host_a[i] = 1;
        host_b[i] = 1;
    }

    // CUDA device arrays
    float *dev_a, *dev_b, *dev_result;
    cudaError_t err;

    // Same values as in PolyHok
    int threadsPerBlock = 256;
    int numberOfBlocks = 1024;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&dev_a, bytes);
    cudaMalloc((void **)&dev_b, bytes);
    cudaMalloc((void **)&dev_result, bytes);

    cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    map_2kernel<<<numberOfBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_result, size);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the result back to the host
    cudaMemcpy(host_result, dev_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("CUDA\t%d\t%3.1f\n", size, time);

    free(host_a);
    free(host_b);
    free(host_result);

    return 0;
}

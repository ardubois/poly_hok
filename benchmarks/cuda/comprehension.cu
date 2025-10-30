#include <stdio.h>

__global__ void comprehension(double *a, double *b, double *result, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int id = index; id < size; id += stride)
    {
        if (id < size)
        {
            result[id] = (2 * a[id]) + b[id];
        }
    }
}

int main(int argc, char const *argv[])
{
    int size = atoi(argv[1]);
    int bytes = size * sizeof(double);

    double *host_a, *host_b, *host_result;
    host_a = (double *)malloc(bytes);
    host_b = (double *)malloc(bytes);
    host_result = (double *)malloc(bytes);

    // Filling a and b arrays
    for (int i = 0; i < size; i++)
    {
        host_a[i] = i + 1;
        host_b[i] = i + 1;
    }

    double *dev_a, *dev_b, *dev_result;
    cudaError_t err;

    int threadsPerBlock = 128;
    int numberOfBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

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

    comprehension<<<numberOfBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_result, size);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

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

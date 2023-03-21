#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include<device_launch_parameters.h>
#include <iostream>

__global__ void matMul(float* A, float* B, float* C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
    }
    C[i * n + j] = sum;
}

void printMatrix(float* mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int n = 4;
    size_t size = n * n * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        h_A[i] = rand() % 10 + 1 /*/ (float)RAND_MAX*/;
        h_B[i] = rand() % 10 + 1/*/ (float)RAND_MAX*/;
    }

    // Print input matrices A and B
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, n);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, n);

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Multiply matrices A and B on device
    dim3 threadsPerBlock(n, n);
    dim3 numBlocks(2,2);
    matMul << <numBlocks, threadsPerBlock/2 >> > (d_A, d_B, d_C, n);

    // Copy result data from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result matrix C
    std::cout << "Result matrix C:" << std::endl;
    printMatrix(h_C, n);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

#include <iostream>
#include <ctime>
#include <cstdlib>

void matMul(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
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
    for (int i = 0; i < n * n; i++) {
        h_A[i] = rand() % 10 +1 /*/ (float)RAND_MAX*/;
        h_B[i] = rand() % 10 +1/*/ (float)RAND_MAX*/;
    }

    // Print input matrices A and B
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, n);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, n);

    // Multiply matrices A and B
    clock_t start = clock();
    matMul(h_A, h_B, h_C, n);
    clock_t end = clock();

    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Print result matrix C
    std::cout << "Result matrix C:" << std::endl;
    printMatrix(h_C, n);

    std::cout << "CPU time: " << cpuTime << " seconds" << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

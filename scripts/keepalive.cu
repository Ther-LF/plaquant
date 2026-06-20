// Tiny CUDA keepalive: tight loop on a single GPU. Compile with:
//   nvcc -O2 -arch=sm_100 -o keepalive scripts/keepalive.cu
// Run with:
//   CUDA_VISIBLE_DEVICES=7 nohup ./keepalive > keepalive.log 2>&1 &
// Used as a fallback when torch isn't installed yet (see scripts/keepalive_matmul.py
// for the Python version).
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstdio>

__global__ void busy(float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = d[idx];
        for (int i = 0; i < 20000; ++i) x = x * 1.0001f + 0.0001f;
        d[idx] = x;
    }
}

int main() {
    cudaSetDevice(0);
    int n = 1 << 20;
    float *d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    long iter = 0;
    while (true) {
        busy<<<(n + 255) / 256, 256>>>(d, n);
        cudaDeviceSynchronize();
        ++iter;
        if (iter % 60 == 0) {
            printf("iter %ld\n", iter);
            fflush(stdout);
        }
    }
    return 0;
}

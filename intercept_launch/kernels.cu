// kernels.cu

extern "C" __global__ void matrixAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" __global__ void matrixSub(int* A, int* B, int* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" __global__ void matrixMul(int* A, int* B, int* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}


// attention_kernels.cu - our CUDA kernels, built as a PyTorch extension (Step 4).
//
// The four __global__ kernels below are copied VERBATIM from
// cuda_attention/benchmark.cu (the file that produced our CUDA benchmark), so the
// code being timed is exactly the code from the project. benchmark_triton.py checks
// that the kernel text still matches that file.
//
// Only the host-side launchers at the bottom are new. They use the same grid/block
// shapes as benchmarkCUDA() in benchmark.cu, plus a CUDA stream argument so the
// kernels run on PyTorch's current stream (needed for fair event timing and CUDA graphs).

#include <cuda_runtime.h>
#include <cmath>

const int TILE = 16;

// ================== BEGIN VERBATIM COPY FROM cuda_attention/benchmark.cu ==================
// ============================================================
// NAIVE CUDA Q × K^T
// ============================================================

__global__ void naiveScoresKernel(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {

        if (col > row) {
            scores[row * seq_len + col] = -INFINITY;
            return;
        }

        float sum = 0.0f;

        for (int k = 0; k < head_dim; k++) {
            sum += Q[row * head_dim + k] *
                   K[col * head_dim + k];
        }

        scores[row * seq_len + col] =
            sum / sqrtf((float)head_dim);
    }
}


// ============================================================
// OPTIMIZED CUDA Q × K^T
// ============================================================

__global__ void optimizedScoresKernel(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim
) {
    __shared__ float Q_tile[TILE][TILE];
    __shared__ float K_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int tileStart = 0;
         tileStart < head_dim;
         tileStart += TILE) {

        int qIndex = tileStart + threadIdx.x;
        int kIndex = tileStart + threadIdx.y;

        // Load Q tile
        if (row < seq_len && qIndex < head_dim) {
            Q_tile[threadIdx.y][threadIdx.x] =
                Q[row * head_dim + qIndex];
        } else {
            Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load K tile
        if (col < seq_len && kIndex < head_dim) {
            K_tile[threadIdx.y][threadIdx.x] =
                K[col * head_dim + kIndex];
        } else {
            K_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += Q_tile[threadIdx.y][k] *
                   K_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < seq_len && col < seq_len) {

        if (col > row) {
            scores[row * seq_len + col] = -INFINITY;
        } else {
            scores[row * seq_len + col] =
                sum / sqrtf((float)head_dim);
        }
    }
}


// ============================================================
// SIMPLE CUDA SOFTMAX
//
// Same for both CUDA versions to make the comparison fair.
// ============================================================

__global__ void softmaxKernel(
    float* scores,
    int seq_len
) {
    int row = blockIdx.x;

    if (row >= seq_len || threadIdx.x != 0)
        return;

    float maxVal = -INFINITY;

    for (int j = 0; j <= row; j++) {
        float value = scores[row * seq_len + j];
        maxVal = fmaxf(maxVal, value);
    }

    float sum = 0.0f;

    for (int j = 0; j <= row; j++) {

        float value =
            expf(scores[row * seq_len + j] - maxVal);

        scores[row * seq_len + j] = value;

        sum += value;
    }

    for (int j = 0; j <= row; j++) {
        scores[row * seq_len + j] /= sum;
    }

    for (int j = row + 1; j < seq_len; j++) {
        scores[row * seq_len + j] = 0.0f;
    }
}


// ============================================================
// ATTENTION × V
// ============================================================

__global__ void outputKernel(
    const float* attention,
    const float* V,
    float* output,
    int seq_len,
    int head_dim
) {
    int dim =
        blockIdx.x * blockDim.x + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && dim < head_dim) {

        float sum = 0.0f;

        for (int j = 0; j <= row; j++) {
            sum += attention[row * seq_len + j] *
                   V[j * head_dim + dim];
        }

        output[row * head_dim + dim] = sum;
    }
}
// =================== END VERBATIM COPY FROM cuda_attention/benchmark.cu ===================


// ============================================================
// HOST LAUNCHERS (same launch shapes as benchmarkCUDA())
// ============================================================

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

void launch_naive_scores(const float* Q, const float* K, float* scores,
                         int seq_len, int head_dim, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid(cdiv(seq_len, TILE), cdiv(seq_len, TILE));
    naiveScoresKernel<<<grid, block, 0, stream>>>(Q, K, scores, seq_len, head_dim);
}

void launch_optimized_scores(const float* Q, const float* K, float* scores,
                             int seq_len, int head_dim, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid(cdiv(seq_len, TILE), cdiv(seq_len, TILE));
    optimizedScoresKernel<<<grid, block, 0, stream>>>(Q, K, scores, seq_len, head_dim);
}

void launch_softmax(float* scores, int seq_len, cudaStream_t stream) {
    softmaxKernel<<<seq_len, 1, 0, stream>>>(scores, seq_len);
}

void launch_output(const float* attention, const float* V, float* output,
                   int seq_len, int head_dim, cudaStream_t stream) {
    dim3 outputBlock(16, 16);
    dim3 outputGrid(cdiv(head_dim, 16), cdiv(seq_len, 16));
    outputKernel<<<outputGrid, outputBlock, 0, stream>>>(attention, V, output, seq_len, head_dim);
}

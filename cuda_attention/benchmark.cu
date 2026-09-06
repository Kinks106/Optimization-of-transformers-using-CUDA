

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define CHECK_CUDA(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        cerr << "CUDA Error: " << cudaGetErrorString(err)            \
             << " at line " << __LINE__ << endl;                     \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

const int HEAD_DIM = 64;
const int TILE = 16;
const int WARMUP = 5;
const int ITERATIONS = 20;


// ============================================================
// CPU SELF-ATTENTION
// ============================================================

void cpuAttention(
    const vector<float>& Q,
    const vector<float>& K,
    const vector<float>& V,
    vector<float>& scores,
    vector<float>& output,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Q × K^T + causal mask
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {

            float sum = 0.0f;

            if (j <= i) {
                for (int k = 0; k < head_dim; k++) {
                    sum += Q[i * head_dim + k] *
                           K[j * head_dim + k];
                }

                scores[i * seq_len + j] = sum * scale;
            } else {
                scores[i * seq_len + j] = -INFINITY;
            }
        }
    }

    // Row-wise softmax
    for (int i = 0; i < seq_len; i++) {

        float maxVal = -INFINITY;

        for (int j = 0; j <= i; j++) {
            maxVal = max(
                maxVal,
                scores[i * seq_len + j]
            );
        }

        float sum = 0.0f;

        for (int j = 0; j <= i; j++) {

            float value =
                expf(scores[i * seq_len + j] - maxVal);

            scores[i * seq_len + j] = value;

            sum += value;
        }

        for (int j = 0; j <= i; j++) {
            scores[i * seq_len + j] /= sum;
        }

        for (int j = i + 1; j < seq_len; j++) {
            scores[i * seq_len + j] = 0.0f;
        }
    }

    // Attention × V
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {

            float sum = 0.0f;

            for (int j = 0; j <= i; j++) {
                sum += scores[i * seq_len + j] *
                       V[j * head_dim + d];
            }

            output[i * head_dim + d] = sum;
        }
    }
}


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


// ============================================================
// RANDOM INITIALIZATION
// ============================================================

void initializeRandom(vector<float>& data) {

    mt19937 gen(42);

    uniform_real_distribution<float>
        dist(-1.0f, 1.0f);

    for (float& x : data) {
        x = dist(gen);
    }
}


// ============================================================
// BENCHMARK CUDA
// ============================================================

float benchmarkCUDA(
    bool optimized,
    float* d_Q,
    float* d_K,
    float* d_V,
    float* d_scores,
    float* d_output,
    int seq_len,
    int head_dim
) {
    dim3 block(TILE, TILE);

    dim3 grid(
        (seq_len + TILE - 1) / TILE,
        (seq_len + TILE - 1) / TILE
    );

    dim3 outputBlock(16, 16);

    dim3 outputGrid(
        (head_dim + 15) / 16,
        (seq_len + 15) / 16
    );

    // Warmup
    for (int i = 0; i < WARMUP; i++) {

        if (optimized) {
            optimizedScoresKernel<<<grid, block>>>(
                d_Q, d_K, d_scores,
                seq_len, head_dim
            );
        } else {
            naiveScoresKernel<<<grid, block>>>(
                d_Q, d_K, d_scores,
                seq_len, head_dim
            );
        }

        softmaxKernel<<<seq_len, 1>>>(
            d_scores, seq_len
        );

        outputKernel<<<outputGrid, outputBlock>>>(
            d_scores, d_V, d_output,
            seq_len, head_dim
        );
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; i++) {

        if (optimized) {
            optimizedScoresKernel<<<grid, block>>>(
                d_Q, d_K, d_scores,
                seq_len, head_dim
            );
        } else {
            naiveScoresKernel<<<grid, block>>>(
                d_Q, d_K, d_scores,
                seq_len, head_dim
            );
        }

        softmaxKernel<<<seq_len, 1>>>(
            d_scores, seq_len
        );

        outputKernel<<<outputGrid, outputBlock>>>(
            d_scores, d_V, d_output,
            seq_len, head_dim
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / ITERATIONS;
}


// ============================================================
// MAIN
// ============================================================

int main() {

    vector<int> sequenceLengths = {
        128,
        256,
        512,
        1024
    };

    cout << fixed << setprecision(4);

    cout << "\n============================================================\n";
    cout << " SELF-ATTENTION CUDA BENCHMARK\n";
    cout << "============================================================\n";

    cout << "\nHead Dimension: "
         << HEAD_DIM << endl;

    cout << "Iterations: "
         << ITERATIONS << endl;

    cout << "\n";

    cout << left
         << setw(12) << "Seq Length"
         << setw(15) << "CPU (ms)"
         << setw(18) << "Naive CUDA"
         << setw(18) << "Optimized CUDA"
         << setw(15) << "Speedup"
         << endl;

    cout << string(78, '-')
         << endl;


    for (int seq_len : sequenceLengths) {

        size_t qkvSize =
            seq_len *
            HEAD_DIM *
            sizeof(float);

        size_t scoresSize =
            seq_len *
            seq_len *
            sizeof(float);


        // Host data

        vector<float> Q(
            seq_len * HEAD_DIM
        );

        vector<float> K(
            seq_len * HEAD_DIM
        );

        vector<float> V(
            seq_len * HEAD_DIM
        );

        vector<float> scores(
            seq_len * seq_len
        );

        vector<float> output(
            seq_len * HEAD_DIM
        );


        initializeRandom(Q);
        initializeRandom(K);
        initializeRandom(V);


        // -------------------------
        // CPU BENCHMARK
        // -------------------------

        auto cpuStart =
            chrono::high_resolution_clock::now();

        cpuAttention(
            Q, K, V,
            scores, output,
            seq_len,
            HEAD_DIM
        );

        auto cpuEnd =
            chrono::high_resolution_clock::now();

        float cpuTime =
            chrono::duration<float, milli>(
                cpuEnd - cpuStart
            ).count();


        // -------------------------
        // GPU MEMORY
        // -------------------------

        float *d_Q;
        float *d_K;
        float *d_V;
        float *d_scores;
        float *d_output;

        CHECK_CUDA(
            cudaMalloc(&d_Q, qkvSize)
        );

        CHECK_CUDA(
            cudaMalloc(&d_K, qkvSize)
        );

        CHECK_CUDA(
            cudaMalloc(&d_V, qkvSize)
        );

        CHECK_CUDA(
            cudaMalloc(
                &d_scores,
                scoresSize
            )
        );

        CHECK_CUDA(
            cudaMalloc(
                &d_output,
                qkvSize
            )
        );


        CHECK_CUDA(
            cudaMemcpy(
                d_Q, Q.data(),
                qkvSize,
                cudaMemcpyHostToDevice
            )
        );

        CHECK_CUDA(
            cudaMemcpy(
                d_K, K.data(),
                qkvSize,
                cudaMemcpyHostToDevice
            )
        );

        CHECK_CUDA(
            cudaMemcpy(
                d_V, V.data(),
                qkvSize,
                cudaMemcpyHostToDevice
            )
        );


        // -------------------------
        // NAIVE CUDA
        // -------------------------

        float naiveTime =
            benchmarkCUDA(
                false,
                d_Q, d_K, d_V,
                d_scores, d_output,
                seq_len,
                HEAD_DIM
            );


        // -------------------------
        // OPTIMIZED CUDA
        // -------------------------

        float optimizedTime =
            benchmarkCUDA(
                true,
                d_Q, d_K, d_V,
                d_scores, d_output,
                seq_len,
                HEAD_DIM
            );


        float speedup =
            cpuTime / optimizedTime;


        // -------------------------
        // PRINT RESULTS
        // -------------------------

        cout << left
             << setw(12) << seq_len
             << setw(15) << cpuTime
             << setw(18) << naiveTime
             << setw(18) << optimizedTime
             << setw(15) << speedup
             << endl;


        // Cleanup

        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_scores);
        cudaFree(d_output);
    }


    cout << "============================================================\n";

    cout << "\nBenchmark completed successfully.\n";

    return 0;
}
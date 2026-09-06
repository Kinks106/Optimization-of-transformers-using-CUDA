#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

using namespace std;

#define CHECK_CUDA(call)                                                \
do {                                                                    \
    cudaError_t error = call;                                           \
    if (error != cudaSuccess) {                                         \
        cerr << "CUDA Error: " << cudaGetErrorString(error)             \
             << " at line " << __LINE__ << endl;                        \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)


// ============================================================
// CONFIGURATION
// ============================================================

const int SEQ_LEN = 512;
const int HEAD_DIM = 64;

const int TILE = 16;


// ============================================================
// OPTIMIZED Q × K^T
//
// Shared memory tiled matrix multiplication.
//
// Each thread calculates one attention score.
// ============================================================

__global__ void optimizedAttentionScores(
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

    // Move across HEAD_DIM in tiles
    for (int tileStart = 0;
         tileStart < head_dim;
         tileStart += TILE) {

        int qCol = tileStart + threadIdx.x;

        // Load Q tile into shared memory
        if (row < seq_len && qCol < head_dim) {
            Q_tile[threadIdx.y][threadIdx.x] =
                Q[row * head_dim + qCol];
        } else {
            Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        /*
        K[col][tileStart + threadIdx.y]

        K is logically transposed during multiplication.
        */
        int kCol = tileStart + threadIdx.y;

        if (col < seq_len && kCol < head_dim) {
            K_tile[threadIdx.y][threadIdx.x] =
                K[col * head_dim + kCol];
        } else {
            K_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply tiles
        for (int k = 0; k < TILE; k++) {
            sum += Q_tile[threadIdx.y][k] *
                   K_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < seq_len && col < seq_len) {

        // Scale attention
        float score =
            sum / sqrtf((float)head_dim);

        // Causal mask
        if (col > row) {
            score = -INFINITY;
        }

        scores[row * seq_len + col] = score;
    }
}


// ============================================================
// SIMPLE SOFTMAX
//
// Temporary implementation.
// Parallel softmax will be added in a later phase.
// ============================================================

__global__ void softmaxKernel(
    float* scores,
    int seq_len
) {
    int row = blockIdx.x;

    if (row >= seq_len)
        return;

    if (threadIdx.x == 0) {

        float maxVal = -INFINITY;

        // Find max
        for (int j = 0; j < seq_len; j++) {
            float value = scores[row * seq_len + j];

            if (value > maxVal)
                maxVal = value;
        }

        float sum = 0.0f;

        // Exponentiation
        for (int j = 0; j < seq_len; j++) {

            float value =
                scores[row * seq_len + j];

            if (value == -INFINITY) {

                scores[row * seq_len + j] = 0.0f;

            } else {

                scores[row * seq_len + j] =
                    expf(value - maxVal);
            }

            sum += scores[row * seq_len + j];
        }

        // Normalize
        for (int j = 0; j < seq_len; j++) {
            scores[row * seq_len + j] /= sum;
        }
    }
}


// ============================================================
// ATTENTION × V
// ============================================================

__global__ void attentionOutputKernel(
    const float* attention,
    const float* V,
    float* output,
    int seq_len,
    int head_dim
) {
    int dim =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (row < seq_len && dim < head_dim) {

        float sum = 0.0f;

        for (int j = 0; j < seq_len; j++) {

            sum +=
                attention[row * seq_len + j] *
                V[j * head_dim + dim];
        }

        output[row * head_dim + dim] = sum;
    }
}


// ============================================================
// RANDOM DATA GENERATION
// ============================================================

void initializeRandom(vector<float>& data) {

    mt19937 generator(42);

    uniform_real_distribution<float>
        distribution(-1.0f, 1.0f);

    for (float& value : data) {
        value = distribution(generator);
    }
}


// ============================================================
// MAIN
// ============================================================

int main() {

    cout << "=============================================\n";
    cout << " OPTIMIZED CUDA SELF-ATTENTION\n";
    cout << "=============================================\n";

    cout << "Sequence Length: "
         << SEQ_LEN << endl;

    cout << "Head Dimension: "
         << HEAD_DIM << endl;

    cout << "Tile Size: "
         << TILE << " x " << TILE << endl;


    size_t qkv_size =
        SEQ_LEN *
        HEAD_DIM *
        sizeof(float);

    size_t scores_size =
        SEQ_LEN *
        SEQ_LEN *
        sizeof(float);


    // Host memory

    vector<float> h_Q(
        SEQ_LEN * HEAD_DIM
    );

    vector<float> h_K(
        SEQ_LEN * HEAD_DIM
    );

    vector<float> h_V(
        SEQ_LEN * HEAD_DIM
    );

    vector<float> h_output(
        SEQ_LEN * HEAD_DIM
    );


    initializeRandom(h_Q);
    initializeRandom(h_K);
    initializeRandom(h_V);


    // Device memory

    float *d_Q;
    float *d_K;
    float *d_V;
    float *d_scores;
    float *d_output;


    CHECK_CUDA(
        cudaMalloc(&d_Q, qkv_size)
    );

    CHECK_CUDA(
        cudaMalloc(&d_K, qkv_size)
    );

    CHECK_CUDA(
        cudaMalloc(&d_V, qkv_size)
    );

    CHECK_CUDA(
        cudaMalloc(
            &d_scores,
            scores_size
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &d_output,
            qkv_size
        )
    );


    // Copy to GPU

    CHECK_CUDA(
        cudaMemcpy(
            d_Q,
            h_Q.data(),
            qkv_size,
            cudaMemcpyHostToDevice
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            d_K,
            h_K.data(),
            qkv_size,
            cudaMemcpyHostToDevice
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            d_V,
            h_V.data(),
            qkv_size,
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // CUDA EVENT TIMING
    // ========================================================

    cudaEvent_t start, stop;

    CHECK_CUDA(
        cudaEventCreate(&start)
    );

    CHECK_CUDA(
        cudaEventCreate(&stop)
    );


    dim3 blockScores(
        TILE,
        TILE
    );

    dim3 gridScores(
        (SEQ_LEN + TILE - 1) / TILE,
        (SEQ_LEN + TILE - 1) / TILE
    );


    dim3 blockOutput(
        16,
        16
    );

    dim3 gridOutput(
        (HEAD_DIM + 15) / 16,
        (SEQ_LEN + 15) / 16
    );


    // ========================================================
    // START TIMER
    // ========================================================

    CHECK_CUDA(
        cudaEventRecord(start)
    );


    // Optimized QK^T

    optimizedAttentionScores<<<
        gridScores,
        blockScores
    >>>(
        d_Q,
        d_K,
        d_scores,
        SEQ_LEN,
        HEAD_DIM
    );

    CHECK_CUDA(cudaGetLastError());


    // Softmax

    softmaxKernel<<<
        SEQ_LEN,
        1
    >>>(
        d_scores,
        SEQ_LEN
    );

    CHECK_CUDA(cudaGetLastError());


    // Attention × V

    attentionOutputKernel<<<
        gridOutput,
        blockOutput
    >>>(
        d_scores,
        d_V,
        d_output,
        SEQ_LEN,
        HEAD_DIM
    );

    CHECK_CUDA(cudaGetLastError());


    // Stop timer

    CHECK_CUDA(
        cudaEventRecord(stop)
    );

    CHECK_CUDA(
        cudaEventSynchronize(stop)
    );


    float milliseconds = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );


    // Copy result

    CHECK_CUDA(
        cudaMemcpy(
            h_output.data(),
            d_output,
            qkv_size,
            cudaMemcpyDeviceToHost
        )
    );


    // Print only a small sample

    cout << "\nOutput Sample:\n";

    for (int i = 0; i < 5; i++) {

        for (int j = 0; j < 5; j++) {

            cout << fixed
                 << setprecision(4)
                 << h_output[
                        i * HEAD_DIM + j
                    ]
                 << " ";
        }

        cout << endl;
    }


    cout << "\n=============================================\n";

    cout << "Total GPU Execution Time: "
         << milliseconds
         << " ms\n";

    cout << "=============================================\n";


    // Cleanup

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}
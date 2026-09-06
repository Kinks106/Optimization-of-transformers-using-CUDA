#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>

#include <cuda_runtime.h>

using namespace std;

const int SEQ_LEN = 4;
const int HEAD_DIM = 3;


// CUDA error checking
#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
        cerr << "CUDA Error: "                                \
             << cudaGetErrorString(error)                     \
             << endl;                                         \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)


// ============================================================
// KERNEL 1: Q × K^T / sqrt(d) + Causal Mask
//
// Thread (row, col) calculates one attention score.
// ============================================================

__global__ void attentionScoresKernel(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {

        float sum = 0.0f;

        // Q[row] dot K[col]
        for (int k = 0; k < head_dim; k++) {

            sum += Q[row * head_dim + k] *
                   K[col * head_dim + k];
        }

        // Scale
        float score = sum / sqrtf((float)head_dim);

        // Causal Mask:
        // Future tokens are not allowed
        if (col > row) {
            score = -INFINITY;
        }

        scores[row * seq_len + col] = score;
    }
}


// ============================================================
// KERNEL 2: Row-wise Softmax
//
// One CUDA block handles one row.
// ============================================================

__global__ void softmaxKernel(
    float* scores,
    int seq_len
) {
    int row = blockIdx.x;

    if (row >= seq_len)
        return;

    // For now, thread 0 calculates the row sequentially.
    // Later we will optimize this using parallel reduction.
    if (threadIdx.x == 0) {

        // Step 1: Find maximum
        float maxVal = -INFINITY;

        for (int j = 0; j < seq_len; j++) {

            float value =
                scores[row * seq_len + j];

            if (value > maxVal) {
                maxVal = value;
            }
        }


        // Step 2: Exponentiate
        float sum = 0.0f;

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


        // Step 3: Normalize
        for (int j = 0; j < seq_len; j++) {

            scores[row * seq_len + j] /= sum;
        }
    }
}


// ============================================================
// KERNEL 3: Attention × V
//
// Thread (row, dimension) calculates one output value.
// ============================================================

__global__ void attentionOutputKernel(
    const float* attention,
    const float* V,
    float* output,
    int seq_len,
    int head_dim
) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

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
// PRINT MATRIX
// ============================================================

void printMatrix(
    const vector<float>& matrix,
    int rows,
    int cols,
    const string& name
) {
    cout << "\n" << name << ":\n";

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            cout << fixed << setprecision(4)
                 << matrix[i * cols + j]
                 << " ";
        }

        cout << endl;
    }
}


// ============================================================
// MAIN
// ============================================================

int main() {

    cout << "========================================\n";
    cout << " NAIVE CUDA SELF-ATTENTION\n";
    cout << "========================================\n";


    // Host matrices

    vector<float> h_Q = {
        1.0, 0.5, 0.2,
        0.3, 0.8, 0.4,
        0.9, 0.1, 0.7,
        0.2, 0.6, 0.9
    };

    vector<float> h_K = {
        0.4, 0.3, 0.9,
        0.7, 0.2, 0.5,
        0.6, 0.8, 0.1,
        0.5, 0.4, 0.7
    };

    vector<float> h_V = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };


    int qkv_size =
        SEQ_LEN * HEAD_DIM * sizeof(float);

    int scores_size =
        SEQ_LEN * SEQ_LEN * sizeof(float);


    // Device pointers

    float *d_Q;
    float *d_K;
    float *d_V;
    float *d_scores;
    float *d_output;


    // Allocate GPU memory

    CHECK_CUDA(cudaMalloc(&d_Q, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_size));

    CHECK_CUDA(
        cudaMalloc(&d_scores, scores_size)
    );

    CHECK_CUDA(
        cudaMalloc(&d_output, qkv_size)
    );


    // Copy Host -> Device

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
    // KERNEL 1
    // ========================================================

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (SEQ_LEN + blockSize.x - 1) /
        blockSize.x,

        (SEQ_LEN + blockSize.y - 1) /
        blockSize.y
    );


    attentionScoresKernel<<<
        gridSize,
        blockSize
    >>>(
        d_Q,
        d_K,
        d_scores,
        SEQ_LEN,
        HEAD_DIM
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // ========================================================
    // KERNEL 2: SOFTMAX
    // ========================================================

    softmaxKernel<<<
        SEQ_LEN,
        1
    >>>(
        d_scores,
        SEQ_LEN
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // ========================================================
    // KERNEL 3: ATTENTION × V
    // ========================================================

    attentionOutputKernel<<<
        gridSize,
        blockSize
    >>>(
        d_scores,
        d_V,
        d_output,
        SEQ_LEN,
        HEAD_DIM
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Host output

    vector<float> h_scores(
        SEQ_LEN * SEQ_LEN
    );

    vector<float> h_output(
        SEQ_LEN * HEAD_DIM
    );


    // Copy Device -> Host

    CHECK_CUDA(
        cudaMemcpy(
            h_scores.data(),
            d_scores,
            scores_size,
            cudaMemcpyDeviceToHost
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            h_output.data(),
            d_output,
            qkv_size,
            cudaMemcpyDeviceToHost
        )
    );


    // Print results

    printMatrix(
        h_scores,
        SEQ_LEN,
        SEQ_LEN,
        "CUDA Attention Weights"
    );

    printMatrix(
        h_output,
        SEQ_LEN,
        HEAD_DIM,
        "CUDA Final Attention Output"
    );


    // Free GPU memory

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_output);


    cout << "\nCUDA Self-Attention completed successfully.\n";

    return 0;
}
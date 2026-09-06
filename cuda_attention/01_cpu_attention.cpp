#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace std;

const int SEQ_LEN = 4;
const int HEAD_DIM = 3;

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
                 << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}


// Q × Kᵀ / sqrt(d)
void calculateAttentionScores(
    const vector<float>& Q,
    const vector<float>& K,
    vector<float>& scores
) {
    for (int i = 0; i < SEQ_LEN; i++) {

        for (int j = 0; j < SEQ_LEN; j++) {

            float sum = 0.0f;

            // Dot product: Q[i] · K[j]
            for (int k = 0; k < HEAD_DIM; k++) {
                sum += Q[i * HEAD_DIM + k] *
                       K[j * HEAD_DIM + k];
            }

            // Scale by sqrt(head dimension)
            scores[i * SEQ_LEN + j] =
                sum / sqrt((float)HEAD_DIM);
        }
    }
}


// Causal Mask
void applyCausalMask(vector<float>& scores) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {

            // Token i cannot attend to future token j
            if (j > i) {
                scores[i * SEQ_LEN + j] =
                    -numeric_limits<float>::infinity();
            }
        }
    }
}


// Row-wise Softmax
void softmax(vector<float>& scores) {

    for (int i = 0; i < SEQ_LEN; i++) {

        // Find maximum value
        float maxVal = -numeric_limits<float>::infinity();

        for (int j = 0; j < SEQ_LEN; j++) {
            maxVal = max(
                maxVal,
                scores[i * SEQ_LEN + j]
            );
        }

        // Exponentiation
        float sum = 0.0f;

        for (int j = 0; j < SEQ_LEN; j++) {

            float value =
                scores[i * SEQ_LEN + j];

            if (value == -numeric_limits<float>::infinity()) {
                scores[i * SEQ_LEN + j] = 0.0f;
            } else {
                scores[i * SEQ_LEN + j] =
                    exp(value - maxVal);
            }

            sum += scores[i * SEQ_LEN + j];
        }

        // Normalize
        for (int j = 0; j < SEQ_LEN; j++) {
            scores[i * SEQ_LEN + j] /= sum;
        }
    }
}


// Attention × V
void calculateOutput(
    const vector<float>& attention,
    const vector<float>& V,
    vector<float>& output
) {
    for (int i = 0; i < SEQ_LEN; i++) {

        for (int d = 0; d < HEAD_DIM; d++) {

            float sum = 0.0f;

            for (int j = 0; j < SEQ_LEN; j++) {

                sum += attention[i * SEQ_LEN + j] *
                       V[j * HEAD_DIM + d];
            }

            output[i * HEAD_DIM + d] = sum;
        }
    }
}


int main() {

    // Q, K, V dimensions = SEQ_LEN × HEAD_DIM

    vector<float> Q = {
        1.0, 0.5, 0.2,
        0.3, 0.8, 0.4,
        0.9, 0.1, 0.7,
        0.2, 0.6, 0.9
    };

    vector<float> K = {
        0.4, 0.3, 0.9,
        0.7, 0.2, 0.5,
        0.6, 0.8, 0.1,
        0.5, 0.4, 0.7
    };

    vector<float> V = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };

    vector<float> scores(SEQ_LEN * SEQ_LEN);
    vector<float> output(SEQ_LEN * HEAD_DIM);

    cout << "========================================\n";
    cout << " CPU SELF-ATTENTION FROM SCRATCH\n";
    cout << "========================================\n";

    printMatrix(Q, SEQ_LEN, HEAD_DIM, "Query Matrix Q");
    printMatrix(K, SEQ_LEN, HEAD_DIM, "Key Matrix K");
    printMatrix(V, SEQ_LEN, HEAD_DIM, "Value Matrix V");


    // Step 1: QKᵀ
    calculateAttentionScores(Q, K, scores);

    printMatrix(
        scores,
        SEQ_LEN,
        SEQ_LEN,
        "Scaled Attention Scores: QK^T / sqrt(d)"
    );


    // Step 2: Causal Mask
    applyCausalMask(scores);


    // Step 3: Softmax
    softmax(scores);

    printMatrix(
        scores,
        SEQ_LEN,
        SEQ_LEN,
        "Attention Weights after Causal Mask + Softmax"
    );


    // Step 4: Attention × V
    calculateOutput(scores, V, output);

    printMatrix(
        output,
        SEQ_LEN,
        HEAD_DIM,
        "Final Self-Attention Output"
    );

    cout << "\nSelf-Attention calculation completed successfully.\n";

    return 0;
}
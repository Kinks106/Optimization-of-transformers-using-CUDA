// attention_ext.cpp - Python bindings for our CUDA attention kernels.
//
// Kept separate from the .cu file so that nvcc never has to compile the large
// torch/extension.h headers (more reliable with MSVC on Windows).
//
// Every launch is followed by C10_CUDA_KERNEL_LAUNCH_CHECK(), so a failed launch
// (for example "no kernel image is available") raises a Python error instead of
// producing an impossible timing, as happened with the prebuilt benchmark.exe.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

void launch_naive_scores(const float* Q, const float* K, float* scores,
                         int seq_len, int head_dim, cudaStream_t stream);
void launch_optimized_scores(const float* Q, const float* K, float* scores,
                             int seq_len, int head_dim, cudaStream_t stream);
void launch_softmax(float* scores, int seq_len, cudaStream_t stream);
void launch_output(const float* attention, const float* V, float* output,
                   int seq_len, int head_dim, cudaStream_t stream);

static void check_2d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 2 && t.is_contiguous(), name, " must be a contiguous 2-D tensor");
}

// ---- individual stages (used for the per-stage timing table) ----

torch::Tensor scores_stage(torch::Tensor q, torch::Tensor k, torch::Tensor scores, bool optimized) {
    check_2d(q, "q"); check_2d(k, "k"); check_2d(scores, "scores");
    const int n = q.size(0), d = q.size(1);
    TORCH_CHECK(k.size(0) == n && k.size(1) == d, "k must have the same shape as q");
    TORCH_CHECK(scores.size(0) == n && scores.size(1) == n, "scores must be [N, N]");
    c10::cuda::CUDAGuard guard(q.device());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    if (optimized) {
        launch_optimized_scores(q.data_ptr<float>(), k.data_ptr<float>(), scores.data_ptr<float>(), n, d, stream);
    } else {
        launch_naive_scores(q.data_ptr<float>(), k.data_ptr<float>(), scores.data_ptr<float>(), n, d, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

torch::Tensor softmax_stage(torch::Tensor scores) {
    check_2d(scores, "scores");
    TORCH_CHECK(scores.size(0) == scores.size(1), "scores must be square");
    c10::cuda::CUDAGuard guard(scores.device());
    launch_softmax(scores.data_ptr<float>(), scores.size(0), c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

torch::Tensor output_stage(torch::Tensor attention, torch::Tensor v, torch::Tensor out) {
    check_2d(attention, "attention"); check_2d(v, "v"); check_2d(out, "out");
    const int n = v.size(0), d = v.size(1);
    TORCH_CHECK(attention.size(0) == n && attention.size(1) == n, "attention must be [N, N]");
    TORCH_CHECK(out.size(0) == n && out.size(1) == d, "out must be [N, d]");
    c10::cuda::CUDAGuard guard(v.device());
    launch_output(attention.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), n, d,
                  c10::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- full attention: the same 3 launches per iteration as benchmarkCUDA() ----

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        torch::Tensor scores, torch::Tensor out, bool optimized) {
    check_2d(v, "v");
    scores_stage(q, k, scores, optimized);
    softmax_stage(scores);
    output_stage(scores, v, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Naive and optimized CUDA causal self-attention (kernels from cuda_attention/benchmark.cu)";
    m.def("attention", &attention, "Full attention: scores -> softmax -> @V",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scores"), py::arg("out"), py::arg("optimized"));
    m.def("scores", &scores_stage, "Stage 1: Q@K^T / sqrt(d) + causal mask",
          py::arg("q"), py::arg("k"), py::arg("scores"), py::arg("optimized"));
    m.def("softmax", &softmax_stage, "Stage 2: row softmax (in place)", py::arg("scores"));
    m.def("output", &output_stage, "Stage 3: attention @ V",
          py::arg("attention"), py::arg("v"), py::arg("out"));
}

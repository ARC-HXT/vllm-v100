# SPDX-License-Identifier: Apache-2.0
"""cuBLASLt algorithm enumeration, checking and launching for fp16 y[M,N] = x[M,K] @ W[N,K]^T (fp32 accumulate), the
layout of torch.nn.functional.linear / vLLM default_unquantized_gemm.  Used by the algorithm table in
vllm/attpatt_v100/gemm.py (switch ATTPATT_V100_GEMM).

Enumeration follows NVIDIA CUDALibrarySamples LtSgemmCustomFind: cublasLtMatmulAlgoGetIds -> cublasLtMatmulAlgoInit ->
capability queries (tile ids, stages ids, split-K support, reduction scheme mask, CTA swizzling, custom options) -> every
combination checked with cublasLtMatmulAlgoCheck.  Each returned row = 8 int64 of the opaque cublasLtMatmulAlgo_t + 8
int64 of info [algo id, tile id, stages id, split-K, reduction scheme, swizzle, custom option, workspace bytes].

Column-major view used by cuBLASLt: D[N,M] = op(A)[N,K] * B[K,M] with A = W (K x N col-major, transa = T), B = x
(K x M col-major, no transpose), D = y (N x M col-major, ld N) - the same call torch makes for F.linear.  With a bias,
CUBLASLT_EPILOGUE_BIAS adds the length-N vector to every column of D (torch's F.linear(x, w, b) uses the same epilogue)."""
import os
import torch

_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
#include <cstring>

#define LT_CHECK(x) do { cublasStatus_t s_ = (x); TORCH_CHECK(s_ == CUBLAS_STATUS_SUCCESS, #x, " failed: ", (int)s_); } while (0)

static cublasLtHandle_t g_lt = nullptr;
struct Descs { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; };
// key (M, N, K, bias): with bias the descriptor carries CUBLASLT_EPILOGUE_BIAS (fp16 bias of length N added in the
// fp32 epilogue before the single rounding to fp16, as F.linear(x, w, b) does), so algorithms are enumerated, checked
// and run for exactly that operation
static std::map<std::tuple<int64_t, int64_t, int64_t, bool>, Descs> g_descs;

static cublasLtHandle_t lt() { if (!g_lt) LT_CHECK(cublasLtCreate(&g_lt)); return g_lt; }

static Descs& descs(int64_t M, int64_t N, int64_t K, bool bias) {
  auto key = std::make_tuple(M, N, K, bias);
  auto it = g_descs.find(key);
  if (it != g_descs.end()) return it->second;
  Descs d;
  LT_CHECK(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  if (bias) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cudaDataType_t bt = CUDA_R_16F;
    LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
  }
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_16F, K, N, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_16F, K, M, K));
  LT_CHECK(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_16F, N, M, N));
  return g_descs.emplace(key, d).first->second;
}

static void push(std::vector<int64_t>& out, const cublasLtMatmulAlgo_t& algo, size_t ws) {
  int64_t raw[8]; std::memcpy(raw, &algo, sizeof(raw));
  for (int i = 0; i < 8; ++i) out.push_back(raw[i]);
  int32_t id = 0; uint32_t tile = 0, stages = 0, splitk = 1, red = 0, swz = 0, cust = 0; size_t w = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz), &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof(cust), &w);
  int64_t info[8] = {id, tile, stages, splitk, red, swz, cust, (int64_t)ws};
  for (int i = 0; i < 8; ++i) out.push_back(info[i]);
}

static torch::Tensor rows(std::vector<int64_t>& out) {
  int64_t n = (int64_t)out.size() / 16;
  auto t = torch::empty({n, 16}, torch::kInt64);
  if (n) std::memcpy(t.data_ptr<int64_t>(), out.data(), out.size() * sizeof(int64_t));
  return t;
}

torch::Tensor heuristic(int64_t M, int64_t N, int64_t K, int64_t ws_bytes, int64_t count, bool bias) {
  Descs& d = descs(M, N, K, bias);
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = (uint64_t)ws_bytes;
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  std::vector<cublasLtMatmulHeuristicResult_t> res(count);
  int got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(lt(), d.op, d.a, d.b, d.c, d.c, pref, (int)count, res.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  std::vector<int64_t> out;
  if (s == CUBLAS_STATUS_SUCCESS)
    for (int i = 0; i < got; ++i) if (res[i].state == CUBLAS_STATUS_SUCCESS) push(out, res[i].algo, res[i].workspaceSize);
  return rows(out);
}

template <typename T> static std::vector<T> cap_array(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoCapAttributes_t a) {
  size_t sz = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&algo, a, nullptr, 0, &sz) != CUBLAS_STATUS_SUCCESS || sz == 0) return {};
  std::vector<T> v(sz / sizeof(T));
  cublasLtMatmulAlgoCapGetAttribute(&algo, a, v.data(), sz, &sz);
  return v;
}

torch::Tensor enumerate(int64_t M, int64_t N, int64_t K, int64_t ws_bytes, std::vector<int64_t> splits, bool bias) {
  Descs& d = descs(M, N, K, bias);
  int ids[256]; int nids = 0;
  LT_CHECK(cublasLtMatmulAlgoGetIds(lt(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                    256, ids, &nids));
  std::vector<int64_t> out;
  for (int i = 0; i < nids; ++i) {
    cublasLtMatmulAlgo_t algo;
    if (cublasLtMatmulAlgoInit(lt(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                               ids[i], &algo) != CUBLAS_STATUS_SUCCESS) continue;
    auto tiles = cap_array<uint32_t>(algo, CUBLASLT_ALGO_CAP_TILE_IDS);
    if (tiles.empty()) tiles.push_back(CUBLASLT_MATMUL_TILE_UNDEFINED);
    auto stages = cap_array<uint32_t>(algo, CUBLASLT_ALGO_CAP_STAGES_IDS);
    if (stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
    int32_t splitk_ok = 0; uint32_t red_mask = 0, swz_max = 0; int32_t cust_max = 0; size_t w = 0;
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, sizeof(splitk_ok), &w);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &red_mask, sizeof(red_mask), &w);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swz_max, sizeof(swz_max), &w);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &cust_max, sizeof(cust_max), &w);
    for (uint32_t tile : tiles) for (uint32_t st : stages) for (int32_t c = 0; c <= cust_max; ++c)
    for (uint32_t swz = 0; swz <= swz_max; ++swz) for (int64_t sk : splits) {
      if (sk > 1 && !splitk_ok) continue;
      std::vector<uint32_t> reds;
      if (sk == 1) reds.push_back(CUBLASLT_REDUCTION_SCHEME_NONE);
      else for (uint32_t r = 1; r < CUBLASLT_REDUCTION_SCHEME_MASK; r <<= 1) if (r & red_mask) reds.push_back(r);
      for (uint32_t red : reds) {
        uint32_t uc = (uint32_t)c, usk = (uint32_t)sk;
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, sizeof(st));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &uc, sizeof(uc));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &usk, sizeof(usk));
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red));
        cublasLtMatmulHeuristicResult_t r;
        if (cublasLtMatmulAlgoCheck(lt(), d.op, d.a, d.b, d.c, d.c, &algo, &r) == CUBLAS_STATUS_SUCCESS
            && r.state == CUBLAS_STATUS_SUCCESS && r.workspaceSize <= (size_t)ws_bytes)
          push(out, algo, r.workspaceSize);
      }
    }
  }
  return rows(out);
}

int64_t version() { lt(); return (int64_t)cublasLtGetVersion(); }

// create the handle and the descriptors of one problem now (host side only), so a later call under CUDA graph capture
// only looks them up
void prepare(int64_t M, int64_t N, int64_t K, bool bias) { descs(M, N, K, bias); }

// bias: fp16 [N] or None; its pointer is set on the shared descriptor right before the call (host side, so a CUDA graph
// captures the pointer of the call being captured)
void matmul(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor algo_row, torch::Tensor workspace,
            c10::optional<torch::Tensor> bias) {
  int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  Descs& d = descs(M, N, K, bias.has_value());
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->numel() == N && bias->is_contiguous(), "fp16 bias [N]");
    const void* p = bias->data_ptr();
    LT_CHECK(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &p, sizeof(p)));
  }
  cublasLtMatmulAlgo_t algo; std::memcpy(&algo, algo_row.data_ptr<int64_t>(), sizeof(algo));
  float alpha = 1.f, beta = 0.f;
  LT_CHECK(cublasLtMatmul(lt(), d.op, &alpha, w.data_ptr(), d.a, x.data_ptr(), d.b, &beta, y.data_ptr(), d.c,
                          y.data_ptr(), d.c, &algo, workspace.data_ptr(), (size_t)workspace.numel(),
                          at::cuda::getCurrentCUDAStream()));
}
"""

_mod = None


def _load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        import nvidia.cublas

        from vllm.attpatt_v100.jit import ext_name
        lib = os.path.join(list(nvidia.cublas.__path__)[0], "lib")
        functions = ["heuristic", "enumerate", "matmul", "version", "prepare"]
        ldflags = [f"-L{lib}", "-l:libcublasLt.so.12", f"-Wl,-rpath,{lib}"]
        _mod = load_inline(ext_name("attpatt_v100_gemm_lt", _CPP, functions, ldflags), cpp_sources=_CPP,
                           functions=functions, with_cuda=True, extra_cflags=["-O2"], extra_ldflags=ldflags,
                           verbose=False)
    return _mod


def heuristic(M, N, K, ws_bytes, count=64, bias=False):
    return _load().heuristic(M, N, K, ws_bytes, count, bias)


def enumerate_algos(M, N, K, ws_bytes, splits=(1, 2, 3, 4, 5, 6, 8, 12, 16), bias=False):
    return _load().enumerate(M, N, K, ws_bytes, list(splits), bias)


def version():
    return _load().version()


def prepare(M, N, K, bias=False):
    _load().prepare(M, N, K, bias)


def matmul(x, w, y, algo_row, workspace, bias=None):
    """y[M,N] = x[M,K] @ w[N,K]^T (+ bias[N] in the epilogue) with the given algo row (CPU int64, first 8 entries =
    algo, chosen for the same bias / no-bias operation); all fp16 contiguous"""
    _load().matmul(x, w, y, algo_row, workspace, bias)

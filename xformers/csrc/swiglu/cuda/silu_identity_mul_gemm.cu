#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <45_dual_gemm/device/silu_identity_mul_gemm.h>

namespace {

std::tuple<at::Tensor> silu_identity_mul_gemm_META(
    const at::Tensor& gx,
    const at::Tensor& ux,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2) {
  TORCH_CHECK(gx.stride(-1) == 1);
  TORCH_CHECK(ux.stride(-1) == 1);
  TORCH_CHECK(w2.stride(-1) == 1);

  at::SymInt B = gx.sym_size(0);
  at::SymInt I = gx.sym_size(1);
  at::SymInt H = w2.sym_size(0);

  at::Tensor d2 = at::empty_symint({B, H}, gx.options());

  return std::make_tuple(d2);
}

template <typename scalar_t>
std::tuple<at::Tensor> silu_identity_mul_gemm_(
    const at::Tensor& gx,
    const at::Tensor& ux,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2) {
  TORCH_CHECK(gx.dim() == 2);
  TORCH_CHECK(ux.dim() == 2);
  TORCH_CHECK(w2.dim() == 2);

  TORCH_CHECK(gx.stride(-1) == 1);
  TORCH_CHECK(ux.stride(-1) == 1);
  TORCH_CHECK(w2.stride(-1) == 1);

  at::cuda::CUDAGuard device_guard(gx.device());

  int64_t B = gx.size(0);
  int64_t I = gx.size(1);
  int64_t H = w2.size(0);

  at::Tensor d2 = at::empty({B, H}, gx.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // templati-ze the cutlass kernel
  cutlass::gemm::GemmCoord problem_size(B, H, I);

  constexpr int kStages = 3;

  using ElementInputA = scalar_t;
  using ElementInputB = scalar_t;
  using ElementOutput = scalar_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  const ElementCompute alpha = ElementCompute(1);
  const ElementCompute beta = b2.has_value() ? ElementCompute(1) : ElementCompute(0);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Alignment of A/B operands
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;

  using ArchTag = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value, // The number of elements per vectorized
                                                        // memory access. This becomes the vector width of
                                                        // math instructions in the epilogue too.
      ElementAccumulator,
      ElementCompute>;

  using SiluIdentityMulGemm = typename cutlass::gemm::device::SiluIdentityMulGemm<
      ElementInputA,
      cutlass::layout::RowMajor,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      kStages,
      AlignmentA,
      AlignmentB>;
  {
    cudaDeviceProp* p = at::cuda::getDeviceProperties(gx.device().index());
    TORCH_CHECK(
        p->major * 10 + p->minor >= ArchTag::kMinComputeCapability,
        "Only A100+ GPUs are supported");
  }

  using RefA = typename cutlass::
      TensorRef<typename SiluIdentityMulGemm::ElementA, typename SiluIdentityMulGemm::LayoutA>;
  using RefB = typename cutlass::
      TensorRef<typename SiluIdentityMulGemm::ElementB, typename SiluIdentityMulGemm::LayoutB>;
  using RefC = typename cutlass::
      TensorRef<typename SiluIdentityMulGemm::ElementC, typename SiluIdentityMulGemm::LayoutC>;
  RefC ref_b2;
  if (b2.has_value()) {
    ref_b2 =
        RefC{(scalar_t*)b2->data_ptr(), typename SiluIdentityMulGemm::LayoutC::Stride(0)};
  }

  int split_k_slices = 1;
  typename SiluIdentityMulGemm::Arguments arguments{
      problem_size,
      RefA{
          (scalar_t*)gx.data_ptr(),
          typename SiluIdentityMulGemm::LayoutA::Stride(gx.stride(0))},
      RefA{
          (scalar_t*)ux.data_ptr(),
          typename SiluIdentityMulGemm::LayoutA::Stride(ux.stride(0))},
      RefB{
          (scalar_t*)w2.data_ptr(),
          typename SiluIdentityMulGemm::LayoutB::Stride(w2.stride(0))},
      ref_b2,
      RefC{
          (scalar_t*)d2.data_ptr(),
          typename SiluIdentityMulGemm::LayoutC::Stride(d2.stride(0))},
      typename SiluIdentityMulGemm::EpilogueOutputOp::Params{alpha, beta},
      split_k_slices};

  SiluIdentityMulGemm silu_identity_mul_gemm;
  at::Tensor workspace = at::empty(
      {int64_t(silu_identity_mul_gemm.get_workspace_size(arguments))},
      gx.options().dtype(at::ScalarType::Byte));
  cutlass::Status status = silu_identity_mul_gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "`silu_identity_mul_gemm` does not support this input: ",
      cutlass::cutlassGetStatusString(status));

  status = silu_identity_mul_gemm.initialize(arguments, (uint8_t*)workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel initialize failed");
  status = silu_identity_mul_gemm(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel run failed");

  return std::make_tuple(d2);
}

std::tuple<at::Tensor> silu_identity_mul_gemm(
    const at::Tensor& gx,
    const at::Tensor& ux,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(gx.dim() == 2);
  TORCH_CHECK(ux.dim() == 2);
  TORCH_CHECK(w2.dim() == 2);

#define FWD_PARAMS gx, ux, w2, b2
  if (gx.scalar_type() == at::ScalarType::Half) {
    return silu_identity_mul_gemm_<cutlass::half_t>(FWD_PARAMS);
  } else {
    TORCH_CHECK(
        gx.scalar_type() == at::ScalarType::BFloat16, "Only supports bf16/f16");
    return silu_identity_mul_gemm_<cutlass::bfloat16_t>(FWD_PARAMS);
  }
}

std::tuple<at::Tensor> silu_identity_mul_gemm_autocast(
    const at::Tensor& gx,
    const at::Tensor& ux,
    const at::Tensor& w2,
    const c10::optional<at::Tensor>& b2) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_gpu_dtype();
  return silu_identity_mul_gemm(
      at::autocast::cached_cast(exec_type, gx),
      at::autocast::cached_cast(exec_type, ux),
      at::autocast::cached_cast(exec_type, w2),
      at::autocast::cached_cast(exec_type, b2));
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::silu_identity_mul_gemm"),
      TORCH_FN(silu_identity_mul_gemm));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::silu_identity_mul_gemm"),
      TORCH_FN(silu_identity_mul_gemm_META));
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::silu_identity_mul_gemm"),
      TORCH_FN(silu_identity_mul_gemm_autocast));
}

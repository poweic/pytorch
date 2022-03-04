#include <ATen/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/ConvUtils.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/ts_backend/ops/cast.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/expand.h>
#include <torch/csrc/lazy/ts_backend/ops/scalar.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/jit.h>

#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

namespace torch_lazy_tensors {
namespace compiler {

torch::lazy::Shape InferRepeat(const ir::ops::Repeat* repeat) {
  const torch::lazy::Output& input = repeat->operand(0);
  const torch::lazy::Shape& input_shape =
      torch::lazy::GetShapeFromTsOutput(input);
  const auto& repeats = repeat->repeats();
  CHECK_GE(repeats.size(), input_shape.dim());

  int64_t num_new_dimensions = repeats.size() - input_shape.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), input_shape.sizes().begin(),
                     input_shape.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  for (const auto idx : c10::irange(repeats.size())) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }
  return torch::lazy::Shape(input_shape.scalar_type(), target_size);
}

torch::lazy::Shape InferShape(const torch::lazy::Node* node) {
  switch (node->op().op) {
    // activation and unary op do not change shape
    case at::aten::repeat: {
      return InferRepeat(torch::lazy::NodeCast<ir::ops::Repeat>(
          node, torch::lazy::OpKind(at::aten::repeat)));
    }
    default:
      LOG(FATAL) << *node << "Not implemented yet.";
  }
}
}  // namespace compiler
}  // namespace torch_lazy_tensors

/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

bool register_custom_outer() {
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    std::vector<habana::custom_op::InputDesc> inputs_desc{input_a_desc, input_b_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
        auto a_tensor = inputs[0].toTensor();
        std::vector<int64_t> a_sizes = a_tensor.sizes().vec();
        std::cout << "in tensor size " << a_sizes << std::endl;
        // std::vector<int64_t> output_sizes(a_sizes.begin(), a_sizes.end());
        // output_sizes.push_back(a_sizes.back());
        std::vector<int64_t> output_sizes{a_sizes[0], a_sizes[1], a_sizes[2], a_sizes[3]*a_sizes[3]};

        return output_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_outer", //schema name
        "custom_outer_product_fwd_f32", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::custom_outer\n";
    return true;
}

at::Tensor custom_outer_execute(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");

  // Registering the custom op, need to be called only once
  static bool registered = register_custom_outer();
  TORCH_CHECK(registered, "custom_outer kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_outer");
  // Actual call for op execution
  std::vector<at::Tensor> outputs = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return outputs[0];
}

bool register_custom_softmax() {
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_desc1{
        habana::custom_op::input_type::USER_PARAMS, 1};

    std::vector<habana::custom_op::InputDesc> inputs_desc{input_a_desc, input_desc1};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
        auto a_tensor = inputs[0].toTensor();
        std::vector<int64_t> a_sizes = a_tensor.sizes().vec();
        std::cout << "in tensor size " << a_sizes << std::endl;
        std::vector<int64_t> output_sizes(a_sizes.begin(), a_sizes.end());
        return output_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};

    // user param callback
    struct SoftMaxParam
    {
        int32_t dim;
    };

    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(SoftMaxParam);
      params->dim = inputs[1].toInt();
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_softmax", //schema name
        "custom_softmax_sparse_f32", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_softmax\n";
    return true;
}

at::Tensor custom_softmax_execute(
    torch::Tensor input_a,
    at::Scalar dim) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");

  // Registering the custom op, need to be called only once
  static bool registered = register_custom_softmax();
  TORCH_CHECK(registered, "custom_softmax kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, dim};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_softmax");
  // Actual call for op execution
  std::vector<at::Tensor> outputs = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return outputs[0];
}


TORCH_LIBRARY(custom_op, m) {
  m.def("custom_outer(Tensor A, Tensor B) -> Tensor"),
  m.def("custom_softmax(Tensor A, Scalar dim) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_outer", custom_outer_execute),
  m.impl("custom_softmax", custom_softmax_execute);
}

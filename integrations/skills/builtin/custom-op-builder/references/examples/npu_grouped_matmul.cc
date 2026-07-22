/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "acl/acl_base.h"
#include "include/utils/tensor_utils.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "mindspore/include/custom_op_api.h"

namespace custom {
using TensorList = std::vector<ms::Tensor>;

struct TensorListWithDType {
  std::vector<mindspore::tensor::TensorPtr> tensors;
  aclDataType dtype;
};

mindspore::device::ascend::aclTensorList *ConvertType(const TensorListWithDType &tensor_list_wrapper);
}  // namespace custom

namespace ms::pynative {
inline custom::TensorListWithDType Arg(const custom::TensorListWithDType &wrapper) { return wrapper; }
}  // namespace ms::pynative

namespace custom {
namespace {
constexpr int64_t kInNotSplitOutNotSplit = 0;
constexpr int64_t kInSplitOutNotSplit = 1;
constexpr int64_t kInNotSplitOutSplit = 2;
constexpr int64_t kInSplitOutSplit = 3;
constexpr int64_t kDefaultSplit = -1;
constexpr int64_t kMSplit = 0;
constexpr int64_t kKSplit = 2;
constexpr int64_t kInt4NumsInInt32 = 8;
constexpr int64_t kFp4NumsInInt8 = 2;
constexpr int64_t kAclDTypeUnset = -1;
constexpr int64_t kQuantPerGroupSize = 0;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    MS_LOG(EXCEPTION) << message;
  }
}

bool HasOpApi(const char *api_name) { return mindspore::device::ascend::GetOpApiFunc(api_name) != nullptr; }

bool IsV4Available() { return HasOpApi("aclnnGroupedMatmulV4GetWorkspaceSize"); }

bool IsV5Available() { return HasOpApi("aclnnGroupedMatmulV5GetWorkspaceSize"); }

bool IsWeightNzAvailable() { return HasOpApi("aclnnGroupedMatmulWeightNzGetWorkspaceSize"); }

int64_t OptIntValue(const std::optional<int64_t> &value, int64_t default_value) {
  return value.has_value() ? value.value() : default_value;
}

bool IsNotSplitOutput(int64_t split_item) {
  return split_item == kInNotSplitOutNotSplit || split_item == kInSplitOutNotSplit;
}

bool IsSplitOutput(int64_t split_item) { return split_item == kInNotSplitOutSplit || split_item == kInSplitOutSplit; }

void CheckDims(int64_t split_item, size_t num_x, size_t num_weight, size_t num_group_list) {
  Check(num_x > 0 && num_weight > 0, "Invalid inputs: neither x nor weight could be empty.");
  Check(split_item == kInNotSplitOutNotSplit || split_item == kInSplitOutNotSplit ||
          split_item == kInNotSplitOutSplit || split_item == kInSplitOutSplit,
        "Invalid value of split_item [" + std::to_string(split_item) + "], which should only be one of 0/1/2/3.");
  if (IsNotSplitOutput(split_item)) {
    if (num_group_list > 0) {
      Check(num_x == 1 && num_weight == num_group_list,
            "When split_item = 0 or 1 and group_list is not None, x length must be 1 and weight length must equal "
            "group_list length.");
    } else {
      Check(num_x == num_weight,
            "When split_item = 0 or 1 and group_list is None, x length must equal weight length.");
    }
  }
}

bool IsWeightTrans(const ms::Tensor &tensor) {
  const auto shape = tensor.shape();
  if (shape.size() < 2) {
    return false;
  }
  const auto stride = tensor.stride();
  const size_t dim1 = shape.size() - 1;
  const size_t dim2 = shape.size() - 2;
  return stride[dim2] == 1 && stride[dim1] == shape[dim2];
}

bool IsWeightNz(const ms::Tensor &tensor) {
  const auto format = tensor.format();
  return format == "FRACTAL_NZ" || format == "FRACTAL_NZ_C0_2" || format == "FRACTAL_NZ_C0_4" ||
         format == "FRACTAL_NZ_C0_16";
}

bool IsFp8Type(ms::TypeId dtype) {
  return dtype == ms::TypeId::kNumberTypeFloat8E5M2 || dtype == ms::TypeId::kNumberTypeFloat8E4M3FN ||
         dtype == ms::TypeId::kNumberTypeHiFloat8;
}

bool IsMxfp4Valid(const TensorList &x, const TensorList &weight, const std::optional<int64_t> &x_acl_dtype,
                  const std::optional<int64_t> &weight_acl_dtype) {
  constexpr int64_t kAclFloat4E2M1 = 27;
  if (x_acl_dtype.has_value() || weight_acl_dtype.has_value()) {
    return x_acl_dtype.value_or(kAclDTypeUnset) == kAclFloat4E2M1 &&
           weight_acl_dtype.value_or(kAclDTypeUnset) == kAclFloat4E2M1;
  }
  return false;
}

ms::TypeId OutputDType(const TensorList &x, const std::optional<int64_t> &output_dtype) {
  return output_dtype.has_value() ? static_cast<ms::TypeId>(output_dtype.value()) : x[0].data_type();
}

void CreateNewTensorMultiDim(TensorList *y, const ms::Tensor &x_i, int64_t n, ms::TypeId dtype) {
  auto y_shape = x_i.shape();
  Check(!y_shape.empty(), "x tensor rank must be at least 1.");
  y_shape.back() = n;
  y->emplace_back(dtype, y_shape);
}

void CreateNewTensor(TensorList *y, int64_t dim_m, int64_t dim_n, ms::TypeId dtype) {
  y->emplace_back(dtype, ShapeVector{dim_m, dim_n});
}

void CreateNewTensorBatch(TensorList *y, int64_t batch, int64_t dim_m, int64_t dim_n, ms::TypeId dtype) {
  y->emplace_back(dtype, ShapeVector{batch, dim_m, dim_n});
}

int64_t HostGroupListValue(const ms::Tensor &group_list, size_t index) {
  Check(group_list.data_type() == ms::TypeId::kNumberTypeInt64, "group_list tensor dtype must be int64.");
  auto *data = group_list.data_ptr<int64_t>();
  Check(data != nullptr, "group_list tensor data pointer is null.");
  return data[index];
}

aclDataType AclDTypeOf(ms::TypeId dtype) { return mindspore::device::ascend::AclConverter::ConvertType(dtype); }

aclDataType AclDTypeValue(const std::optional<int64_t> &acl_dtype, ms::TypeId fallback_dtype) {
  return acl_dtype.has_value() ? static_cast<aclDataType>(acl_dtype.value()) : AclDTypeOf(fallback_dtype);
}

TensorListWithDType MakeTensorListWithDType(const TensorList &tensors, aclDataType dtype) {
  TensorListWithDType wrapped;
  wrapped.dtype = dtype;
  wrapped.tensors.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    wrapped.tensors.emplace_back(tensor.tensor());
  }
  return wrapped;
}

TensorListWithDType MakeOptionalTensorListWithDType(const std::optional<TensorList> &tensors, aclDataType dtype) {
  return MakeTensorListWithDType(tensors.value_or(TensorList{}), dtype);
}

ms::Tensor EmptyIfNone(const std::optional<ms::Tensor> &tensor) {
  return tensor.has_value() ? tensor.value() : ms::Tensor();
}

TensorList EmptyListIfNone(const std::optional<TensorList> &tensors) { return tensors.value_or(TensorList{}); }

TensorList CollectRunInputs(const TensorList &x, const TensorList &weight, const TensorList &bias, const TensorList &scale,
                            const TensorList &offset, const TensorList &antiquant_scale,
                            const TensorList &antiquant_offset, const TensorList &per_token_scale,
                            const std::optional<ms::Tensor> &group_list, const TensorList &activation_input,
                            const TensorList &activation_quant_scale, const TensorList &activation_quant_offset) {
  TensorList inputs;
  auto append = [&inputs](const TensorList &items) { inputs.insert(inputs.end(), items.begin(), items.end()); };
  append(x);
  append(weight);
  append(bias);
  append(scale);
  append(offset);
  append(antiquant_scale);
  append(antiquant_offset);
  append(per_token_scale);
  inputs.emplace_back(EmptyIfNone(group_list));
  append(activation_input);
  append(activation_quant_scale);
  append(activation_quant_offset);
  return inputs;
}

TensorList InferLegacyOutputs(const TensorList &x, const TensorList &weight, const std::vector<int64_t> &group_list,
                              int64_t split_item, ms::TypeId dtype) {
  const size_t num_x = x.size();
  const size_t num_weight = weight.size();
  CheckDims(split_item, num_x, num_weight, group_list.size());

  TensorList y;
  if (IsNotSplitOutput(split_item)) {
    if (!group_list.empty()) {
      y.reserve(group_list.size());
      Check(group_list[0] >= 0, "group_list[0] should be larger than or equal to 0.");
      CreateNewTensor(&y, group_list[0], weight[0].shape()[1], dtype);
      for (size_t i = 1; i < group_list.size(); ++i) {
        Check(group_list[i] - group_list[i - 1] >= 0, "group_list must be nondecreasing.");
        CreateNewTensor(&y, group_list[i] - group_list[i - 1], weight[i].shape()[1], dtype);
      }
    } else {
      y.reserve(num_x);
      for (size_t i = 0; i < num_x; ++i) {
        CreateNewTensorMultiDim(&y, x[i], weight[i].shape()[1], dtype);
      }
    }
  } else if (IsSplitOutput(split_item)) {
    if (num_x > 1) {
      int64_t dim_m = 0;
      for (const auto &x_i : x) {
        dim_m += x_i.shape()[0];
      }
      CreateNewTensor(&y, dim_m, weight[0].shape()[1], dtype);
    } else if (num_x == 1) {
      CreateNewTensor(&y, x[0].shape()[0], weight[0].shape()[1], dtype);
    }
  }
  return y;
}

TensorList InferTensorGroupListOutputs(const TensorList &x, const TensorList &weight,
                                       const std::optional<ms::Tensor> &group_list, int64_t split_item,
                                       int64_t group_type, bool mxfp4_valid, ms::TypeId dtype) {
  const size_t num_x = x.size();
  const bool single_weight = weight.size() == 1 && weight[0].shape().size() == 3;
  const size_t num_weight = single_weight ? static_cast<size_t>(weight[0].shape()[0]) : weight.size();
  const size_t num_group_list = group_list.has_value() ? static_cast<size_t>(group_list->shape()[0]) : 0;
  CheckDims(split_item, num_x, num_weight, num_group_list);

  const auto dim_num_w = weight[0].shape().size();
  const int64_t n0 = weight[0].shape()[dim_num_w - 1];
  const bool weight_trans = IsWeightTrans(weight[0]);
  const int64_t n_new = (mxfp4_valid && !weight_trans) ? (n0 * kFp4NumsInInt8) : n0;
  if (mxfp4_valid) {
    Check(x[0].shape()[1] != 1, "In mxfp4, dim K should not be 2.");
  }

  TensorList y;
  if (IsNotSplitOutput(split_item)) {
    if (num_group_list > 0) {
      y.reserve(num_group_list);
      int64_t prev = HostGroupListValue(*group_list, 0);
      Check(prev >= 0, "group_list[0] should be larger than or equal to 0.");
      CreateNewTensor(&y, prev, n0, dtype);
      for (size_t i = 1; i < num_group_list; ++i) {
        const int64_t cur = HostGroupListValue(*group_list, i);
        Check(cur - prev >= 0, "group_list must be nondecreasing.");
        const int64_t ni = single_weight ? n0 : weight[i].shape()[dim_num_w - 1];
        CreateNewTensor(&y, cur - prev, ni, dtype);
        prev = cur;
      }
    } else {
      y.reserve(num_x);
      for (size_t i = 0; i < num_x; ++i) {
        const int64_t ni = single_weight ? n0 : weight[i].shape()[dim_num_w - 1];
        CreateNewTensorMultiDim(&y, x[i], ni, dtype);
      }
    }
  } else if (IsSplitOutput(split_item)) {
    if (num_x > 1) {
      int64_t dim_m = 0;
      for (const auto &x_i : x) {
        dim_m += x_i.shape()[0];
      }
      const int64_t dim_n = weight[0].data_type() == ms::TypeId::kNumberTypeInt32 ? n0 * kInt4NumsInInt32 : n_new;
      CreateNewTensor(&y, dim_m, dim_n, dtype);
    } else if (num_x == 1) {
      if (group_type == kKSplit) {
        Check(num_weight == 1, "When group_type is 2 and split_item is 2/3, weight length must be 1.");
        const int64_t dim_n =
          weight[0].data_type() == ms::TypeId::kNumberTypeInt32 ? n0 * kInt4NumsInInt32 : n_new;
        CreateNewTensorBatch(&y, num_group_list, x[0].shape()[0], dim_n, dtype);
      } else {
        const bool int_like =
          (weight[0].data_type() == ms::TypeId::kNumberTypeInt32 ||
           (weight[0].data_type() == ms::TypeId::kNumberTypeFloat32 && weight[0].data_type() != x[0].data_type())) &&
          !weight_trans;
        CreateNewTensor(&y, x[0].shape()[0], int_like ? n0 * kInt4NumsInInt32 : n_new, dtype);
      }
    }
  }
  return y;
}

TensorList UnifyWeightShapeForInt4(const TensorList &weight) {
  TensorList new_weight;
  new_weight.reserve(weight.size());
  for (const auto &w : weight) {
    if (w.data_type() != ms::TypeId::kNumberTypeInt4) {
      new_weight.emplace_back(w);
      continue;
    }
    Check(w.is_contiguous(), "GroupedMatmulV4 does not support noncontiguous int4 weight.");
    auto new_shape = w.shape();
    new_shape.back() *= 2;
    auto new_w = w.reshape(new_shape);
    new_weight.emplace_back(new_w);
  }
  return new_weight;
}

TensorList npu_grouped_matmul_impl(const TensorList &x, const TensorList &weight,
                                   const std::optional<TensorList> &bias_opt,
                                   const std::optional<TensorList> &scale_opt,
                                   const std::optional<TensorList> &offset_opt,
                                   const std::optional<TensorList> &antiquant_scale_opt,
                                   const std::optional<TensorList> &antiquant_offset_opt,
                                   const std::optional<TensorList> &per_token_scale_opt,
                                   const std::optional<ms::Tensor> &group_list_tensor_opt,
                                   const std::optional<std::vector<int64_t>> &group_list_vector_opt,
                                   const std::optional<TensorList> &activation_input_opt,
                                   const std::optional<TensorList> &activation_quant_scale_opt,
                                   const std::optional<TensorList> &activation_quant_offset_opt,
                                   const std::optional<int64_t> &split_item_opt,
                                   const std::optional<int64_t> &group_type_opt,
                                   const std::optional<int64_t> &group_list_type_opt,
                                   const std::optional<int64_t> &act_type_opt,
                                   const std::optional<std::vector<int64_t>> &tuning_config_opt,
                                   const std::optional<int64_t> &output_dtype_opt,
                                   const std::optional<int64_t> &x_acl_dtype_opt,
                                   const std::optional<int64_t> &weight_acl_dtype_opt,
                                   const std::optional<int64_t> &scale_acl_dtype_opt,
                                   const std::optional<int64_t> &per_token_scale_acl_dtype_opt) {
  Check(!(group_list_tensor_opt.has_value() && group_list_vector_opt.has_value()),
        "group_list tensor and group_list vector cannot both be provided.");
  const int64_t split_item = OptIntValue(split_item_opt, 0);
  const auto out_dtype = OutputDType(x, output_dtype_opt);

  const auto bias = EmptyListIfNone(bias_opt);
  const auto scale = EmptyListIfNone(scale_opt);
  const auto offset = EmptyListIfNone(offset_opt);
  const auto antiquant_scale = EmptyListIfNone(antiquant_scale_opt);
  const auto antiquant_offset = EmptyListIfNone(antiquant_offset_opt);
  const auto per_token_scale = EmptyListIfNone(per_token_scale_opt);
  const auto activation_input = EmptyListIfNone(activation_input_opt);
  const auto activation_quant_scale = EmptyListIfNone(activation_quant_scale_opt);
  const auto activation_quant_offset = EmptyListIfNone(activation_quant_offset_opt);

  if (group_list_vector_opt.has_value()) {
    const auto &group_list = group_list_vector_opt.value();
    auto y = InferLegacyOutputs(x, weight, group_list, split_item, out_dtype);
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmul");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmul, x, weight, bias, scale, offset, antiquant_scale,
                                            antiquant_offset, group_list, split_item, y));
    auto run_inputs = CollectRunInputs(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, {}, std::nullopt,
                                       {}, {}, {});
    runner->Run(run_inputs, y);
    return y;
  }

  const bool v4_available = IsV4Available();
  if (!v4_available) {
    Check(!group_list_tensor_opt.has_value(),
          "group_list tensor requires aclnnGroupedMatmulV4 or newer; please use group_list vector or update CANN.");
    const std::vector<int64_t> empty_group_list;
    auto y = InferLegacyOutputs(x, weight, empty_group_list, split_item, out_dtype);
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmul");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmul, x, weight, bias, scale, offset, antiquant_scale,
                                            antiquant_offset, empty_group_list, split_item, y));
    auto run_inputs = CollectRunInputs(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, {}, std::nullopt,
                                       {}, {}, {});
    runner->Run(run_inputs, y);
    return y;
  }

  Check(group_type_opt.has_value(), "Requires manual passing group_type, current is None.");
  const int64_t group_type = group_type_opt.value();
  Check(group_type == kDefaultSplit || group_type == kMSplit || group_type == kKSplit,
        "The group_type must be -1, 0 or 2.");

  const bool mxfp4_valid = IsMxfp4Valid(x, weight, x_acl_dtype_opt, weight_acl_dtype_opt);
  auto y = InferTensorGroupListOutputs(x, weight, group_list_tensor_opt, split_item, group_type, mxfp4_valid, out_dtype);
  const int64_t group_list_type = OptIntValue(group_list_type_opt, 0);
  const int64_t act_type = OptIntValue(act_type_opt, 0);
  const auto tuning_config = tuning_config_opt.value_or(std::vector<int64_t>{});
  TensorList act_out;
  TensorList dynamic_quant_scale_out;

  auto launch_weight = UnifyWeightShapeForInt4(weight);
  const bool need_dtype_wrappers = x_acl_dtype_opt.has_value() || weight_acl_dtype_opt.has_value() ||
                                   scale_acl_dtype_opt.has_value() || per_token_scale_acl_dtype_opt.has_value();
  const auto x_wrapper = MakeTensorListWithDType(x, AclDTypeValue(x_acl_dtype_opt, x[0].data_type()));
  const auto weight_wrapper =
    MakeTensorListWithDType(launch_weight, AclDTypeValue(weight_acl_dtype_opt, weight[0].data_type()));
  const auto scale_wrapper = MakeOptionalTensorListWithDType(
    scale_opt, AclDTypeValue(scale_acl_dtype_opt, scale.empty() ? ms::TypeId::kNumberTypeUInt64 : scale[0].data_type()));
  const auto per_token_scale_wrapper = MakeOptionalTensorListWithDType(
    per_token_scale_opt,
    AclDTypeValue(per_token_scale_acl_dtype_opt,
                  per_token_scale.empty() ? ms::TypeId::kNumberTypeFloat32 : per_token_scale[0].data_type()));
  const auto antiquant_scale_wrapper = MakeTensorListWithDType(antiquant_scale, antiquant_scale.empty()
                                                                                 ? ACL_FLOAT16
                                                                                 : AclDTypeOf(antiquant_scale[0].data_type()));

  const bool weight_nz = IsWeightNz(weight[0]);
  auto run_inputs = CollectRunInputs(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale,
                                     group_list_tensor_opt, activation_input, activation_quant_scale,
                                     activation_quant_offset);
  if (weight_nz) {
    Check(IsWeightNzAvailable(), "FRACTAL_NZ weight requires aclnnGroupedMatmulWeightNz.");
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulWeightNz");
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulWeightNz, x_wrapper, weight_wrapper, bias, scale_wrapper,
                                            offset, antiquant_scale_wrapper, antiquant_offset, per_token_scale_wrapper,
                                            group_list_tensor_opt, activation_input, activation_quant_scale,
                                            activation_quant_offset, split_item, group_type, group_list_type, act_type,
                                            tuning_config, kQuantPerGroupSize, y, act_out, dynamic_quant_scale_out));
    runner->Run(run_inputs, y);
    return y;
  }

  const bool dtype_valid = !IsFp8Type(x[0].data_type()) && !x_acl_dtype_opt.has_value() && !weight_acl_dtype_opt.has_value();
  if (!IsV5Available() || !dtype_valid || mxfp4_valid) {
    auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulV4");
    if (need_dtype_wrappers) {
      runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulV4, x_wrapper, weight_wrapper, bias, scale_wrapper,
                                              offset, antiquant_scale, antiquant_offset, per_token_scale_wrapper,
                                              group_list_tensor_opt, activation_input, activation_quant_scale,
                                              activation_quant_offset, split_item, group_type, group_list_type, act_type,
                                              y, act_out, dynamic_quant_scale_out));
    } else {
      runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulV4, x, launch_weight, bias, scale, offset,
                                              antiquant_scale, antiquant_offset, per_token_scale, group_list_tensor_opt,
                                              activation_input, activation_quant_scale, activation_quant_offset,
                                              split_item, group_type, group_list_type, act_type, y, act_out,
                                              dynamic_quant_scale_out));
    }
    runner->Run(run_inputs, y);
    return y;
  }

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("GroupedMatmulV5");
  if (need_dtype_wrappers) {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulV5, x_wrapper, weight_wrapper, bias, scale_wrapper, offset,
                                            antiquant_scale, antiquant_offset, per_token_scale_wrapper,
                                            group_list_tensor_opt, activation_input, activation_quant_scale,
                                            activation_quant_offset, split_item, group_type, group_list_type, act_type,
                                            tuning_config, y, act_out, dynamic_quant_scale_out));
  } else {
    runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGroupedMatmulV5, x, launch_weight, bias, scale, offset,
                                            antiquant_scale, antiquant_offset, per_token_scale, group_list_tensor_opt,
                                            activation_input, activation_quant_scale, activation_quant_offset,
                                            split_item, group_type, group_list_type, act_type, tuning_config, y, act_out,
                                            dynamic_quant_scale_out));
  }
  runner->Run(run_inputs, y);
  return y;
}

std::vector<mindspore::tensor::TensorPtr> ToTensorPtrVector(const TensorList &outputs) {
  std::vector<mindspore::tensor::TensorPtr> tensor_ptrs;
  tensor_ptrs.reserve(outputs.size());
  for (const auto &output : outputs) {
    tensor_ptrs.emplace_back(output.tensor());
  }
  return tensor_ptrs;
}

void ConvertStubArg(TensorList *tensors) {
  for (auto &tensor : *tensors) {
    ms::inner::ConvertStubNodeToTensor(tensor);
  }
}

void ConvertStubArg(std::optional<TensorList> *tensors) {
  if (tensors->has_value()) {
    ConvertStubArg(&tensors->value());
  }
}

void ConvertStubArg(std::optional<ms::Tensor> *tensor) {
  if (tensor->has_value()) {
    ms::inner::ConvertStubNodeToTensor(tensor->value());
  }
}

template <typename T>
void ConvertStubArg(T *) {}

template <typename Tuple, size_t... I>
void ConvertStubArgs(Tuple *args, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(ConvertStubArg(&std::get<I>(*args)), 0)...};
}

size_t InferOutputNum(const TensorList &x, const TensorList &, const std::optional<TensorList> &,
                      const std::optional<TensorList> &, const std::optional<TensorList> &,
                      const std::optional<TensorList> &, const std::optional<TensorList> &,
                      const std::optional<TensorList> &, const std::optional<ms::Tensor> &group_list_tensor,
                      const std::optional<std::vector<int64_t>> &group_list_vector,
                      const std::optional<TensorList> &, const std::optional<TensorList> &,
                      const std::optional<TensorList> &, const std::optional<int64_t> &split_item_opt,
                      const std::optional<int64_t> &, const std::optional<int64_t> &, const std::optional<int64_t> &,
                      const std::optional<std::vector<int64_t>> &, const std::optional<int64_t> &,
                      const std::optional<int64_t> &, const std::optional<int64_t> &, const std::optional<int64_t> &,
                      const std::optional<int64_t> &) {
  const int64_t split_item = OptIntValue(split_item_opt, 0);
  if (IsSplitOutput(split_item)) {
    return 1;
  }
  if (group_list_vector.has_value() && !group_list_vector->empty()) {
    return group_list_vector->size();
  }
  if (group_list_tensor.has_value() && !group_list_tensor->shape().empty()) {
    return static_cast<size_t>(group_list_tensor->shape()[0]);
  }
  return x.size();
}

template <typename Ret, typename... Args, size_t... I>
py::object DynamicPyboostArgsCallerImpl(Ret (*func)(Args...), const py::args &args, std::index_sequence<I...>) {
  auto cast_args = std::make_tuple(args[I].cast<std::decay_t<Args>>()...);
  const size_t output_num = std::apply(InferOutputNum, cast_args);
  auto py_output = mindspore::tensor::MakeVector<false>(output_num);
  auto promises = mindspore::tensor::TransformVectorPromise(py_output);
  mindspore::pynative::DispatchOp(std::make_shared<mindspore::pynative::PassthroughFrontendTask>(
    [func, cast_args, promises]() mutable {
      ConvertStubArgs(&cast_args, std::make_index_sequence<sizeof...(Args)>());
      auto outputs = std::apply(func, cast_args);
      mindspore::tensor::SetPromise(promises, ToTensorPtrVector(outputs));
    },
    [promises]() { mindspore::tensor::SetException(promises); }));
  return py::reinterpret_steal<py::object>(mindspore::tensor::TransformVectorOutput(py_output));
}

template <typename Ret, typename... Args>
py::object DynamicPyboostArgsCaller(Ret (*func)(Args...), const py::args &args) {
  constexpr size_t n = sizeof...(Args);
  if (args.size() != n) {
    MS_LOG(EXCEPTION) << "Argument count mismatch: expected " << n << ", got " << args.size();
  }
  return DynamicPyboostArgsCallerImpl(func, args, std::make_index_sequence<n>());
}
}  // namespace

std::vector<ms::Tensor> npu_grouped_matmul(
  const TensorList &x, const TensorList &weight, const std::optional<TensorList> &bias,
  const std::optional<TensorList> &scale, const std::optional<TensorList> &offset,
  const std::optional<TensorList> &antiquant_scale, const std::optional<TensorList> &antiquant_offset,
  const std::optional<TensorList> &per_token_scale, const std::optional<ms::Tensor> &group_list_tensor,
  const std::optional<std::vector<int64_t>> &group_list_vector, const std::optional<TensorList> &activation_input,
  const std::optional<TensorList> &activation_quant_scale, const std::optional<TensorList> &activation_quant_offset,
  const std::optional<int64_t> &split_item, const std::optional<int64_t> &group_type,
  const std::optional<int64_t> &group_list_type, const std::optional<int64_t> &act_type,
  const std::optional<std::vector<int64_t>> &tuning_config, const std::optional<int64_t> &output_dtype,
  const std::optional<int64_t> &x_acl_dtype, const std::optional<int64_t> &weight_acl_dtype,
  const std::optional<int64_t> &scale_acl_dtype, const std::optional<int64_t> &per_token_scale_acl_dtype) {
  return npu_grouped_matmul_impl(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale,
                                 group_list_tensor, group_list_vector, activation_input, activation_quant_scale,
                                 activation_quant_offset, split_item, group_type, group_list_type, act_type,
                                 tuning_config, output_dtype, x_acl_dtype, weight_acl_dtype, scale_acl_dtype,
                                 per_token_scale_acl_dtype);
}
}  // namespace custom

namespace custom {
inline mindspore::device::ascend::aclTensor *ConvertTensorWithDType(
  const mindspore::tensor::TensorPtr &tensor, aclDataType acl_data_type) {
  MS_EXCEPTION_IF_NULL(tensor);
  static const auto aclCreateTensor = reinterpret_cast<mindspore::device::ascend::_aclCreateTensor>(
    mindspore::device::ascend::GetOpApiFunc("aclCreateTensor"));
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }
  const auto &shape = tensor->shape();
  aclFormat format = ACL_FORMAT_ND;
  if (shape.size() == 3) {
    format = ACL_FORMAT_NCL;
  } else if (shape.size() == 4) {
    format = ACL_FORMAT_NCHW;
  } else if (shape.size() == 5) {
    format = ACL_FORMAT_NCDHW;
  }
  auto device_address = tensor->device_address();
  if (device_address->size() != 0 && device_address->GetMutablePtr() == nullptr) {
    MS_LOG(EXCEPTION) << "The device memory is null, please allocate device memory for tensor " << tensor->ToString();
  }
  static const auto get_tensor_num = [](const std::vector<int64_t> &tensor_shape) {
    return std::accumulate(tensor_shape.begin(), tensor_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  };
  const auto &strides = tensor->stride();
  std::vector<int64_t> storage_shape;
  const auto &storage_info = tensor->storage_info();
  if (storage_info) {
    if (tensor->format() == mindspore::Format::FRACTAL_NZ) {
      format = ACL_FORMAT_FRACTAL_NZ;
      storage_shape = storage_info->ori_shape;
    } else {
      storage_shape = std::vector<int64_t>{get_tensor_num(storage_info->ori_shape)};
    }
  } else {
    storage_shape = std::vector<int64_t>{get_tensor_num(shape)};
  }
  return aclCreateTensor(shape.data(), shape.size(), acl_data_type, strides.data(), tensor->storage_offset(), format,
                         storage_shape.data(), storage_shape.size(), device_address->GetMutablePtr());
}

inline mindspore::device::ascend::aclTensorList *ConvertType(const TensorListWithDType &tensor_list_wrapper) {
  static const auto aclCreateTensorList = reinterpret_cast<mindspore::device::ascend::_aclCreateTensorList>(
    mindspore::device::ascend::GetOpApiFunc("aclCreateTensorList"));
  std::vector<mindspore::device::ascend::aclTensor *> tmp;
  tmp.reserve(tensor_list_wrapper.tensors.size());
  for (const auto &tensor : tensor_list_wrapper.tensors) {
    tmp.emplace_back(ConvertTensorWithDType(tensor, tensor_list_wrapper.dtype));
  }
  return aclCreateTensorList(tmp.data(), tmp.size());
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_grouped_matmul", [](const py::args &args) -> py::object {
    return custom::DynamicPyboostArgsCaller(custom::npu_grouped_matmul, args);
  });
}

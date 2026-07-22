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

#include <memory>
#include <vector>
#include "ms_extension/all.h"

namespace custom {
ms::Tensor npu_mul_add(const ms::Tensor &x, const ms::Tensor &y, const ms::Tensor &z) {
  auto tmp = ms::Tensor(x.data_type(), x.shape());
  auto out = ms::Tensor(x.data_type(), x.shape());

  auto mul_runner = std::make_shared<ms::pynative::AclnnOpRunner>("Mul");
  mul_runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMul, x, y, tmp));
  mul_runner->Run({x, y}, {tmp});

  constexpr int64_t alpha = 1;
  auto add_runner = std::make_shared<ms::pynative::AclnnOpRunner>("Add");
  add_runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAdd, tmp, z, alpha, out));
  add_runner->Run({tmp, z}, {out});

  return out;
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_mul_add", PYBOOST_CALLER(1, custom::npu_mul_add)); }

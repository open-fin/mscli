# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Custom Ops"""
import mindspore as ms


_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops",
    ["module.cc",
     "dense_lightning_indexer_grad_kl_loss.cc",
     "dense_lightning_indexer_softmax_lse.cc",
     "sparse_lightning_indexer_grad_kl_loss.cc",
     "mhc_post.cc",
     "mhc_post_backward.cc",
     "mhc_pre_sinkhorn.cc",
     "mhc_pre_sinkhorn_backward.cc"],
    backend="Ascend",
).load()


def npu_dense_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        softmax_max,
        softmax_sum,
        softmax_max_index,
        softmax_sum_index,
        scale_value,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout=None,
        sparse_mode=None,
        pre_tokens=None,
        next_tokens=None):
    """npu_dense_lightning_indexer_grad_kl_loss"""
    return _custom_ops.npu_dense_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        softmax_max,
        softmax_sum,
        softmax_max_index,
        softmax_sum_index,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )


def npu_dense_lightning_indexer_softmax_lse(
        query_index,
        key_index,
        weight,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout=None,
        sparse_mode=None,
        pre_tokens=None,
        next_tokens=None):
    """npu_dense_lightning_indexer_softmax_lse"""
    return _custom_ops.npu_dense_lightning_indexer_softmax_lse(
        query_index,
        key_index,
        weight,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens)


def npu_sparse_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout=None,
        sparse_mode=None,
        pre_tokens=None,
        next_tokens=None):
    """npu_sparse_lightning_indexer_grad_kl_loss"""
    return _custom_ops.npu_sparse_lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens)


def npu_mhc_post(x, h_res, h_out, h_post):
    """npu_mhc_post"""
    return _custom_ops.npu_mhc_post(x, h_res, h_out, h_post)


def npu_mhc_post_backward(grad_y, x, h_res, h_out, h_post):
    """npu_mhc_post_backward"""
    return _custom_ops.npu_mhc_post_backward(grad_y, x, h_res, h_out, h_post)


def npu_mhc_pre_sinkhorn(
        x,
        phi,
        alpha,
        bias,
        hc_mult=4,
        num_iters=20,
        hc_eps=1e-6,
        norm_eps=1e-6,
        out_flag=True):
    """npu_mhc_pre_sinkhorn."""
    return _custom_ops.npu_mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag)


def npu_mhc_pre_sinkhorn_backward(
        grad_h_in,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps=1e-6):
    """npu_mhc_pre_sinkhorn_backward"""
    return _custom_ops.npu_mhc_pre_sinkhorn_backward(
        grad_h_in,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps)

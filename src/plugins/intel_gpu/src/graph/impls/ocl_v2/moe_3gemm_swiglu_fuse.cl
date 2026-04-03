// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if SOFTMAX_TOPK_ENABLE

KERNEL(softmax_topk)(
    const __global MOE_DTYPE* input, // [input_batch, sort_in_num]
    __global uint* output_index, // [input_batch, TOP_K]
    __global MOE_DTYPE* output // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;

    uint sort_position = 0;

    __local MOE_DTYPE local_input[VALUE_NUM];
    __local MOE_DTYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

#if MOE_DTYPE_SIZE == 2
    MOE_DTYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
#elif MOE_DTYPE_SIZE == 4
    MOE_DTYPE in_value = as_float(intel_sub_group_block_read((const __global uint*)(input)));
#else
#    error "softmax_topk: unsupported MOE_DTYPE_SIZE"
#endif
    local_input[sort_index] = in_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        MOE_DTYPE value = local_input[i];
        if(value >= in_value) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        MOE_DTYPE value = local_input[i];
        if(value > in_value) {
            sort_position++;
        }
    }
    if (sort_position < TOP_K) {
        local_output[sort_position] = in_value;
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sort_position == 0) {
        float softmax_total = 1.0;
        MOE_DTYPE max_v = local_output[0];
        local_output[0] = 1;
        for(uint i = 1; i < TOP_K; i++) {
            local_output[i] = exp(local_output[i] - max_v);
            softmax_total += local_output[i];
        }
        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < TOP_K; i++) {
            output[i] = local_output[i]/softmax_total;
            output_index[i] = local_index[i];
        }
    }
}

#elif SIGMOID_BIAS_TOPK_ENABLE

KERNEL(sigmoid_bias_topk)(
    const __global MOE_DTYPE* input,    // routing logits [input_batch, num_experts] (unused, kept for arg ordering)
    const __global MOE_DTYPE* bias,     // routing bias [1, num_experts] or [num_experts]
    const __global MOE_DTYPE* eps_ptr,  // routing epsilon scalar [1]
    const __global MOE_DTYPE* hidden_states,  // [input_batch, hidden_dim] - for FP32 gate GEMV
#if GATE_WEIGHT_IS_F32
    const __global float*    gate_weight,     // [num_experts, hidden_dim] - dequantized (f32)
#else
    const __global MOE_DTYPE* gate_weight,    // [num_experts, hidden_dim] - dequantized (f16)
#endif
    __global uint* output_index,        // [input_batch, TOP_K]
    __global MOE_DTYPE* output          // [input_batch, TOP_K]
) {
    // gws [batch, num_experts]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);  // num_experts

    // Use float for sigmoid/selection to preserve precision during expert selection.
    // FP16 has only ~3.3 decimal digits — with many experts and close sigmoid scores,
    // FP16 sorting can select different experts than FP32, causing large output divergence.
    __local float local_sigmoid[VALUE_NUM];       // raw sigmoid values (float precision)
    __local float local_selection[VALUE_NUM];     // sigmoid + bias for sorting (float precision)
    __local float local_output[TOP_K];
    __local uint local_index[TOP_K];

    // Compute gate logit via FP32 dot product: dot(hidden_states[batch], gate_weight[sort_index])
    // This avoids FP16 accumulation error in the external gate MatMul (K=2048),
    // which causes ~0.07 absolute error — far exceeding the inter-expert gap (~0.0004).
    float gate_logit = 0.0f;
    const uint expert_offset = sort_index * GATE_HIDDEN_DIM;
    const uint batch_offset = batch * GATE_HIDDEN_DIM;
    for (uint i = 0; i < GATE_HIDDEN_DIM; i++) {
        float h = (float)hidden_states[batch_offset + i];
#if GATE_WEIGHT_IS_F32
        float w = gate_weight[expert_offset + i];
#else
        float w = (float)gate_weight[expert_offset + i];
#endif
        gate_logit += h * w;
    }
    float sigmoid_val = 1.0f / (1.0f + exp(-gate_logit));

    // Add bias for selection (determines which experts are chosen)
    float bias_val = (float)bias[sort_index];
    float selection_val = sigmoid_val + bias_val;

    local_sigmoid[sort_index] = sigmoid_val;
    local_selection[sort_index] = selection_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sort by selection_val (sigmoid + bias) to find top-K
    uint sort_position = 0;
    uint actual_topk = (TOP_K < sort_cnt) ? TOP_K : sort_cnt;

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        float value = local_selection[i];
        if(value >= selection_val) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        float value = local_selection[i];
        if(value > selection_val) {
            sort_position++;
        }
    }

    // Store raw sigmoid values (NOT sigmoid+bias) for the top-K experts
    if (sort_position < actual_topk) {
        local_output[sort_position] = local_sigmoid[sort_index];
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Normalize: weights / (sum + eps)
    if(sort_position == 0) {
        float sum_weights = 0.0f;
        for(uint i = 0; i < actual_topk; i++) {
            sum_weights += local_output[i];
        }
        sum_weights += (float)eps_ptr[0];  // epsilon to avoid division by zero

        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < actual_topk; i++) {
            output[i] = (MOE_DTYPE)(local_output[i] / sum_weights);
            output_index[i] = local_index[i];
        }
        // Zero out remaining positions if TOP_K > actual_topk
        for(uint i = actual_topk; i < TOP_K; i++) {
            output[i] = (MOE_DTYPE)0.0f;
            output_index[i] = 0;
        }
    }
}

#elif GATHER_ENABLE
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (gather_2d_ref)(
    const __global MOE_DTYPE* src_tok,       // input tokens [total_token, hidden_size] - hidden_states_mem_ptr
    const __global MOE_DTYPE* src_rweight,   // topk_weights [total_token, topk_experts]
    __global int * tok_index,                // token index [expert_idx][] = [actual_token_num]   - expert_mask_mem.batch
    __global int * top_index,                // topk  index [expert_idx][] = [actual_token_num]   - expert_mask_mem.topk
    __global MOE_DTYPE* dst_tok,             // output tokens [batch_size, hidden_size] - scratch.x
    __global MOE_DTYPE* dst_rweight) {       // output topk_weights [batch_size] - scratch.routing_weights

    int k = get_global_id(0);   // token_idx
    int off = get_global_id(1); // hidden_size offset
    int tok_idx = tok_index[k];

    src_tok += tok_idx * HIDDEN_SIZE;
    dst_tok += k * HIDDEN_SIZE;

    if (off >= HIDDEN_SIZE) {
        // printf("Warning off >= HIDDEN_SIZE: k = %d, off = %d, HIDDEN_SIZE = %d\n", k, off, HIDDEN_SIZE);
        return;
    }

    #if MOE_DTYPE_SIZE == 2
        ushort value = intel_sub_group_block_read_us((const __global ushort *)(src_tok + off));
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), value);
    #elif MOE_DTYPE_SIZE == 4
        uint value = intel_sub_group_block_read((const __global uint *)(src_tok + off));
        intel_sub_group_block_write((__global uint *)(dst_tok + off), value);
    #else
        dst_tok[off] = src_tok[off];
    #endif

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[top_idx];
    }
}

#elif SCATTER_ENABLE
// Accumulate expert outputs into an FP32 buffer to avoid FP16 truncation at each step.
// With 16 experts, accumulating in FP16 loses ~1-2 bits per step (16 steps = catastrophic).
// The FP32 buffer is converted to FP16 once after all experts are done (see SCATTER_F32_TO_F16_ENABLE).
KERNEL (index_add_)(const __global MOE_DTYPE* src_tok,
    __global int * tok_index,
    __global float* dst_tok_f32) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];

    src_tok += k * HIDDEN_SIZE;
    dst_tok_f32 += tok_idx * HIDDEN_SIZE;

    #if MOE_DTYPE_SIZE == 2
        float src_value = (float)as_half(intel_sub_group_block_read_us((const __global ushort *)(src_tok + off)));
        float dst_value = as_float(intel_sub_group_block_read((const __global uint *)(dst_tok_f32 + off)));
        float value = dst_value + src_value;
        intel_sub_group_block_write((__global uint *)(dst_tok_f32 + off), as_uint(value));
    #elif MOE_DTYPE_SIZE == 4
        float src_value = as_float(intel_sub_group_block_read((const __global uint *)(src_tok + off)));
        float dst_value = as_float(intel_sub_group_block_read((const __global uint *)(dst_tok_f32 + off)));
        float value = dst_value + src_value;
        intel_sub_group_block_write((__global uint *)(dst_tok_f32 + off), as_uint(value));
    #else
        dst_tok_f32[off] += (float)src_tok[off];
    #endif
}

#elif SCATTER_F32_TO_F16_ENABLE
// Final conversion: FP32 accumulation buffer → FP16 output
KERNEL (scatter_f32_to_f16)(const __global float* src_f32,
    __global MOE_DTYPE* dst) {

    int idx = get_global_id(0);
    int off = get_global_id(1);

    src_f32 += idx * HIDDEN_SIZE;
    dst += idx * HIDDEN_SIZE;

    #if MOE_DTYPE_SIZE == 2
        float value = as_float(intel_sub_group_block_read((const __global uint *)(src_f32 + off)));
        intel_sub_group_block_write_us((__global ushort *)(dst + off), as_ushort((half)value));
    #elif MOE_DTYPE_SIZE == 4
        float value = as_float(intel_sub_group_block_read((const __global uint *)(src_f32 + off)));
        intel_sub_group_block_write((__global uint *)(dst + off), as_uint(value));
    #else
        dst[off] = (MOE_DTYPE)src_f32[off];
    #endif
}

#elif PREFILL_SWIGLU_ENABLE

#define SWISH_BETA 1.0f
#define ACC_DTYPE float
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(swiglu_ref) (
    const __global MOE_DTYPE* up, // [token_len * expert_topK, inter_size]
    const __global MOE_DTYPE* gate,
    __global MOE_DTYPE* output    // [token_len * expert_topK, inter_size]
) {
    const uint token_idx = get_global_id(1);
    const uint n_offset = get_global_id(0);
    // gws = {_intermediate_size, token_cnt,  1}
    // lws = {subgroup_size, 1, 1};

#if MOE_DTYPE_SIZE == 2
    const uint sg_id = get_sub_group_local_id();
    const uint offset = token_idx * INTERMEDIA_SIZE + n_offset - sg_id;
    ACC_DTYPE up_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(up + offset)));
    ACC_DTYPE gate_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(gate + offset)));
    ACC_DTYPE value = gate_value / (1.0f + exp(-SWISH_BETA * gate_value));
    half result = value * up_value;
    intel_sub_group_block_write_us((__global ushort *)(output + offset), as_ushort(result));
#else
    const uint offset = token_idx * INTERMEDIA_SIZE + n_offset;
    ACC_DTYPE gate_value = gate[offset];
    ACC_DTYPE up_value = up[offset];
    ACC_DTYPE value = gate_value / (1.0f + exp(-SWISH_BETA * gate_value));
    ACC_DTYPE result = value * up_value;
    output[offset] = result;
#endif
}

#endif

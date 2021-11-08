// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_fuzzy.hpp"

#include <cnpy.h>

#include <fstream>

#include "engines_util/test_engines.hpp"
#include "ngraph/ngraph.hpp"
#include "paddle_utils.hpp"
#include "util/test_control.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace ngraph;
using namespace ngraph::frontend;
using TestEngine = test::IE_CPU_Engine;

using PDPDFuzzyOpTest = FrontEndFuzzyOpTest;

static const std::vector<std::string> models{std::string("argmax"),
                                             std::string("argmax1"),
                                             std::string("assign_value_boolean"),
                                             std::string("assign_value_fp32"),
                                             std::string("assign_value_int32"),
                                             std::string("assign_value_int64"),
                                             std::string("avgAdaptivePool2D_test1"),
                                             std::string("avgPool_test1"),
                                             std::string("avgPool_test10"),
                                             std::string("avgPool_test11"),
                                             std::string("avgPool_test2"),
                                             std::string("avgPool_test3"),
                                             std::string("avgPool_test4"),
                                             std::string("avgPool_test5"),
                                             // avgPool_test6<nchw support is disabled now>,
                                             std::string("avgPool_test7"),
                                             std::string("avgPool_test8"),
                                             std::string("avgPool_test9"),
                                             std::string("batch_norm_nchw"),
                                             std::string("batch_norm_nhwc"),
                                             std::string("bilinear_downsample_false_0"),
                                             std::string("bilinear_downsample_false_1"),
                                             std::string("bilinear_downsample_true_0"),
                                             std::string("bilinear_upsample_false_0"),
                                             std::string("bilinear_upsample_false_1"),
                                             std::string("bilinear_upsample_scales"),
                                             std::string("bilinear_upsample_scales2"),
                                             std::string("bilinear_upsample_true_0"),
                                             std::string("bmm"),
                                             std::string("clip"),
                                             std::string("conv2d_dilation_assymetric_pads_strides"),
                                             std::string("conv2d_SAME_padding"),
                                             std::string("conv2d_strides_assymetric_padding"),
                                             std::string("conv2d_strides_no_padding"),
                                             std::string("conv2d_strides_padding"),
                                             std::string("conv2d_transpose_dilation_assymetric_pads_strides"),
                                             // conv2d_transpose_SAME_padding(PDPD outputs wrong results),
                                             std::string("conv2d_transpose_strides_assymetric_padding"),
                                             std::string("conv2d_transpose_strides_no_padding"),
                                             std::string("conv2d_transpose_strides_padding"),
                                             std::string("conv2d_transpose_VALID_padding"),
                                             std::string("conv2d_VALID_padding"),
                                             std::string("cumsum"),
                                             std::string("cumsum_i32"),
                                             std::string("cumsum_i64"),
                                             std::string("cumsum_f32"),
                                             std::string("cumsum_f64"),
                                             std::string("depthwise_conv2d_convolution"),
                                             std::string("depthwise_conv2d_transpose_convolution"),
                                             std::string("dropout"),
                                             std::string("dropout_upscale_in_train"),
                                             std::string("elementwise_add1"),
                                             std::string("elementwise_div1"),
                                             std::string("elementwise_max1"),
                                             std::string("elementwise_min1"),
                                             std::string("elementwise_mul1"),
                                             std::string("elementwise_pow1"),
                                             std::string("elementwise_sub1"),
                                             std::string("embedding_0"),
                                             std::string("embedding_sparse"),
                                             std::string("embedding_none_weight"),
                                             std::string("embedding_paddings"),
                                             std::string("embedding_paddings_neg1"),
                                             std::string("embedding_tensorIds"),
                                             std::string("embedding_tensorIds_paddings"),
                                             std::string("equal"),
                                             std::string("expand_v2"),
                                             std::string("expand_v2_tensor"),
                                             std::string("expand_v2_tensor_list"),
                                             std::string("exp_test_float32"),
                                             std::string("fill_any_like"),
                                             std::string("fill_any_like_f16"),
                                             std::string("fill_any_like_f32"),
                                             std::string("fill_any_like_f64"),
                                             std::string("fill_any_like_i32"),
                                             std::string("fill_any_like_i64"),
                                             std::string("fill_constant"),
                                             std::string("fill_constant_batch_size_like"),
                                             std::string("fill_constant_int32"),
                                             std::string("fill_constant_int64"),
                                             std::string("fill_constant_tensor"),
                                             std::string("fill_constant_shape_tensor"),
                                             std::string("fill_constant_shape_tensor_list"),
                                             std::string("fill_zeros_like"),
                                             std::string("fill_zeros_like_f16"),
                                             std::string("fill_zeros_like_f32"),
                                             std::string("fill_zeros_like_f64"),
                                             std::string("fill_zeros_like_i32"),
                                             std::string("fill_zeros_like_i64"),
                                             std::string("flatten_contiguous_range_test1"),
                                             std::string("gelu_erf"),
                                             std::string("gelu_tanh"),
                                             // greater_equal_big_int64(failure due to CPU inference),
                                             std::string("greater_equal_float32"),
                                             std::string("greater_equal_int32"),
                                             std::string("greater_equal_int64"),
                                             std::string("hard_sigmoid"),
                                             std::string("hard_swish"),
                                             std::string("layer_norm"),
                                             std::string("layer_norm_noall"),
                                             std::string("layer_norm_noscale"),
                                             std::string("layer_norm_noshift"),
                                             std::string("leaky_relu"),
                                             std::string("log"),
                                             std::string("logical_not"),
                                             std::string("matmul_xt"),
                                             std::string("matmul_xt_yt"),
                                             std::string("matmul_yt"),
                                             std::string("matmul_v2_1dx1d"),
                                             std::string("matmul_v2_1dx2d"),
                                             std::string("matmul_v2_2dx1d"),
                                             std::string("matmul_v2_ndxmd"),
                                             std::string("matmul_v2_xt"),
                                             std::string("matmul_v2_xt_yt"),
                                             std::string("matmul_v2_yt"),
                                             std::string("maxAdaptivePool2D_test1"),
                                             std::string("maxPool_test1"),
                                             std::string("maxPool_test10"),
                                             std::string("maxPool_test11"),
                                             std::string("maxPool_test2"),
                                             std::string("maxPool_test3"),
                                             std::string("maxPool_test4"),
                                             std::string("maxPool_test5"),
                                             // maxPool_test6(nchw support is disabled now),
                                             std::string("maxPool_test7"),
                                             std::string("maxPool_test8"),
                                             std::string("maxPool_test9"),
                                             std::string("mul_fp32"),
                                             std::string("nearest_downsample_false_0"),
                                             std::string("nearest_downsample_false_1"),
                                             std::string("nearest_upsample_false_0"),
                                             std::string("nearest_upsample_false_1"),
                                             std::string("pad3d_test1"),
                                             std::string("pad3d_test2"),
                                             std::string("pad3d_test3"),
                                             // pad3d_test4,
                                             std::string("pow_float32"),
                                             std::string("pow_int32"),
                                             std::string("pow_int64"),
                                             // pow_int64_out_of_range(out of range of OV int64),
                                             std::string("pow_y_tensor"),
                                             std::string("prior_box_attrs_mmar_order_true"),
                                             std::string("prior_box_default"),
                                             std::string("prior_box_flip_clip_false"),
                                             std::string("prior_box_max_sizes_none"),
                                             std::string("range0"),
                                             std::string("range1"),
                                             std::string("range2"),
                                             std::string("relu"),
                                             std::string("relu6"),
                                             std::string("relu6_1"),
                                             std::string("reshape"),
                                             std::string("reshape_tensor"),
                                             std::string("reshape_tensor_list"),
                                             std::string("rnn_lstm_layer_1_bidirectional"),
                                             std::string("rnn_lstm_layer_1_forward"),
                                             std::string("rnn_lstm_layer_2_bidirectional"),
                                             std::string("rnn_lstm_layer_2_forward"),
                                             std::string("scale_bias_after_float32"),
                                             std::string("scale_bias_after_int32"),
                                             std::string("scale_bias_after_int64"),
                                             std::string("scale_bias_before_float32"),
                                             std::string("scale_bias_before_int32"),
                                             std::string("scale_bias_before_int64"),
                                             std::string("scale_tensor_bias_after"),
                                             std::string("scale_tensor_bias_before"),
                                             std::string("shape"),
                                             std::string("sigmoid"),
                                             std::string("slice"),
                                             std::string("slice_1d"),
                                             std::string("softmax"),
                                             std::string("softmax_minus"),
                                             std::string("split_test1"),
                                             std::string("split_test2"),
                                             std::string("split_test3"),
                                             std::string("split_test4"),
                                             std::string("split_test5"),
                                             std::string("split_test6"),
                                             std::string("split_test_dim_int32"),
                                             std::string("split_test_dim_int64"),
                                             std::string("split_test_list"),
                                             std::string("split_test_list_tensor"),
                                             std::string("squeeze"),
                                             std::string("squeeze_null_axes"),
                                             std::string("stack_test_float32"),
                                             std::string("stack_test_int32"),
                                             std::string("stack_test_neg_axis"),
                                             std::string("stack_test_none_axis"),
                                             std::string("tanh"),
                                             std::string("unsqueeze"),
                                             // Temporily disable them until root caused to secure CI stable.
                                             // CVS-66703 to track this.
                                             // std::string("yolo_box_clip_box"),
                                             // std::string("yolo_box_default"),
                                             // std::string("yolo_box_scale_xy"),
                                             std::string("yolo_box_uneven_wh")};

INSTANTIATE_TEST_SUITE_P(PDPDFuzzyOpTest,
                         FrontEndFuzzyOpTest,
                         ::testing::Combine(::testing::Values(PADDLE_FE),
                                            ::testing::Values(std::string(TEST_PADDLE_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         PDPDFuzzyOpTest::getTestCaseName);

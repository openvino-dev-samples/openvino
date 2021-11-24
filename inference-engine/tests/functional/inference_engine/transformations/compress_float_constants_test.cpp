// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include "openvino/core/function.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/compress_float_constants.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_subgraphs.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, CompressConstants_f32) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 12, 12 });
        auto const_weights = ov::opset8::Constant::create(ov::element::f32,
            ov::Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
            const_weights,
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        auto const_scales = ov::opset8::Constant::create(ov::element::f32, ov::Shape{ 1 }, { 1.4 });

        auto shape = std::make_shared<ov::opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<ov::opset8::Convert>(shape, ov::element::f32);
        auto mul = std::make_shared<ov::opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<ov::opset8::Convert>(mul, ov::element::i32);

        auto default_scales_node = ov::opset8::Constant::create(ov::element::f32, ov::Shape{ 4 }, { 1., 1., 1.4, 1.4 });
        auto axes_node = ov::opset8::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 0, 1, 2, 3 });

        auto interpolate4_attr = ov::opset8::Interpolate::InterpolateAttrs(ov::opset8::Interpolate::InterpolateMode::NEAREST,
            ov::opset8::Interpolate::ShapeCalcMode::SIZES, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
            ov::opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC, ov::opset8::Interpolate::NearestMode::SIMPLE,
            false, -0.75);

        auto resize = std::make_shared<ov::opset8::Interpolate>(conv, convert2, default_scales_node, axes_node, interpolate4_attr);

        f = std::make_shared<ov::Function>(ov::NodeVector{ resize }, ov::ParameterVector{ input });

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MarkPrecisionSensitiveSubgraphs>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 12, 12 });
        auto const_weights = ov::opset8::Constant::create(ov::element::f16,
            ov::Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
            convert_ins1,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
        auto const_scales = ov::opset8::Constant::create(ov::element::f32, ov::Shape{ 1 }, { 1.4 });

        auto shape = std::make_shared<ov::opset8::ShapeOf>(conv);
        auto convert1 = std::make_shared<ov::opset8::Convert>(shape, ov::element::f32);
        auto mul = std::make_shared<ov::opset8::Multiply>(convert1, const_scales);
        auto convert2 = std::make_shared<ov::opset8::Convert>(mul, ov::element::i32);

        auto default_scales_node = ov::opset8::Constant::create(ov::element::f32, ov::Shape{ 4 }, { 1., 1., 1.4, 1.4 });
        auto axes_node = ov::opset8::Constant::create(ov::element::i64, ov::Shape{ 4 }, { 0, 1, 2, 3 });

        auto interpolate4_attr = ov::opset8::Interpolate::InterpolateAttrs(ov::opset8::Interpolate::InterpolateMode::NEAREST,
            ov::opset8::Interpolate::ShapeCalcMode::SIZES, std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
            ov::opset8::Interpolate::CoordinateTransformMode::ASYMMETRIC, ov::opset8::Interpolate::NearestMode::SIMPLE,
            false, -0.75);

        auto resize = std::make_shared<ov::opset8::Interpolate>(conv, convert2, default_scales_node, axes_node, interpolate4_attr);

        f_ref = std::make_shared<ov::Function>(ov::NodeVector{ resize }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompressConstants_f64) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f64, ov::Shape{ 1, 3, 12, 12 });
        auto const_weights = ov::opset8::Constant::create(ov::element::f64,
            ov::Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
            const_weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
        f = std::make_shared<ov::Function>(ov::NodeVector{ conv }, ov::ParameterVector{ input });

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MarkPrecisionSensitiveSubgraphs>();
        manager.register_pass<ov::pass::CompressFloatConstants>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f64, ov::Shape{ 1, 3, 12, 12 });
        auto const_weights = ov::opset8::Constant::create(ov::element::f16,
            ov::Shape{ 1, 3, 3, 3 },
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f64);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
            convert_ins1,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

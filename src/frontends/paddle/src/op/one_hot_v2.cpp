// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs one_hot_v2(const NodeContext& node) {
    auto x = node.get_input("X");
    Output<Node> depth_expected_node;
    if (node.has_input("depth_tensor")) {
        auto depth_tensor = node.get_input("depth_tensor");
        auto depth_node = std::make_shared<default_opset::Convert>(depth_tensor, element::i32);
        depth_expected_node = std::make_shared<default_opset::Squeeze>(depth_node);
    } else {
        const auto depth_expected = node.get_attribute<int>("depth");
        depth_expected_node = default_opset::Constant::create(element::i32, {}, {depth_expected});
    }

    bool allow_out_of_range = node.get_attribute<bool>("allow_out_of_range", false);
    PADDLE_OP_CHECK(node, allow_out_of_range == false, "allow_out_of_range can not be true in one_hot_v2.");
    
    const auto on_value = default_opset::Constant::create(element::i32, Shape{}, {1});
    const auto off_value = default_opset::Constant::create(element::i32, Shape{}, {0});
    int64_t axis = -1;
    auto node_onehot = std::make_shared<default_opset::OneHot>(x, depth_expected_node, on_value, off_value, axis);
    return node.default_single_output_mapping({std::make_shared<default_opset::Convert>(node_onehot, element::f32)}, {"Out"});

}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

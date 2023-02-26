// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <limits.h>
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
Output<Node> idx_node(const std::string& tensor_alias,
                      const std::string& list_alias,
                      const std::string& attr_alias,
                      const NodeContext& node) {
    if (node.has_input(tensor_alias)) {
        return std::make_shared<default_opset::Convert>(node.get_input(tensor_alias), element::i64);
    } else if (node.has_input(list_alias)) {
        auto inputs = node.get_ng_inputs(list_alias);
        return std::make_shared<default_opset::Convert>(std::make_shared<default_opset::Concat>(inputs, 0),
                                                        element::i64);
    } else {
        auto values = node.get_attribute<std::vector<int64_t>>(attr_alias);
        return default_opset::Constant::create(element::i64, {values.size()}, values);
    }
}

NamedOutputs set_value(const NodeContext& node) {
    const auto data = node.get_input("Input");
    const auto value_tensor = node.get_input("ValueTensor");


    Output<Node> start_idx_node = idx_node("StartsTensor", "StartsTensorList", "starts", node);
    Output<Node> end_idx_node = idx_node("EndsTensor", "EndsTensorList", "ends", node);

    const auto shape_node = std::make_shared<default_opset::ShapeOf>(data, element::Type_t::i64);
    const auto rank_node = std::make_shared<default_opset::ShapeOf>(shape_node, element::i64);
    std::shared_ptr<Node> axes_node;
    if (node.has_attribute("axes")) {
        const auto axes = node.get_attribute<std::vector<int64_t>>("axes");
        axes_node = default_opset::Constant::create(element::i64, {axes.size(), 1}, axes);

    }
    else {
        const auto axes = std::make_shared<default_opset::Range>(default_opset::Constant::create<int64_t>(element::i64, {}, {0}),
                                                  rank_node,
                                                  default_opset::Constant::create<int64_t>(element::i64, {}, {1}),
                                                  element::i64);
        axes_node = std::make_shared<default_opset::Squeeze>(axes);
        // axes_node = std::make_shared<default_opset::Unsqueeze>(axes, default_opset::Constant::create<int64_t>(element::i64, {1}, {1}));
    }


    const auto const_0_node = default_opset::Constant::create(element::i64, {}, {0});
    const auto const_max_node = default_opset::Constant::create(element::i64, {}, {INT_MAX});
    const auto const_1_node = default_opset::Constant::create(element::i64, {}, {1});
    const auto start_node = std::make_shared<default_opset::Broadcast>(const_0_node, rank_node);
    const auto end_node = std::make_shared<default_opset::Broadcast>(const_max_node, rank_node);

    const auto pads_begin_node =
        std::make_shared<default_opset::ScatterNDUpdate>(start_node, axes_node, start_idx_node);
    const auto fixed_end_node = std::make_shared<default_opset::ScatterNDUpdate>(end_node, axes_node, end_idx_node);
    const auto pads_end_node = std::make_shared<default_opset::Subtract>(shape_node, fixed_end_node);
    const auto value_pad_node = std::make_shared<default_opset::Pad>(value_tensor, pads_begin_node, pads_end_node, const_max_node, ov::op::PadMode::CONSTANT);
    const auto condition_node = std::make_shared<default_opset::Equal>(value_pad_node, const_max_node);
    const auto set_value_node = std::make_shared<default_opset::Select>(condition_node, data, value_pad_node,  ov::op::AutoBroadcastType::NUMPY);
    return node.default_single_output_mapping({set_value_node}, {"Out"});
}  // namespace
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

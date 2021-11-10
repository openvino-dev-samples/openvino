// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs top_k(const NodeContext& node) {
    auto x = node.get_ng_input("X");

    Output<Node> k;
    NamedOutputs named_outputs;
    if (node.has_ng_input("K")) {
        auto k_variable = node.get_ng_input("K");
        k = std::make_shared<ngraph::opset6::Convert>(k_variable, ngraph::element::i32);
    }else {
        int32_t k_expected;
        if (node.has_attribute<int32_t>("k")) {
            k_expected = node.get_attribute<int32_t>("k");
        } else {
            // k default value is 1
            k_expected = 1;
        }
        k = ngraph::opset6::Constant::create(ngraph::element::i32, {}, {k_expected});
    }
    // last axis
    int32_t axis = -1;
    const element::Type& index_element_type = element::i64;
    auto node_topk_results = std::make_shared<ngraph::opset6::TopK>(x, k, axis, "max", "value", index_element_type);

    named_outputs["Out"] = {node_topk_results->output(0)};
    named_outputs["Indices"] = {node_topk_results->output(1)};

    return named_outputs;

}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph

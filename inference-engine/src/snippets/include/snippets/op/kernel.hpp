// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Kernel
 * @brief Generated by Canonicalization and represents compute kernel legal for sheduling
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Kernel : public ngraph::op::Op {
public:
    OPENVINO_OP("Kernel", "SnippetsOpset");

    Kernel(const std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>>& region);
    Kernel() = default;

    std::vector<std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo>> region;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<Kernel>(region);
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
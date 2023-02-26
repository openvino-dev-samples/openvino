# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
 
import os
import sys

import numpy as np
import paddle

from save_model import exportModel

'''
assign w/ ouput
'''
@paddle.jit.to_static
def set_value(Input, StartsTensor, EndsTensor, ValueTensor):
    #tensor = paddle.to_tensor(input)
    Output = Input.clone()
    Output[StartsTensor:EndsTensor] = ValueTensor
    return Output


in_dtype = 'int64'
array = paddle.assign([[1, 1, 1],[1, 1, 1]]).astype(in_dtype)
start = paddle.assign([0, 0]).astype(in_dtype)
end = paddle.assign([0,1]).astype(in_dtype)
value = paddle.assign([[2, 2], [2, 2]]).astype(in_dtype)
exportModel('set_value', set_value, [array,start,end,value], target_dir=sys.argv[1])
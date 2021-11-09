#
# expand paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def expand(name:str, x, shape:list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        out = paddle.fluid.layers.expand(node_x, shape, name='expand')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def expand_tensor(name:str, x, shape:list, use_tensor_in_list=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            shape[0] = paddle.assign(np.array((shape[0],)).astype('int32'))
            out = paddle.fluid.layers.expand(node_x, shape, name='expand')
        else:
            shape = np.array(shape).astype('int32')
            node_shape = paddle.assign(shape, output=None)
            out = paddle.fluid.layers.expand(node_x, node_shape, name='expand')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    data = np.random.rand(6, 2).astype(data_type)

    expand("expand", data, [2, 3])
    expand_tensor("expand_tensor", data, [2, 3], False)
    expand_tensor("expand_tensor_list", data, [2, 3], True)

if __name__ == "__main__":
    main()

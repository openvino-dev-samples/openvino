#
# expand paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
from paddle.fluid import layers
import sys

data_type = 'float32'

def expand(name:str, x, expand_times:list):
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        out = layers.expand(node_x, expand_times, name='expand')
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def expand_tensor(name:str, x, expand_times, use_tensor_in_list):
    pdpd.enable_static()
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            expand_times[0] = pdpd.assign(np.array((expand_times[0],)).astype('int32'))
            out = layers.expand(node_x, expand_times, name='expand')
        else:
            expand_times = np.array(expand_times).astype('int32')
            node_shape = pdpd.assign(expand_times, output=None)
            out = layers.expand(node_x, node_shape, name='expand')
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    data = np.random.rand(2, 3, 6).astype(data_type)

    expand("expand", data, [3, 2, 4])
    expand_tensor("expand_tensor", data, [3, 2, 4], False)
    expand_tensor("expand_tensor_list", data, [3, 2, 4], True)

if __name__ == "__main__":
    main()

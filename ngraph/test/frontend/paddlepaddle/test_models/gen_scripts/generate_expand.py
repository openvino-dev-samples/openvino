#
# expand paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys

data_type = 'float32'

def expand(name, x, times):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        out = pdpd.fluid.layers.expand(node_x, times, name='expand')

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

def expand_tensor(name, x, times, use_tensor_in_list=False):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            times[0] = pdpd.assign(np.array((times[0],)).astype('int32'))
            out = pdpd.fluid.layers.expand(node_x, times, name='expand')
        else:
            times = np.array(times).astype('int32')
            node_shape = pdpd.assign(times, output=None)
            out = pdpd.fluid.layers.expand(node_x, node_shape, name='expand')

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
    data = np.random.rand(5, 2, 1).astype(data_type)

    expand("expand", data, [1, 2, 5])
    expand_tensor("expand_tensor", data, [1, 2, 5], False)
    expand_tensor("expand_tensor_list", data, [1, 2, 5], True)

if __name__ == "__main__":
    main()

#
# not_equal paddle model generator
#
import numpy as np
import paddle as pdpd
from save_model import saveModel
import sys

data_type = 'float32'

def not_equal(name : str, x, y):
    pdpd.enable_static()

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
    node_y = pdpd.static.data(name='y', shape=y.shape, dtype=data_type)

    out = pdpd.not_equal(node_x, node_y)
    out = pdpd.cast(out, data_type)

    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())

    outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])

    saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out],
              inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_x = np.array([[[[-5, -1, 3]], [[1, 9, 2]]]]).astype(data_type)
    data_y = np.array([[[[1, -1, 2]], [[3, -4, 9]]]]).astype(data_type)

    not_equal("not_equal", data_x, data_y)


if __name__ == "__main__":
    main()

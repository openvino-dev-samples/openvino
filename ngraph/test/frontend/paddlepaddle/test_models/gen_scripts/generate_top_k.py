#
# top_k paddle model generator
#
import numpy as np
import paddle as pdpd
from save_model import saveModel
import sys

data_type = 'float32'

def top_k(name : str, x, k:int):
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        value, indices = pdpd.fluid.layers.topk(node_x, k, name="top_k")
        indices = pdpd.cast(indices, data_type)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[value, indices])
            
        saveModel(name, exe, feedkeys=['x'], fetchlist=[value, indices], inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]

def main():
    data = np.random.random([5, 2, 10]).astype(data_type)
    # top_k k=5
    top_k("top_k_5", data, k=5)
    # top_k k=2
    top_k("top_k_2", data, k=2)
    # top_k k=4
    top_k("top_k_4", data, k=4)


if __name__ == "__main__":
    main()     

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
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        value, idx = pdpd.fluid.layers.topk(node_x, k, name="top_k")
        idx = pdpd.cast(idx, np.float32)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[value, idx])
            
        saveModel(name, exe, feedkeys=['x'], fetchlist=[value, idx], inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]

def main():
    data = np.random.random([3, 5]).astype("float32")
    
    top_k("top_k_test_2", data, k=2)
    top_k("top_k_test_3", data, k=3)
    top_k("top_k_test_4", data, k=4)


if __name__ == "__main__":
    main()     

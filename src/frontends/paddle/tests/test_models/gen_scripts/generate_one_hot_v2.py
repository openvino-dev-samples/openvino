#
# one_hot_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def one_hot_v2(name: str, x, depth: int, depth_is_var=True):

    paddle.enable_static()

    depth = np.array([depth], dtype='int32') if depth_is_var else depth

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        input_depth = paddle.static.data(name='depth', shape=[1], dtype='int32') if depth_is_var else depth
        out = paddle.nn.functional.one_hot(node_x, input_depth, name="one_hot")

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        feed_list = {'x': x, 'depth': depth} if depth_is_var else {'x': x}
        outs = exe.run(
            feed=feed_list,
            fetch_list=[out])

        feedkey_list = ['x', 'depth'] if depth_is_var else ['x']
        input_list = [x, depth] if depth_is_var else [x]
        saveModel(name, exe, feedkeys=feedkey_list, fetchlist=[out], inputs=input_list, outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def main():
    data_1d = np.array([1, 1, 3, 0]).astype("int32")
    data_2d = np.array([[2, 0, 3], [3, 1, 4]]).astype("int64")
    one_hot_v2("one_hot_v2_test_1", data_1d, depth=4)
    one_hot_v2("one_hot_v2_test_2", data_2d, depth=5)

if __name__ == "__main__":
    main()

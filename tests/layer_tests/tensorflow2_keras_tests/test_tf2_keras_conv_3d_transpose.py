# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasConv3DTranspose(CommonTF2LayerTest):
    def create_keras_conv_3d_transpose_net(self, conv_params, input_names, input_shapes, input_type,
                                           ir_version):
        activation_func_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.nn.<activation> operation have no "==" operation to be compared
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid
        }
        if "activation" in conv_params:
            conv_params["activation"] = activation_func_structure[conv_params["activation"]]

        # create TensorFlow 2 model with Keras Conv3DTranspose operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type,
                           name=input_names[0])  # Variable-length sequence of ints

        y = tf.keras.layers.Conv3DTranspose(**conv_params, input_shape=input_shapes[0][1:])(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        pytest.param(
            dict(conv_params=dict(filters=27, kernel_size=3, padding="valid", strides=(1, 1, 2)),
                 input_names=["x"], input_shapes=[[5, 3, 5, 7, 6]], input_type=tf.float32),
            marks=pytest.mark.precommit),
        pytest.param(
            dict(conv_params=dict(filters=10, kernel_size=5, padding="same", strides=(3, 4, 5),
                                  activation="relu", use_bias=True, output_padding=2),
                 input_names=["x"], input_shapes=[[5, 3, 5, 7, 8]], input_type=tf.float32),
            marks=pytest.mark.precommit),

        pytest.param(dict(conv_params=dict(filters=27, kernel_size=3, data_format="channels_first"),
                          input_names=["x"], input_shapes=[[5, 3, 5, 7, 6]], input_type=tf.float32),
                     marks=pytest.mark.xfail(reason="49529")),

        dict(conv_params=dict(filters=10, kernel_size=5, padding="same", strides=(4, 3, 2),
                              output_padding=1),
             input_names=["x"], input_shapes=[[5, 3, 5, 7, 8]], input_type=tf.float32),
        dict(conv_params=dict(filters=27, kernel_size=3, padding="valid", dilation_rate=3),
             input_names=["x"],
             input_shapes=[[5, 8, 9, 7, 6]], input_type=tf.float32),
        dict(conv_params=dict(filters=10, kernel_size=3, padding="same", dilation_rate=4),
             input_names=["x"],
             input_shapes=[[5, 3, 9, 7, 8]], input_type=tf.float32),
        dict(conv_params=dict(filters=27, kernel_size=3, padding="valid", dilation_rate=3,
                              activation="sigmoid",
                              use_bias=False), input_names=["x"], input_shapes=[[5, 8, 9, 7, 6]],
             input_type=tf.float32),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_conv_3D_transpose_float32(self, params, ie_device, precision, ir_version,
                                             temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_keras_conv_3d_transpose_net(**params, ir_version=ir_version),
                   ie_device, precision,
                   temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

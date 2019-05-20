import os
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils
from keras.optimizers import Adam

from unet3d.metrics import minh_dice_coef_metric
from unet3d.utils.model_utils import compile_model

from brats.config import config, config_unet


def create_3d_convolution_block(input_layer, n_filters, batch_normalization=True, kernel=(3, 3, 3), activation=None,
                                padding='same', strides=(1, 1, 1), instance_normalization=False,
                                is_unet_original=True):

    layer = Conv3D(n_filters, kernel, padding=padding,
                   strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    if activation is None:
        layer = Activation('relu')(layer)

    return layer


def SegNet3D(input_shape, n_labels, n_base_filters=16, depth=4, pool_size=(2, 2, 2), loss_function="weighted",
             metrics=minh_dice_coef_metric, initial_learning_rate=1e-4):

    img_input = Input(input_shape)
    current_layer = img_input

    for layer_depth in range(depth):
        if layer_depth < (depth/2):
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = MaxPooling3D(pool_size=pool_size)(current_layer)
        else:
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = MaxPooling3D(pool_size=pool_size)(current_layer)

    for layer_depth in range(depth-1, -1, -1):
        if layer_depth >= (depth/2):
            current_layer = UpSampling3D(size=pool_size)(current_layer)
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
        else:
            current_layer = UpSampling3D(size=pool_size)(current_layer)
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))
            current_layer = create_3d_convolution_block(input_layer=current_layer,
                                                        n_filters=n_base_filters*(2**layer_depth))

    current_layer = Conv3D(n_labels, (1, 1, 1))(current_layer)
    current_layer = Activation("sigmoid")(current_layer)
    model = Model(inputs=img_input, outputs=current_layer)

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


def main():

    model = SegNet3D(input_shape=(1, 128, 128, 128),
                     n_labels=3)
    model.summary()


if __name__ == "__main__":
    main()

from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation
from keras.layers import SpatialDropout3D, Conv3D, BatchNormalization
from keras.engine import Model
from keras.optimizers import Adam

from .unet import concatenate
from ..metrics import weighted_dice_coefficient_loss, tversky_loss
from ..metrics import minh_dice_coef_loss, minh_dice_coef_metric

from keras.utils import multi_gpu_model
from keras import regularizers
from unet3d.utils.model_utils import compile_model
from unet3d.model.unet_vae import GroupNormalization

import tensorflow as tf
import external.gradient_checkpointing.memory_saving_gradients as memory_saving_gradients
# from tensorflow.python.keras._impl.keras import backend as K
import tensorflow.keras.backend as K
# K.__dict__["gradients"] = memory_saving_gradients.gradients_memory
# K.__dict__["gradients"] = memory_saving_gradients.gradients_speed


def mnet(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
         n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
         loss_function="weighted", activation_name="sigmoid", metrics=minh_dice_coef_metric):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 challenge:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf

    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(
                current_layer, n_level_filters, strides=(2, 2, 2))
            in_conv = SpatialDropout3D(rate=0.3,
                                       data_format="channels_first")(in_conv)

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number])
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(
                0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    # layer_names = [layer.name for layer in model.layers]
    # [tf.add_to_collection("checkpoints", model.get_layer(l).get_output_at(0))
    #     for l in [i for i in layer_names if 'conv3d' in i]]
    # K.__dict__[
    #     "gradients"] = memory_saving_gradients.gradients_collection

    return compile_model(model, loss_function=loss_function,
                         metrics=metrics,
                         initial_learning_rate=initial_learning_rate)


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=LeakyReLU,
                             padding='same', strides=(1, 1, 1), normalization="Batch"):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel,
                   padding=padding,
                   strides=strides,
                   # kernel_regularizer=regularizers.l2(l=1e-4))(input_layer)#doesn't work
                   )(input_layer)
    if normalization == " Batch":
        layer = BatchNormalization(axis=1)(layer)
    elif normalization == " Instance":
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    else:
        layer = GroupNormalization(groups=8, axis=1)(layer)
    if activation is None:
        layer = Activation('relu')(layer)
    else:
        layer = activation()(layer)
    return layer


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(
        convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(
        input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(
        rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(
        input_layer=dropout, n_filters=n_level_filters)
    return convolution2

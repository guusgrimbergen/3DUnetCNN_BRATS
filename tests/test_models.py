from unet3d.model import se_unet_3d
from unet3d.model import densefcn_model_3d
from unet3d.model import isensee2017_model
from unet3d.model import unet_model_3d
from unet3d.model import dense_unet_3d
from unet3d.model import res_unet_3d

from unet2d.model import unet_model_2d
from unet25d.model import unet_model_25d

from keras.utils import plot_model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_contrib.applications import densenet
import sys
import os
import keras.backend as K

from keras.models import Model

import sys
sys.path.append('external/Fully-Connected-DenseNets-Semantic-Segmentation')

save_dir = "doc/"


def save_plot(model, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
        print(">> remove", save_path)
    plot_model(model, to_file=save_path, show_shapes=True)
    print(">> save plot to", save_path)


def get_path(name):
    return save_dir + name + ".png"


K.set_image_data_format('channels_first')

input_shape = (4, 128, 128, 128)


# name = "unet3d"
# model = unet_model_3d(input_shape=(4, 128, 128, 128),
#                       n_labels=3,
#                       depth=4,
#                       n_base_filters=16,
#                       is_unet_original=True)
# model.summary()
# save_plot(model, get_path(name))


# name = "seunet3d"
# model = unet_model_3d(input_shape=(4, 128, 128, 128),
#                       n_labels=3,
#                       depth=4,
#                       n_base_filters=16,
#                       is_unet_original=False)
# model.summary()
# save_plot(model, get_path(name))


# name = "unet2d"
# model = unet_model_2d(input_shape=(4, 128, 128),
#                       n_labels=3,
#                       depth=4,
#                       n_base_filters=32,
#                       batch_normalization=True,
#                       is_unet_original=True)
# model.summary()
# save_plot(model, get_path(name))


# name = "seunet2d"
# model = unet_model_2d(input_shape=(4, 128, 128),
#                       n_labels=3,
#                       depth=4,
#                       n_base_filters=32,
#                       batch_normalization=True,
#                       is_unet_original=False)
# model.summary()
# save_plot(model, get_path(name))


name = "unet25d"
model = unet_model_25d(input_shape=(4, 160, 192, 7),
                       n_labels=3,
                       depth=4,
                       n_base_filters=16,
                       batch_normalization=False,
                       is_unet_original=True)
model.summary()
save_plot(model, get_path(name))
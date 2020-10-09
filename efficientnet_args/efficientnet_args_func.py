#!/usr/bin/env python
# coding: utf-8

# efficientnet_args_func.py

"""
Optionally loads weights pre-trained on ImageNet. Note that the data format convention used by the model 
is the one specified in your Keras config at `~/.keras/keras.json`.

Model Characteristics
The EfficientNet effectively conducts the model scaling and balance network depth, width, and resolution 
for the better performance. The reserchers uniformly scales all dimensions by using a simple but effective 
compound coefficient.

EfficientNet B0~B7
Based on the effectiveness of both MobileNets and ResNet, NasNet(neural architecture search) is uded to 
design a new baseline network and scale it up to obtain a family of models, called EfficientNets B0-B7, 
which achieve better accuracy and efficiency.

Modifications
It is a complicated realization based on the origianl realization with Choltt. Make the necessary changes 
to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. 
In addition, write the new lines of code to replace the deprecated code. I would like to thank Francois 
Chollet, Björn Barz, Pavel Yakubovskiyand related creators and interptretors for their contributions.

Reference
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
https://arxiv.org/abs/1905.11946
"""

import os
import math
import string
import collections

import warnings
import numpy as np
import tensorflow as tf 

from keras.preprocessing import image
from keras.layers import add, multiply
from keras.layers import Conv2D, Input, Dense, Dropout, Reshape, Activation, DepthwiseConv2D, \
    BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.models import Model
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}

# Change the origial dict to the namedtuple 
Args = collections.namedtuple('Args', 
    ['kernel_size', 'repeats', 'filters_in', 'filters_out',
     'expand_ratio', 'id_skip', 'strides', 'se_ratio'])

# Default a public argument for namedtuple
Args.__new__.__defaults__ = (None,) * len(Args._fields)

# Add Args to organize the arguments but it is too complex 
DEFAULT_BLOCKS_ARGS = [
    Args(kernel_size=3, repeats=1, filters_in=32, filters_out=16,
         expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    Args(kernel_size=3, repeats=2, filters_in=16, filters_out=24,
         expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    Args(kernel_size=5, repeats=2, filters_in=24, filters_out=40,
         expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    Args(kernel_size=3, repeats=3, filters_in=40, filters_out=80,
         expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    Args(kernel_size=5, repeats=3, filters_in=80, filters_out=112,
         expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    Args(kernel_size=5, repeats=4, filters_in=112, filters_out=192,
         expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    Args(kernel_size=3, repeats=1, filters_in=192, filters_out=320,
         expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def correct_pad(K, inputs, kernel_size):
    # Return a tuple for zero-padding for 2D convolution with downsampling.
    """
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 1 if K.image_data_format() == 'channels_last' else 2
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0]%2, 1 - input_size[1]%2)

    correct = (kernel_size[0]//2, kernel_size[1]//2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def get_swish(**kwargs):
    def swish(x):
        # Swish activation function: x * sigmoid(x).
        if K.backend() == 'tensorflow':
            try:
                # Implement a more memory-efficient gradient
                return tf.nn.swish(x)
            except AttributeError:
                pass
        return x * K.sigmoid(x)

    return swish


def round_filters(filters, width_coefficient, depth_divisor):
    # Get the round number of filters based on the width multiplier
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Ensure the round-down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    # Get the round number of repeats based on depth multiplier

    return int(math.ceil(depth_coefficient * repeats))


# Adopt the argument of args to replace the arguments in the above DEFAULT_BLOCKS_ARGS
def block(inputs, args, activation, drop_rate=None, prefix='', ):
    # Define the mobile inverted residual bottleneck.
    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1 # Assign bn_axis = -1 or 1

    # The expansion phase
    filters = args.filters_in * args.expand_ratio
    if args.expand_ratio != 1:
        x = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False, 
                   kernel_initializer=CONV_KERNEL_INITIALIZER, 
                   name=prefix + 'expand_conv')(inputs)
        x = BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Conduct the depthwise convolution
    x = DepthwiseConv2D(args.kernel_size, strides=args.strides, padding='same', use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER, name=prefix + 'dwconv')(x)
    x = BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < args.se_ratio <= 1:
        filters_se = max(1, int(args.filters_in*args.se_ratio))
        se = GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        if bn_axis == -1:  # It is equality: bn_axis == -1
            se = Reshape((1,1,filters), name=prefix + 'se_reshape')(se)
        else: 
            se = Reshape((filters,1,1), name=prefix + 'se_reshape')(se)

        se = Conv2D(filters_se, kernel_size=(1,1), activation=activation, padding='same', use_bias=True,
                    kernel_initializer=CONV_KERNEL_INITIALIZER, name=prefix + 'se_reduce')(se)
        se = Conv2D(filters, kernel_size=(1,1), activation='sigmoid', padding='same', use_bias=True,
                    kernel_initializer=CONV_KERNEL_INITIALIZER, name=prefix + 'se_expand')(se)
        
        if K.backend() == 'theano':
            # For the Theano, make the excitation weights broadcastable explicitly.
            if K.image_data_format() == 'channels_last':
                pattern = [True, True, True, False]
            else: 
                pattern = [True, False, True, True]
            se = Lambda(lambda x: K.pattern_broadcast(x, pattern), name=prefix + 'se_broadcast')(se)
        x = multiply([x, se], name=prefix + 'se_excite')

    # Output phase
    x = Conv2D(args.filters_out, kernel_size=(1,1), padding='same', use_bias=False, 
               kernel_initializer=CONV_KERNEL_INITIALIZER, name=prefix + 'project_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if args.id_skip and all(s == 1 for s in args.strides) and args.filters_in == args.filters_out:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate, noise_shape=(None,1,1,1), name=prefix + 'drop')(x)
        x = add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient, depth_coefficient, default_resolution, dropout_rate=0.2,
                 drop_connect_rate=0.2, depth_divisor=8, blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet', include_top=True, weights='imagenet', 
                 input_tensor=None, input_shape=None, pooling=None, num_classes=1000, **kwargs):
    # Instantiate the EfficientNet architecture using given scaling coefficients.
    """
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization), 'imagenet' or the path to any weights.
        input_tensor: optional Keras tensor (output of `layers.Input()`)
        input_shape: tuple, only to be specified if `include_top` is False.
        pooling: Optional mode for feature extraction when `include_top` is `False`.
            - `None`: the output of model is the 4D tensor of the last conv layer 
            - `avg` means global average pooling and the output as a 2D tensor.
            - `max` means global max pooling will be applied.
        classes: specified if `include_top` is True
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
    """
    if not (weights in {'imagenet', 'noisy-student', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and num_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=default_resolution, 
                                      min_size=32, data_format=K.image_data_format(), 
                                      require_flatten=include_top, weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = -1 if K.image_data_format() == 'channels_last' else 1
    
    # Call the function of get_swish()
    activation = get_swish(**kwargs)

    # Build the stem
    x = img_input
    x = Conv2D(round_filters(32,width_coefficient,depth_divisor), 
               kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, 
               kernel_initializer=CONV_KERNEL_INITIALIZER, name='stem_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = Activation(activation, name='stem_activation')(x)

    # Build the blocks
    blocks = sum(args.repeats for args in blocks_args)
    block_num = 0
    for (i, args) in enumerate(blocks_args):
        assert args.repeats > 0
        # Update block input and output filters based on depth multiplier.
        args = args._replace(
            filters_in=round_filters(args.filters_in, width_coefficient, depth_divisor),
            filters_out=round_filters(args.filters_out, width_coefficient, depth_divisor),
            repeats=round_repeats(args.repeats, depth_coefficient))

        # The first block needs to take care of stride and filter size growth.
        drop_rate = drop_connect_rate * float(block_num) / blocks
        x = block(x, args, activation=activation, drop_rate=drop_rate, prefix='block{}a_'.format(i+1))
        block_num += 1
        if args.repeats > 1:
            args = args._replace(filters_in=args.filters_out, strides=[1,1])
            for bi in range(args.repeats - 1): # bi is block index 
                drop_rate = drop_connect_rate * float(block_num) / blocks
                block_prefix = 'block{}{}_'.format(i+1, string.ascii_lowercase[bi+1])
                # Call the function of block()
                x = block(x, args, activation=activation, drop_rate=drop_rate, prefix=block_prefix)
                block_num += 1

    # Build the top
    x = Conv2D(round_filters(1280, width_coefficient, depth_divisor), 
               kernel_size=(1,1), padding='same', use_bias=False, 
               kernel_initializer=CONV_KERNEL_INITIALIZER, name='top_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = Activation(activation, name='top_activation')(x)
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = Dropout(dropout_rate, name='top_dropout')(x)
        x = Dense(num_classes, activation='softmax', 
                  kernel_initializer=DENSE_KERNEL_INITIALIZER, name='probs')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        # -inputs = keras_utils.get_source_inputs(input_tensor)
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        else:
            file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suff
        weights_path = get_file(file_name, BASE_WEIGHTS_PATH + file_name,
                                cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.0, 1.0, 224, 0.2, model_name='efficientnet-b0',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes,
                        **kwargs)


def EfficientNetB1(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.0, 1.1, 240, 0.2, model_name='efficientnet-b1',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


def EfficientNetB2(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.1, 1.2, 260, 0.3, model_name='efficientnet-b2',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


def EfficientNetB3(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.2, 1.4, 300, 0.3, model_name='efficientnet-b3',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


def EfficientNetB4(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.4, 1.8, 380, 0.4, model_name='efficientnet-b4',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


def EfficientNetB5(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.6, 2.2, 456, 0.4, model_name='efficientnet-b5',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


def EfficientNetB6(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(1.8, 2.6, 528, 0.5, model_name='efficientnet-b6',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


def EfficientNetB7(include_top=True, weights='imagenet', input_tensor=None,
                   input_shape=None, pooling=None, num_classes=1000, **kwargs):

    return EfficientNet(2.0, 3.1, 600, 0.5, model_name='efficientnet-b7',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, num_classes=num_classes, **kwargs)


setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)

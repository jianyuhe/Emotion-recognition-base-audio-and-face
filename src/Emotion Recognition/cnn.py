#Importing Libraries
##Deep Learning 
from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D,DepthwiseConv2D,Dense,Input,Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization,Flatten,Conv2D,AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D,Activation,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Reshape,Add,Concatenate

import keras.backend as K


#Build Tensorflow 2.0 keras MobileNet model

##Depthwise Separable block for MobileNet
def depthwise_separable_block(x, nb_filter, stride=(1, 1), name=None):

    x = DepthwiseConv2D((3,3), padding='same', strides=stride, depth_multiplier=1, use_bias=False, name=name+'_dpconv')(x)
    x = BatchNormalization(axis=3, name=name+'_bn1')(x)
    x = Activation(relu6, name=name+'_relu1')(x)

    x = Conv2D(nb_filter, (1,1), padding='same', use_bias=False, strides=(1,1), name=name+'conv_2')(x)
    x = BatchNormalization(axis=3, name=name+'_bn2')(x)
    x = Activation(relu6, name=name+'_relu2')(x)

    return x

##Conv block for Mobilenet, it a standard 3x3 convolution block
def conv_block (x, nb_filter, stride=(1,1), name=None):

    x = Conv2D(nb_filter, (3,3), strides=stride, padding='same', use_bias=False, name=name+'_conv1')(x)
    x = BatchNormalization(axis=3, name=name+'bn1')(x)
    x = Activation(relu6, name=name+'relu')(x)

    return x

##The ReLu6 activation function
def relu6(x):
    return K.relu(x, max_value=6)

##MobileNet
def mobileNet (num_classes, input_size=(112,112,3), dropout=0.5):

    input = Input(shape=input_size)

    x = conv_block(input, 32, (2,2), name='conv_block')

    x = depthwise_separable_block(x, 64, stride=(1,1), name='dep1')
    x = depthwise_separable_block(x, 128, stride=(2,2), name='dep2')
    x = depthwise_separable_block(x, 128, stride=(1,1), name='dep3')
    x = depthwise_separable_block(x, 256, stride=(2,2), name='dep4')
    x = depthwise_separable_block(x, 256, stride=(1,1), name='dep5')
    x = depthwise_separable_block(x, 512, stride=(2,2), name='dep6')

    x = depthwise_separable_block(x, 512, stride=(1,1), name='dep7')
    x = depthwise_separable_block(x, 512, stride=(1,1), name='dep8')
    x = depthwise_separable_block(x, 512, stride=(1,1), name='dep9')
    x = depthwise_separable_block(x, 512, stride=(1,1), name='dep10')
    x = depthwise_separable_block(x, 512, stride=(1,1), name='dep11')

    x = depthwise_separable_block(x, 1024, stride=(2,2), name='dep12')
    x = depthwise_separable_block(x, 1024, stride=(1,1), name='dep13')

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(num_classes, (1,1), padding='same', name='conv_preds')(x)
    x = Reshape((num_classes,), name='reshape_2')(x)
    x = Activation('softmax', name='act_softmax')(x)

    model = Model(input, x, name='MobileNet')

    return model
    

#Build ms_model_R and ms_model_M models

##Mish activation function
def mish(x):
    return x * K.tanh(K.softplus(x))

##The Original residual cell Bottleneck_R and modified inverted residual cell Bottleneck_M
def bottleneck(input, in_channels, out_channels, strides, channels_expand, activation='Mish'):
    feature_m = Conv2D(filters=in_channels*channels_expand, kernel_size=(1,1), strides=strides, padding='same')(input)
    feature_m = Activation(mish)(feature_m) if activation == 'Mish' else layers.ReLU(6.)(feature_m)
    feature_m = DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same')(feature_m)
    feature_m = Activation(mish)(feature_m) if activation == 'Mish' else layers.ReLU(6.)(feature_m)
    feature_m = Conv2D(filters=out_channels, kernel_size=(1,1), strides=1, padding='same')(feature_m)
    feature_m_res = Conv2D(filters=out_channels, kernel_size=(1,1), strides=strides, padding='same')(input)
    feature_m = Add()([feature_m, feature_m_res])
    return feature_m

## Process of feature selection module
def feature_selection_modeule(input, pool_size, strides, out_channels):
    avg_pool_output = AveragePooling2D(pool_size=pool_size, strides=strides)(input)
    max_pool_output = MaxPooling2D(pool_size=pool_size, strides=strides)(input)
    sum_output = Add()([avg_pool_output, max_pool_output])
    output = Conv2D(filters= out_channels, kernel_size=(1,1), strides=1, padding='same')(sum_output)
    output = GlobalAveragePooling2D()(output)
    return output


## ms_model_R model
def ms_model_R(num_classes, input_size=(112,112,3), dropout_rate=0.5):

    input = Input(shape=input_size)

    # Initial feature extraction
    feature_init = Conv2D(filters= 16, kernel_size=(3,3), strides=2, padding='same')(input)

    # Bottleneck R1
    feature_m1 = bottleneck(feature_init, 16, 16, 1, 1, 'Relu6')

    # Bottleneck R2
    feature_m2 = bottleneck(feature_m1, 16, 24, 2, 5, 'Relu6')

    # Bottleneck R3
    feature_m3 = bottleneck(feature_m2, 24, 24, 1, 5, 'Relu6')

    # Bottleneck R3_1
    feature_m3_1 = bottleneck(feature_m3, 24, 32, 1, 5, 'Relu6')

    # Bottleneck R3_2
    feature_m3_2 = bottleneck(feature_m3_1, 32, 32, 1, 5, 'Relu6')

    # Feature selection module 1
    fsm_1 = feature_selection_modeule(feature_m3_2, (4,4), 4, 32)

    # Bottleneck R4
    feature_m4 = bottleneck(feature_m3, 24, 32, 2, 5, 'Relu6')

    # Bottleneck R5
    feature_m5 = bottleneck(feature_m4, 32, 32, 1, 5, 'Relu6')

    # Feature selection module 2
    fsm_2 = feature_selection_modeule(feature_m5, (2,2), 2, 32)

    # Bottleneck R6
    feature_m6 = bottleneck(feature_m5, 32, 40, 1, 5, 'Relu6')

    # Bottleneck R7
    feature_m7 = bottleneck(feature_m6, 40, 40, 1, 5, 'Relu6')

    # Feature selection module 3
    fsm_3 = feature_selection_modeule(feature_m7, (2,2), 2, 40)

    # Bottleneck R8
    feature_m8 = bottleneck(feature_m7, 40, 40, 1, 5, 'Relu6')

    # Bottleneck R9
    feature_m9 = bottleneck(feature_m8, 40, 48, 2, 5, 'Relu6')

    # Bottleneck R10
    feature_m10 = bottleneck(feature_m9, 48, 64, 1, 5, 'Relu6')

    fs = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same')(feature_m10)
    fs = GlobalAveragePooling2D()(fs)

    # Concat
    output = Concatenate()([fsm_1, fsm_2, fsm_3, fs])
    output = Reshape((1,1,-1))(output)
    output = Dropout(dropout_rate)(output)
    output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=1, padding='same')(output)
    output = Activation('softmax')(output)
    output = Flatten()(output)

    # Model
    model = Model(inputs=input, outputs=output, name='ms_model_R')

    return model

## ms_model_M model
def ms_model_M(num_classes, input_size=(112,112,3), dropout_rate=0.5):

    input = Input(shape=input_size)

    # Initial feature extraction
    feature_init = Conv2D(filters= 16, kernel_size=(3,3), strides=2, padding='same')(input)

    # Bottleneck M1
    feature_m1 = bottleneck(feature_init, 16, 16, 1, 1, 'Mish')

    # Bottleneck M2
    feature_m2 = bottleneck(feature_m1, 16, 24, 2, 5, 'Mish')

    # Bottleneck M3
    feature_m3 = bottleneck(feature_m2, 24, 24, 1, 5, 'Mish')

    # Bottleneck M3_1
    feature_m3_1 = bottleneck(feature_m3, 24, 32, 1, 5, 'Mish')

    # Bottleneck M3_2
    feature_m3_2 = bottleneck(feature_m3_1, 32, 32, 1, 5, 'Mish')

    # Feature selection module 1
    fsm_1 = feature_selection_modeule(feature_m3_2, (4,4), 4, 32)

    # Bottleneck M4
    feature_m4 = bottleneck(feature_m3, 24, 32, 2, 5, 'Mish')

    # Bottleneck M5
    feature_m5 = bottleneck(feature_m4, 32, 32, 1, 5, 'Mish')

    # Feature selection module 2
    fsm_2 = feature_selection_modeule(feature_m5, (2,2), 2, 32)

    # Bottleneck M6
    feature_m6 = bottleneck(feature_m5, 32, 40, 1, 5, 'Mish')

    # Bottleneck M7
    feature_m7 = bottleneck(feature_m6, 40, 40, 1, 5, 'Mish')

    # Feature selection module 3
    fsm_3 = feature_selection_modeule(feature_m7, (2,2), 2, 40)

    # Bottleneck M8
    feature_m8 = bottleneck(feature_m7, 40, 40, 1, 5, 'Mish')

    # Bottleneck M9
    feature_m9 = bottleneck(feature_m8, 40, 48, 2, 5, 'Mish')

    # Bottleneck M10
    feature_m10 = bottleneck(feature_m9, 48, 64, 1, 5, 'Mish')

    fs = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same')(feature_m10)
    fs = GlobalAveragePooling2D()(fs)

    # Concat
    output = Concatenate()([fsm_1, fsm_2, fsm_3, fs])
    output = Reshape((1,1,-1))(output)
    output = Dropout(dropout_rate)(output)
    output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=1, padding='same')(output)
    output = Activation('softmax')(output)
    output = Flatten()(output)

    # Model
    model = Model(inputs=input, outputs=output, name = 'ms_model_M')

    return model





import tensorflow as tf

from keras.models import Model, Input, Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, \
                         MaxPooling2D, Flatten, Dense


def conv2_block(input, filters):

    out = Conv2D(filters, 3, strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, 3, strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(pool_size=(2,2))(out)

    return out


def conv4_block(input, filters):

    out = Conv2D(filters, 3, strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, 3, strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, 3, strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, 3, strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(pool_size=(2,2))(out)

    return out


def VGG19(input_shape=(224,224,3), class_num = 2):
    '''
    VGG Functional的实现
    '''

    filters = [64, 128, 256, 512, 512]

    inputs = Input(shape=input_shape)

    # layer_1
    conv1 = conv2_block(inputs, filters[0])

    # layer_2
    conv2 = conv2_block(conv1, filters[1])

    # layer_3
    conv3 = conv4_block(conv2, filters[2])

    # layer_4
    conv4 = conv4_block(conv3, filters[3])

    # layer_5
    conv5 = conv4_block(conv4, filters[4])

    # 拉平
    conv6 = Flatten(name='flatten')(conv5)

    conv7 = Dense(units=4096, name='dense4096_1')(conv6)
    conv7 = Dense(units=4096, name='dense4096_2')(conv7)
    conv7 = Dense(units=class_num, name='dense1000', activation='softmax')(conv7)

    model = Model(inputs=inputs, outputs=conv7)

    return model


def VGG():
    '''
    VGG Sequential顺序模型的实现
    '''

    model = Sequential()
    
    # layer_1
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',data_format='channels_last',kernel_initializer='uniform',activation='relu'))
    model.add(MaxPooling2D((2,2)))

    #layer_2
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(128,(2,2),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D((2,2)))

    #layer_3
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #layer_4
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))

    #layer_5
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1000,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    return model


if __name__ == '__main__':
    model = VGG19()
    model.summary() 
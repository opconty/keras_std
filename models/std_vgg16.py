#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 上午11:20
#
# @Author: Greg Gao(laygin)
#'''
'''adapted from advanced east: many thanks to the original author'''
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input,Concatenate, Conv2D,UpSampling2D, BatchNormalization, Deconv2D


class HEAD():
    def __init__(self, num_scale=1):
        self.num_scale = num_scale

    def __call__(self, feats):
        feats = Conv2D(256, (3, 3), padding='same', activation='relu')(feats)
        feats = BatchNormalization()(feats)

        center_cls = Conv2D(1, (1, 1), activation='sigmoid', name='center_cls')(feats)
        scale_regr = Conv2D(self.num_scale, (1, 1), activation='linear', name='scale')(feats)

        offset = Conv2D(2, (1, 1), activation='linear', name='offset')(feats)
        angle = Conv2D(1, (1, 1), activation='linear', name='angle')(feats)

        # for head, no mater what the condition, return all anyway
        return center_cls, scale_regr, offset, angle


class StdVGG16():
    '''
    class Semantic Text Detection definition
    '''
    def __init__(self,
                 input_shape=(None, None, 3),
                 locked_layers=False,
                 feature_layers_range=range(5, 1, -1),
                 mode='deconv'):
        self.mode = mode
        self.inp = Input(name='inp', shape=input_shape, dtype='float32')
        vgg16 = VGG16(input_tensor=self.inp, weights='imagenet', include_top=False)
        if locked_layers:
            locked_ls = [vgg16.get_layer('block1_conv1'),
                         vgg16.get_layer('block1_conv2')]
            for l in locked_ls:
                l.trainable = False
        self.feature_layers_range = feature_layers_range
        self.feature_layers_num = len(self.feature_layers_range)
        self.f = [vgg16.get_layer('block%d_pool' % i).output for i in self.feature_layers_range]
        self.f.insert(0, None)
        self.diff = self.feature_layers_range[0] - self.feature_layers_num

    @staticmethod
    def up_2x(x, unit, mode='deconv'):
        assert mode in ['deconv', 'upsample']
        if mode == 'deconv':
            x = Deconv2D(unit, kernel_size=4, strides=2, padding='same')(x)
            return x
        elif mode == 'upsample':
            return UpSampling2D(size=(2, 2))(x)

    def h(self, i):
        if i==1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i-1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128//2**(i-2), 1, activation='relu', padding='same')(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_2 = Conv2D(128//2**(i-2), 3, activation='relu',padding='same')(bn2)

            return conv_2

    def g(self, i):
        if i == self.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return self.up_2x(self.h(i), int(128//2**(i-2)), self.mode)

    def std_net(self):
        out = self.g(self.feature_layers_num)
        center_cls, scale_regr, offset, angle = HEAD(1)(out)

        m = Model(inputs=self.inp, outputs=[center_cls, scale_regr, offset])

        conv2d_6 = m.get_layer('conv2d_6').output
        x = self.up_2x(conv2d_6, 32, self.mode)
        stage1 = m.get_layer('block1_pool').output
        x = Concatenate(axis=-1)([x, stage1])
        bn1 = BatchNormalization()(x)
        conv_1 = Conv2D(32, 1, activation='relu', padding='same')(bn1)
        bn2 = BatchNormalization()(conv_1)
        conv_2 = Conv2D(32, 3, activation='relu', padding='same')(bn2)
        x = self.up_2x(conv_2, 32, self.mode)
        x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)

        center_line = Conv2D(1, kernel_size=1, strides=1, activation='sigmoid', name='cl')(x)
        outs = m.outputs
        return Model(m.input, [center_line] + outs)


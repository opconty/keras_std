#-*- coding:utf-8 -*-
#'''
# Created on 19-5-11 下午3:56
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from models.losses import cls_center, regr_h, regr_offset, cl
from data import IcdarDataset, IcdarConfig
from data import SynthTextDataset, SynthTextConfig
from data.img_aug import RandomContrast,RandomBrightness
from models import StdVGG16


dataset = 'synthText'  # synthText, icdar13
if dataset == 'synthText':
    cfg = SynthTextConfig()  # SynthTextConfig()
elif dataset == 'icdar13':
    cfg = IcdarConfig()
else:
    raise Exception(f'dataset not defined: {dataset}')

cfg.input_size = 384  # 384,512,608,768
cfg.batch_size = 8
cfg.model = 'vgg16'
cfg.mode = 'caffe'

vggmode = 'deconv'  # only for vgg16,  'deconv', 'upsample'
cfg.center_line_stride = 1  # 1, 4
optimizer = 'adam'  # sgd, adam
lr = 1e-4

cfg.shrink_ratio = 1.2
cfg.center_line_version = 0

random_contrast = RandomContrast()
random_bright = RandomBrightness()
augs = [random_bright, random_contrast]

pre_weights = os.path.join(cfg.checkpoints_dir, 'aaaaa')
init_ep = 0


def create_callbacks(mm, monitor='loss'):
    dd = os.path.join(cfg.checkpoints_dir, cfg.Name)
    if not os.path.exists(dd):
        os.mkdir(dd)

    nn = '%s__ep{epoch:02d}_{loss:.3f}.h5' % (mm)
    checkpoint = ModelCheckpoint(os.path.join(dd, nn),
                                 monitor=monitor,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1)
    earlystop = EarlyStopping(patience=10, monitor=monitor, verbose=1)
    reduce = ReduceLROnPlateau(monitor=monitor, patience=2)

    return [checkpoint, earlystop, reduce]


def _main():
    if dataset == 'synthText':
        datagen_train = SynthTextDataset(cfg, shuffle=True, augs=augs)
    elif dataset == 'icdar13':
        datagen_train = IcdarDataset(cfg, shuffle=True, augments=augs)
    else:
        raise Exception('dataset does not defined..')

    print('datagen train length: ', len(datagen_train) * cfg.batch_size)
    M = StdVGG16(mode=vggmode)
    model = M.std_net()

    print('count_params: ', model.count_params())
    if os.path.exists(pre_weights):
        print('using pretrained weights: ', pre_weights)
        model.load_weights(pre_weights)
    else:
        print('training from scratch...')

    if optimizer == 'adam':
        opt = Adam(lr=lr)
    else:
        opt =SGD(lr=lr, momentum=0.99, decay=1e-6)

    loss = [cl, cls_center, regr_h, regr_offset]

    model.compile(optimizer=opt,
                  loss=loss)

    model.fit_generator(datagen_train,
                        steps_per_epoch=len(datagen_train),
                        epochs=cfg.epochs + init_ep,
                        initial_epoch=init_ep,
                        callbacks=create_callbacks(mm=M.__class__.__name__),
                        verbose=1
                        )


if __name__ == '__main__':

    _main()


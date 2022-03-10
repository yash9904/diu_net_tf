import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Cropping2D, Input,
                                     MaxPooling2D, UpSampling2D, ZeroPadding2D,
                                     concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DilatedInceptionUNetModel:
    def __init__(self, input=(444, 444, 3), filters=32, print_summary=True):
        self.input = input
        self.compile = True
        self.filters = filters
        self.print_summary = print_summary

    def aji_score(self, y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))

    def dice_coef(self, y_true, y_pred, smooth = 0):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def iou(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_pred_f * y_true_f)
        return intersection/(K.sum(y_pred_f) + K.sum(y_true_f) - intersection)

    def dice_coef_loss(self, y_true, y_pred):
        return -K.log(self.dice_coef(y_true, y_pred))

    def di_block(self, inputs, fn, crop_size=0, batch_norm=True, upsampling=False, downsampling=False):

        f1 = Conv2D(fn, (1, 1), activation='relu')(inputs)
        f1 = Conv2D(fn//2, (3, 3), activation='relu', dilation_rate=(1, 1),
                    padding='valid', kernel_initializer='he_normal')(f1)
        f1 = Conv2D(fn//2, (3, 3), activation='relu', dilation_rate=(1, 1),
                    padding='valid', kernel_initializer='he_normal')(f1)

        f2 = Conv2D(fn, (1, 1), activation='relu')(inputs)
        f2 = Conv2D(fn//2, (3, 3), activation='relu', dilation_rate=(2, 2),
                    padding='valid', kernel_initializer='he_normal')(f2)

        f3 = Conv2D(fn, (1, 1), activation='relu')(inputs)
        f3 = ZeroPadding2D(1)(f3)
        f3 = Conv2D(fn//2, (3, 3), activation='relu', dilation_rate=(3, 3),
                    padding='valid', kernel_initializer='he_normal')(f3)

        filter_concat = concatenate([f1, f2, f3], axis=-1)

        if batch_norm:
            output = BatchNormalization()(filter_concat)
        else:
            output = filter_concat

        if downsampling:
            skip_out = Cropping2D(crop_size)(output)

        if upsampling or downsampling:
            if upsampling:
                output = UpSampling2D((2, 2))(output)
            elif downsampling:
                output = MaxPooling2D((2, 2), 2)(output)
        elif upsampling and downsampling:
            print('You cant downsample and upsample at the same time!!')
            raise Exception
        if downsampling:
            return output, skip_out
        else:
            return output

    def diu_net_model(self):

        inputs = Input(self.input)

        di_block_1, skip_1 = self.di_block(
            inputs, self.filters, 88, batch_norm=True, upsampling=False, downsampling=True)
        di_block_2, skip_2 = self.di_block(
            di_block_1, self.filters*2, 40, batch_norm=True, upsampling=False, downsampling=True)
        di_block_3, skip_3 = self.di_block(
            di_block_2, self.filters*3, 16, batch_norm=True, upsampling=False, downsampling=True)
        di_block_4, skip_4 = self.di_block(
            di_block_3, self.filters*4, 4, batch_norm=True, upsampling=False, downsampling=True)

        di_block_5 = self.di_block(
            di_block_4, self.filters*5, batch_norm=True, upsampling=True, downsampling=False)

        di_block_6 = self.di_block(concatenate(
            [di_block_5, skip_4], axis=-1), self.filters*4, batch_norm=True, upsampling=True, downsampling=False)
        di_block_7 = self.di_block(concatenate(
            [di_block_6, skip_3], axis=-1), self.filters*3, batch_norm=True, upsampling=True, downsampling=False)
        di_block_8 = self.di_block(concatenate(
            [di_block_7, skip_2], axis=-1), self.filters*2, batch_norm=True, upsampling=True, downsampling=False)
        di_block_9 = self.di_block(concatenate(
            [di_block_8, skip_1], axis=-1), self.filters, batch_norm=True, upsampling=False, downsampling=False)

        output = Conv2D(1, (1, 1), activation='sigmoid')(di_block_9)

        diunet = Model(inputs, output)

        if self.print_summary:
            print(diunet.summary())

        if self.compile:
            diunet.compile(optimizer=Adam(), loss=[
                           self.dice_coef_loss], metrics=[self.dice_coef, self.iou, self.aji_score])

        return diunet


    def train(self, X, Y, validation_split= 0.2, batch_size=8, epochs=10):
        self.model = self.diu_net_model()
        self.model_history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose = 1)

        return self.model, self.model_history

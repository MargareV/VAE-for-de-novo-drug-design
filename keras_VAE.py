import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator


img_rows, img_cols, img_chns = 300, 300, 3
latent_dim = 20
intermediate_dim = 128
epsilon_std = 1.0
epochs = 15
filters = 32
num_conv = 3
batch_size = 256

# encoder
x = Input(shape=(300, 300, 3))
conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu')(x)
conv_2 = Conv2D(filters, kernel_size=(2, 2,), padding='same', activation='relu', strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_2)
conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', strides=1)(conv_3)

flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

# mu and std_dev for latent variables
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

# sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters*150*150, activation='relu')

output_shape = (batch_size, 150, 150, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
decoder_deconv2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu')
decoder_deconv3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')
decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv1(reshape_decoded)
deconv_2_decoded = decoder_deconv2(deconv_1_decoded)
x_decoded_relu = decoder_deconv3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# defining loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

datagen = ImageDataGenerator()

train_set = datagen.flow_from_directory('/home/margs/Drug dicovery and machine learning/Images_zinc/Train', target_size=(300, 300))
validation_set = datagen.flow_from_directory('/home/margs/Drug dicovery and machine learning/Images_zinc/validation', target_size=(300, 300))

# training
history = vae.fit_generator(train_set,
        shuffle=True,
        epochs=epochs,
        steps_per_epoch=batch_size,
        validation_data=validation_set,
        validation_steps=50)

# encoder from learned model
encoder = Model(x, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)
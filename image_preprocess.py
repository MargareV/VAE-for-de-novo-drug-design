import os
from keras.preprocessing import image as image_utils
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import pickle

inp_dir = '/home/margs/Drug dicovery and machine learning/Images_zinc/Train'
target_size = (300, 300)

classes = os.listdir(inp_dir)
all_images = []
all_labels = []

i = 0
for idx, c in enumerate(classes):
    img_list = os.listdir(inp_dir + '/' + '12170(80%)')
    print(idx)
    j = 0
    for img in img_list:
        fname = inp_dir + '/' + '12170(80%)' + '/' + img
        image = image_utils.load_img(fname).resize(target_size,Image.ANTIALIAS)
        image = np.array(image.getdata()).reshape(target_size[0], target_size[1], 3)
        image = image.astype('float32')/255
        all_images.append(image)
        all_labels.append(idx)
        #j += 1
        #if j >= 20:
        #    break
        #plt.imshow(image)
        #plt.show()
    #i += 1
    #if i >= 50:
    #    break


all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(all_images.shape)
print(all_labels.shape)

np.save('full_x', all_images)
np.save('full_y', all_labels)


'''
IGNORE THIS!! THIS IS THE CODE FOR THE MODEL. IT'S NOW AVAILABLE IN keras_VAE.py
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

# defined as separate layers because they will be reused later
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder network
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 150 * 150, activation='relu')

#if K.image_data_format() == 'channels_first':
#    output_shape = (batch_size, filters, 150, 150)
#else:
#    output_shape = (batch_size, 150, 150, filters)

output_shape = (batch_size, 150, 150, filters)

print('Output shape 1: ', output_shape)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')

#if K.image_data_format() == 'channels_first':
#    output_shape = (batch_size, filters, 300, 300)
#else:
#    output_shape = (batch_size, 300, 300, filters)

output_shape = (batch_size, 300, 300, filters)

print('Output shape 2: ', output_shape)

decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# Custom loss layer
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

# defining VAE model
vae = Model(x, y)

#def my_vae_loss(y_true, y_pred):
#    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
#    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#    vae_loss = K.mean(xent_loss + kl_loss)
#    return vae_loss

vae.compile(optimizer='rmsprop', loss=None, metrics=['accuracy'])
vae.summary()
#working code'''
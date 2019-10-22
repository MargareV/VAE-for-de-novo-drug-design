import tensorflow as tf
import functools
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import create_folder, get_model_description, delete_old_logs, get_files, create_results_folder
from time import time

#def remove_noise(image):
 #   alpha = tf.greater(image[:, :, 3], 50)
 #   alpha = tf.expand_dims(tf.cast(alpha, dtype=tf.uint8), 2)
 #   noise_filtered = tf.multiply(alpha, image)

 #  return noise_filtered[..., :3]


def parse_function(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    #image = remove_noise(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float16)
    image.set_shape([300, 300, 3])

    return image


def load_and_process_data(filenames, batch_size, shuffle=True):
    '''
    Revises a list of filenames and returns preprocessed images as a tensorflow dataset
    :param filenames: list of file paths
    :param batch_size: mini-batch size
    :param shuffle: Boolean
    :return:
    '''
    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(parse_function, num_parallel_calls=4)

        if shuffle:
            dataset = dataset.shuffle(5000) # Number of imgs to keep in a buffer to randomly sample

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)

    return dataset


def define_scope(function):
    # Decorator to lazy loading from https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class VAE:

    def __init__(self, data, latent_dim, learning_rate, image_size=300, channels=3):
        self.data = data
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        self.inputs_decoder = ((image_size / 5)**2) * channels
        self.encode
        self.decode
        self.optimize


    @define_scope
    def encode(self):
        activation = tf.nn.relu
        with tf.variable_scope('Data'):
            x = self.data
        with tf.variable_scope('Encoder'):
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.layers.flatten(x)

            # Local latent variables
            self.mean_ = tf.layers.dense(x, units=self.latent_dim, name='mean')
            self.std_dev = tf.nn.softplus(tf.layers.dense(x, units=self.latent_dim), name='std_dev')  # softplus to force >0

            # Reparametrization trick
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.latent_dim]), name='epsilon')
            self.z = self.mean_ + tf.multiply(tf.dtypes.cast(epsilon, tf.float16), self.std_dev)
            latent = tf.identity(self.z, name='latent_output')

            return self.z, self.mean_, self.std_dev

    @define_scope
    def decode(self):
        activation = tf.nn.relu
        with tf.variable_scope('Decoder'):
            x = tf.layers.dense(self.z, units=self.inputs_decoder, activation=activation)
            x = tf.layers.dense(x, units=self.inputs_decoder, activation=activation)
            recovered_size = int(np.sqrt(self.inputs_decoder/3))

            x = tf.reshape(x, [-1, recovered_size, recovered_size, 3])
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                           activation=activation)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=300 * 300 * 3, activation=None)

            x = tf.layers.dense(x, units=300 * 300 * 3, activation=tf.nn.sigmoid)
            output = tf.reshape(x, shape=[-1, 300, 300, 3])
            output = tf.identity(output, name='decoded_output')

        return output

    @define_scope
    def optimize(self):
        with tf.variable_scope('Optimize'):
            # Reshape input and output to flat vectors
            flat_output = tf.reshape(self.decode, [-1, 300 * 300 * 3])
            flat_input = tf.reshape(self.data, [-1, 300 * 300 * 3])

            with tf.name_scope('loss'):
                img_loss = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1 - flat_input) * -tf.log(1 - flat_output), 1)

                latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mean_) + tf.square(self.std_dev) - tf.log(tf.square(self.std_dev)) - 1, 1)

                loss = tf.reduce_mean(img_loss + latent_loss)
                tf.summary.scalar('batch_loss', loss)

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

# Program parameters
tf.flags.DEFINE_float('learning_rate', .0001, 'Initial learning rate.')
tf.flags.DEFINE_integer('epochs', 100, 'Number of steps to run trainer.')
tf.flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
tf.flags.DEFINE_integer('latent_dim', 2, 'Number of latent dimensions')
tf.flags.DEFINE_integer('test_image_number', 5, 'Number of test images to recover during training')
tf.flags.DEFINE_integer('epochs_to_plot', 2, 'Number of epochs before saving test sample of reconstructed images')
tf.flags.DEFINE_integer('save_after_n', 20, 'Number of epochs before saving network')
tf.flags.DEFINE_string('logdir', './logs', 'Logs folder')
tf.flags.DEFINE_string('data_path', '/home/margs/Drug dicovery and machine learning/VAE/Fifa-master/Data', 'Logs folder')
tf.flags.DEFINE_bool('shuffle', True, 'Shuffle dataset for training')
FLAGS = tf.flags.FLAGS


# Prepare output directories
model_description = get_model_description(FLAGS)
results_folder = create_results_folder(os.path.join('Results', model_description))
model_folder = create_folder(os.path.join('Models', model_description))
delete_old_logs(FLAGS.logdir)


# Create tf dataset
with tf.name_scope('DataPipe'):
    filenames = tf.compat.v1.placeholder_with_default(get_files(FLAGS.data_path), shape=[None], name='filenames_tensor')
    dataset = load_and_process_data(filenames, batch_size=FLAGS.batch_size, shuffle=FLAGS.shuffle)
    iterator = dataset.make_initializable_iterator()
    input_batch = iterator.get_next()

# Create model
vae = VAE(input_batch, FLAGS.latent_dim, FLAGS.learning_rate, )

init_vars = [tf.local_variables_initializer(), tf.global_variables_initializer()]

saver = tf.train.Saver()

# Training loop
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(init_vars)
    merged_summary_op = tf.summary.merge_all()
    write_graph = True

    for epoch in range(FLAGS.epochs):
        print('Actual epochs is: {}'.format(epoch), end='', flush=True)
        sess.run(iterator.initializer)
        flag = True
        ts = time()

        while True:
            try:
                sess.run(vae.optimize)

                # Get sample of images and their decoded couples
                if flag and not epoch % FLAGS.epochs_to_plot:
                    flag = False
                    summ, target, output_ = sess.run([merged_summary_op, input_batch, vae.decode])
                    writer.add_summary(summ, epoch)
                    f, axarr = plt.subplots(FLAGS.test_image_number, 2)
                    for j in range(FLAGS.test_image_number):
                        for pos, im in enumerate([target, output_]):
                            axarr[j, pos].imshow(im[j].reshape((300, 300, 3)))
                            axarr[j, pos].axis('off')

                    plt.savefig(os.path.join(results_folder, 'Train/Epoch_{}').format(epoch))
                    plt.close(f)

            except tf.errors.OutOfRangeError:
                print('\t Epoch time: {}'.format(time() - ts))

                # Save model
                if not epoch % FLAGS.save_after_n and epoch > 0:
                    print('Saving model...')
                    saver.save(sess, model_folder, global_step=epoch)
                break




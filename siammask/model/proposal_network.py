"""
MIT License

Copyright (c) 2020 Licht Takeuchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow as tf

from .adjuster import Adjuster
from .convolution import Conv2DWithBatchNorm


class ProposalNetwork(tf.keras.Model):
    def __init__(self, output_channels):
        super(ProposalNetwork, self).__init__()

        self.exampler_adjuster = Adjuster()
        self.search_adjuster = Adjuster()

        self.conv1 = Conv2DWithBatchNorm(256, 1)
        self.conv2 = tf.keras.layers.Conv2D(output_channels, 1)

    def call(self, inputs, training=None, mask=None):
        exampler_features, search_features = inputs

        kernel = self.exampler_adjuster(exampler_features)
        search_features = self.search_adjuster(search_features)

        batch_size = tf.shape(kernel)[0]
        kernel_h = 5  # == tf.shape(kernel)[1]
        kernel_w = 5  # == tf.shape(kernel)[2]
        channel_size = 256  # == tf.shape(kernel)[3]

        image_patches = tf.image.extract_patches(
            search_features,
            [1, kernel_h, kernel_w, 1],
            [1, 1, 1, 1], [1, 1, 1, 1],
            padding='VALID'
        )

        correlation_h = tf.shape(image_patches)[1]
        correlation_w = tf.shape(image_patches)[2]

        kernel = tf.reshape(kernel, (batch_size, 1, 1, kernel_h * kernel_w * channel_size))

        correlation = tf.reduce_sum(
            tf.reshape(
                tf.multiply(image_patches, kernel),
                (batch_size, correlation_h, correlation_w, -1, channel_size)
            ),
            axis=-2
        )

        output = tf.keras.activations.relu(self.conv1(correlation))

        return [self.conv2(output), correlation]

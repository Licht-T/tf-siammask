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


class MaskRefinementBlock(tf.keras.Model):
    def __init__(self, horizontal_filters, post_filters, upsample_size):
        super(MaskRefinementBlock, self).__init__()

        self.upsample_size = upsample_size

        self.conv_h1 = tf.keras.layers.Conv2D(horizontal_filters, 3, padding='SAME')
        self.conv_h2 = tf.keras.layers.Conv2D(horizontal_filters, 3, padding='SAME')

        self.conv_v1 = tf.keras.layers.Conv2D(4 * horizontal_filters, 3, padding='SAME')
        self.conv_v2 = tf.keras.layers.Conv2D(horizontal_filters, 3, padding='SAME')

        self.conv_post = tf.keras.layers.Conv2D(post_filters, 3, padding='SAME')

        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=None, mask=None):
        h, v = inputs

        h = tf.keras.activations.relu(self.conv_h1(h))
        h = tf.keras.activations.relu(self.conv_h2(h))

        v = tf.keras.activations.relu(self.conv_v1(v))
        v = tf.keras.activations.relu(self.conv_v2(v))

        return self.conv_post(tf.image.resize(self.add([h, v]), [self.upsample_size, self.upsample_size], tf.image.ResizeMethod.BICUBIC))


class MaskRefinementNet(tf.keras.Model):
    def __init__(self):
        super(MaskRefinementNet, self).__init__()

        self.deconvolution = tf.keras.layers.Conv2DTranspose(32, 15, padding='VALID')

        self.block1 = MaskRefinementBlock(32, 16, 31)
        self.block2 = MaskRefinementBlock(16, 4, 61)
        self.block3 = MaskRefinementBlock(4, 1, 127)

    def call(self, inputs, training=None, mask=None):
        h, v1, v2, v3 = inputs

        h = self.deconvolution(h)
        h = self.block1([h, v1])
        h = self.block2([h, v2])
        h = self.block3([h, v3])

        return tf.sigmoid(h)

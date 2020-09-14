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

from .convolution import Conv2DWithBatchNorm


class BottleneckA(tf.keras.Model):
    def __init__(
            self, filters: int, downsample_kernel_size: int,
            strides: int, padding: str, downsample_padding: str
    ):
        super(BottleneckA, self).__init__()

        self.conv1 = Conv2DWithBatchNorm(filters, 1)
        self.conv2 = Conv2DWithBatchNorm(filters, 3, strides, padding, 1)
        self.conv3 = Conv2DWithBatchNorm(4 * filters, 1)
        self.conv4 = Conv2DWithBatchNorm(4 * filters, downsample_kernel_size, strides, downsample_padding, 1)

    def call(self, inputs, training=None, mask=None):
        x = tf.keras.activations.relu(self.conv1(inputs))
        x = tf.keras.activations.relu(self.conv2(x))
        x = tf.keras.layers.add([self.conv3(x), self.conv4(inputs)])

        return tf.keras.activations.relu(x)


class BottleneckB(tf.keras.Model):
    def __init__(
            self, filters: int, dilation_rate: int
    ):
        super(BottleneckB, self).__init__()

        self.conv1 = Conv2DWithBatchNorm(filters, 1)
        self.conv2 = Conv2DWithBatchNorm(filters, 3, 1, 'SAME', dilation_rate)
        self.conv3 = Conv2DWithBatchNorm(4 * filters, 1)

    def call(self, inputs, training=None, mask=None):
        x = tf.keras.activations.relu(self.conv1(inputs))
        x = tf.keras.activations.relu(self.conv2(x))
        x = tf.keras.layers.add([self.conv3(x), inputs])

        return tf.keras.activations.relu(x)


class ResBlocks(tf.keras.Model):
    def __init__(
            self, num_blocks: int, filters: int, downsample_kernel_size: int,
            strides_a: int, padding_a: str, downsample_padding: str,
            dilation_rate: int = 1
    ):
        super(ResBlocks, self).__init__()

        self.block0 = BottleneckA(filters, downsample_kernel_size, strides_a, padding_a, downsample_padding)

        self.blocks = tf.keras.Sequential()

        for i in range(1, num_blocks):
            self.blocks.add(BottleneckB(filters, dilation_rate))

    def call(self, inputs, training=None, mask=None):
        return self.blocks(self.block0(inputs))


class CustomResNet50(tf.keras.Model):
    def __init__(self):
        super(CustomResNet50, self).__init__()

        self.conv = Conv2DWithBatchNorm(64, 7, 2)

        self.max_pooling = tf.keras.layers.MaxPool2D(3, 2, 'SAME')

        self.res2 = ResBlocks(3, 64, 1, 1, 'SAME', 'VALID')
        self.res3 = ResBlocks(4, 128, 3, 2, 'VALID', 'VALID')
        self.res4 = ResBlocks(6, 256, 3, 1, 'SAME', 'SAME', 2)

        self.downsample = Conv2DWithBatchNorm(256, 1)

    def call(self, inputs, training=None, mask=None):
        res1 = tf.keras.activations.relu(self.conv(inputs))

        x = self.max_pooling(res1)
        res2 = self.res2(x)
        res3 = self.res3(res2)
        res4 = self.res4(res3)

        return [self.downsample(res4), res3, res2, res1]

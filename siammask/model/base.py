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

from .resnet import CustomResNet50
from .proposal_network import ProposalNetwork


class BaseNet(tf.keras.Model):
    def __init__(self, num_anchors: int, mask_pixels: int, kernel_cut_off_size: int):
        super(BaseNet, self).__init__()

        self.num_anchors = num_anchors
        self.cut_off = kernel_cut_off_size // 2

        self.resnet = CustomResNet50()

        self.score_proposal = ProposalNetwork(2 * self.num_anchors)
        self.box_proposal = ProposalNetwork(4 * self.num_anchors)
        self.mask_proposal = ProposalNetwork(mask_pixels * mask_pixels)

    def call(self, inputs, training=None, mask=None):
        exampler, search = inputs

        exampler_features, _, _, _ = self.resnet(exampler)
        exampler_features = exampler_features[:, self.cut_off:-self.cut_off, self.cut_off:-self.cut_off, :]
        search_features, res3, res2, res1 = self.resnet(search)

        scores, _ = self.score_proposal([exampler_features, search_features])
        boxes, _ = self.box_proposal([exampler_features, search_features])
        masks, mask_features = self.mask_proposal([exampler_features, search_features])

        batch_size = tf.shape(scores)[0]
        h = tf.shape(scores)[1]
        w = tf.shape(scores)[2]
        scores = tf.reshape(scores, (batch_size, h, w, self.num_anchors, 2))

        scores = tf.keras.activations.softmax(scores)[..., 1]

        batch_size = tf.shape(boxes)[0]
        h = tf.shape(boxes)[1]
        w = tf.shape(boxes)[2]
        boxes = tf.reshape(boxes, (batch_size, h, w, self.num_anchors, 4))

        xy = boxes[..., :2]
        wh = tf.keras.backend.exp(boxes[..., 2:])

        return [scores, tf.concat([xy, wh], -1), tf.sigmoid(masks), mask_features, [res3, res2, res1]]

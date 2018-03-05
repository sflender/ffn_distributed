# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classes for FFN model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow.python.util import deprecation
from . import optimizer


class FFNModel(object):
  """Base class for FFN models."""

  # Dimensionality of the model (2 or 3).
  dim = None

  ############################################################################
  # (x, y, z) tuples defining various properties of the network.
  # Note that 3-tuples should be used even for 2D networks, in which case
  # the third (z) value is ignored.

  # How far to move the field of view in the respective directions.
  deltas = None

  # Size of the input image and seed subvolumes to be used during inference.
  # This is enough information to execute a single prediction step, without
  # moving the field of view.
  input_image_size = None
  input_seed_size = None

  # Size of the predicted patch as returned by the model.
  pred_mask_size = None
  ###########################################################################

  # TF op to compute loss optimized during training. This should include all
  # loss components in case more than just the pixelwise loss is used.
  loss = None

  # TF op to call to perform loss optimization on the model.
  train_op = None

  def __init__(self, deltas, batch_size=None):
    assert self.dim is not None

    self.deltas = deltas
    self.batch_size = batch_size

    # Initialize the shift collection. This is used during training with the
    # fixed step size policy.
    self.shifts = []
    for dx in (-self.deltas[0], 0, self.deltas[0]):
      for dy in (-self.deltas[1], 0, self.deltas[1]):
        for dz in (-self.deltas[2], 0, self.deltas[2]):
          if dx == 0 and dy == 0 and dz == 0:
            continue
          self.shifts.append((dx, dy, dz))

    # Mask identifying valid examples within the batch. Only valid examples
    # contribute to the loss and see object mask updates.
    self.offset_label = tf.placeholder(tf.string, name='offset_label')
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # The seed is always a placeholder which is fed externally from the
    # training/inference drivers.
    self.input_seed = tf.placeholder(tf.float32, name='seed')
    self.input_patches = tf.placeholder(tf.float32, name='patches')

    # For training, labels should be defined as a TF object.
    self.labels = None

    # Optional. Provides per-pixel weights with which the loss is multiplied.
    # If specified, should have the same shape as self.labels.
    self.loss_weights = None

    # List of image tensors to save in summaries. The images are concatenated
    # along the X axis.
    self._images = []

    # set up placeholder for adjustable learning rate
    self.learning_rate = tf.placeholder(tf.float32,name="learning_rate")

  def set_uniform_io_size(self, patch_size):
    """Initializes unset input/output sizes to 'patch_size', sets input shapes.

    This assumes that the inputs and outputs are of equal size, and that exactly
    one step is executed in every direction during training.

    Args:
      patch_size: (x, y, z) specifying the input/output patch size

    Returns:
      None
    """
    if self.pred_mask_size is None:
      self.pred_mask_size = patch_size
    if self.input_seed_size is None:
      self.input_seed_size = patch_size
    if self.input_image_size is None:
      self.input_image_size = patch_size
    self.set_input_shapes()

  def set_input_shapes(self):
    """Sets the shape inference for input_seed and input_patches.

    Assumes input_seed_size and input_image_size are already set.
    """
    self.input_seed.set_shape([self.batch_size] +
                              list(self.input_seed_size[::-1]) + [1])
    self.input_patches.set_shape([self.batch_size] +
                                 list(self.input_image_size[::-1]) + [1])

  def set_up_sigmoid_pixelwise_loss(self, logits):
    """Sets up the loss function of the model."""
    assert self.labels is not None
    assert self.loss_weights is not None

    pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                         labels=self.labels)
    pixel_loss *= self.loss_weights
    self.loss = tf.reduce_mean(pixel_loss)
    tf.summary.scalar('pixel_loss', self.loss)
    self.loss = tf.verify_tensor_all_finite(self.loss, 'Invalid loss detected')

  def set_up_optimizer(self, loss=None, max_gradient_entry_mag=0.7):
    """Sets up the training op for the model."""
    if loss is None:
      loss = self.loss
    tf.summary.scalar('optimizer_loss', self.loss)

    #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    opt = optimizer.optimizer_from_flags(self.learning_rate)
    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)
    grads_and_vars = opt.compute_gradients(loss)

    for g, v in grads_and_vars:
      if g is None:
        tf.logging.error('Gradient is None: %s', v.op.name)

    if max_gradient_entry_mag > 0.0:
      grads_and_vars = [(tf.clip_by_value(g,
                                          -max_gradient_entry_mag,
                                          +max_gradient_entry_mag), v)
                        for g, v, in grads_and_vars]

    # TODO(b/34707785): Hopefully remove need for these deprecated calls.  Let
    # one warning through so that we have some (low) possibility of noticing if
    # the message changes.
    trainables = tf.trainable_variables()
    if trainables:
      var = trainables[0]
      tf.contrib.deprecated.histogram_summary(var.op.name, var)
    with deprecation.silence():
      for var in trainables[1:]:
        tf.contrib.deprecated.histogram_summary(var.op.name, var)
      for grad, var in grads_and_vars:
        tf.contrib.deprecated.histogram_summary(
            'gradients/' + var.op.name, grad)

    self.train_op = opt.apply_gradients(grads_and_vars,
                                        global_step=self.global_step,
                                        name='train')

  def show_center_slice(self, image, sigmoid=True):
    image = image[:, image.get_shape().dims[1] // 2, :, :, :]
    if sigmoid:
      image = tf.sigmoid(image)
    self._images.append(image)

  def add_summaries(self, max_images=4):
    tf.contrib.deprecated.image_summary(
        'state/' + self.offset_label, tf.concat(self._images, 2),
        max_images=max_images)

  def update_seed(self, seed, update):
    """Updates the initial 'seed' with 'update'."""
    dx = self.input_seed_size[0] - self.pred_mask_size[0]
    dy = self.input_seed_size[1] - self.pred_mask_size[1]
    dz = self.input_seed_size[2] - self.pred_mask_size[2]

    if dx == 0 and dy == 0 and dz == 0:
      seed += update
    else:
      seed += tf.pad(update, [[0, 0],
                              [dz // 2, dz - dz // 2],
                              [dy // 2, dy - dy // 2],
                              [dx // 2, dx - dx // 2],
                              [0, 0]])
    return seed

  def define_tf_graph(self):
    """Creates the TensorFlow graph representing the model.

    If self.labels is not None, the graph should include operations for
    computing and optimizing the loss.
    """
    raise NotImplementedError(
        'DefineTFGraph needs to be defined by a subclass.')

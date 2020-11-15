import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualBlock(layers.Layer):
  """
    A Residual block.
  """

  def __init__(self, filters, kernel_size,
               dropout_percentage, strides=1, **kwargs):
    """
      Arguments:
        filters: input tensor.
        kernel_size: integer, the number of building blocks.
        name: string, block label.
        dropout_percentage:
        strides:
      Returns:
        a `ResidualBlock` instance.
    """
    super(ResidualBlock, self).__init__(**kwargs)
    self.conv_1 = layers.Conv2D(filters, (1, 1), strides=strides)
    self.bn_1 = layers.BatchNormalization()
    self.rel_1 = layers.ReLU()
    self.conv_2 = layers.Conv2D(filters, kernel_size, padding="same",
                                strides=strides)
    self.dropout_percentage = dropout_percentage
    self.dropout_layer = layers.Dropout(dropout_percentage)
    self.bn_2 = layers.BatchNormalization()
    self.rel_2 = layers.ReLU()
    self.conv_3 = layers.Conv2D(filters, kernel_size, padding="same")
    self.add = layers.Add()
    self.strides = strides


  def call(self, inputs, training=None):
    x = inputs
    if self.strides > 1:
      x = self.conv_1(x)
    res_x = self.bn_1(inputs)
    res_x = self.rel_1(res_x)
    res_x = self.conv_2(res_x)
    if self.dropout_percentage:
      res_x = self.dropout_layer(res_x, training=training)
    res_x = self.bn_2(res_x)
    res_x = self.rel_2(res_x)
    res_x = self.conv_3(res_x)
    inputs = self.add([x, res_x])
    return inputs

  def get_config(self):
    base_config = super().get_config()
    return {**base_config,
            "conv_1": self.conv_1,
            "bn_1": self.bn_1,
            "rel_1": self.rel_1,
            "conv_2": self.conv_2,
            "dropout_1": self.dropout,
            "bn_2": self.bn_2,
            "rel_2": self.rel_2,
            "conv_3": self.conv_3,
            "add": self.add,
            "dropout_percentage": self.dropout_percentage,
            "strides": self.strides
            }

class WideResidualNetwork(models.Model):

  """Instantiates the DenseNet architecture.
  Reference:
  - [Wide Residual Networks](
      https://arxiv.org/abs/1605.07146
  """

  def __init__(self, n_classes, d, k, kernel_size=(3, 3), activation='softmax',
               dropout_percentage=None, includeActivation=True, **kwargs):
    """
  Arguments:
    n_classes: number of output classes.
    d: the depth of the network
    k: the width of the network
    kernel_size: kernel size to pass to the conv layers, default value it's ok in the majority of cases
    dropout_percentage: percentage of dropout if you want to use it
    strides:
    includeActivation: whether to include the activation layer if you want to work with logits
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
  Returns:
    A `WideResidualNetwork` instance.
  Raises:
    ValueError: in case of invalid argument for invalid couple (k, d)

    """
    super(WideResidualNetwork, self).__init__(**kwargs)
    if (d-4)%6 != 0:
      raise ValueError('Please choose a correct depth!')

    self.dropout_percentage = dropout_percentage
    self.N = int((d - 4) / 6)
    self.k = k
    self.d = d
    self.includeActivation = includeActivation
    self.kernel_size = kernel_size

    self.bn_1 = layers.BatchNormalization()
    self.rel_1 = layers.ReLU()
    self.conv_1 = layers.Conv2D(16, (3, 3), padding='same')
    self.conv_2 = layers.Conv2D(16*k, (1, 1))
    if n_classes>2:
        self.dense = layers.Dense(n_classes)
    else:
        self.dense = layers.Dense(1)

    self.res_block_1 = [ResidualBlock(16*self.k, self.kernel_size,  self.dropout_percentage) for _ in range(self.N)]
    self.res_single_1 = ResidualBlock(32*self.k, self.kernel_size,  self.dropout_percentage, strides=2)
    self.res_block_2 = [ResidualBlock(32*self.k, self.kernel_size,  self.dropout_percentage) for _ in range(self.N-1)]
    self.res_single_2 = ResidualBlock(64*self.k, self.kernel_size,  self.dropout_percentage, strides=2)
    self.res_block_3 = [ResidualBlock(64*self.k, self.kernel_size,  self.dropout_percentage) for _ in range(self.N-1)]
    self.pooling = layers.GlobalAveragePooling2D()
    self.activation_layer = layers.Activation(activation)



  def call(self, inputs, training=None):
    x = self.bn_1(inputs)
    x = self.rel_1(x)
    x = self.conv_1(x)
    x = self.conv_2(x)
    for layer in self.res_block_1:
      x = layer(x, training=training)

    x = self.res_single_1(x, training=training)

    for layer in self.res_block_2:
      x  = layer(x, training=training)

    x = self.res_single_2(x, training=training)

    for layer in self.res_block_3:
      x = x = layer(x, training=training)

    x = self.pooling(x)
    x = self.dense(x)
    if self.includeActivation:
        x = self.activation_layer(x)

    return x

  def get_config(self):
    base_config = super().get_config()
    return {**base_config,
            "bn_1": self.bn_1,
            "rel_1": self.rel_1,
            "conv_1": self.conv_1,
            "conv_2": self.conv_2,
            "dense": self.dense,
            "res_block_1": self.res_block_1,
            "res_single_1": self.res_single_1,
            "res_block_2": self.res_block_2,
            "res_single_2": self.res_single_2,
            "res_block_3": self.res_block_3,
            "pooling": self.pooling,
            "activation_layer": self.activation_layer,
            "dropout_percentage": self.dropout_percentage,
            "N": self.N,
            "k": self.k,
            "d": self.d,
            "includeTop": self.includeTop,
            "kernel_size": self.kernel_size
            }

import tensorflow as tf
from keras.layers import (
    Layer,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    BatchNormalization,
    Multiply,
)
from keras.activations import relu, sigmoid, get as get_activation
from keras import Model


class ConvBlock(Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", activation="relu", **kwargs
    ):
        """Implements the Conv-Block described in Fig 1. of the paper

        Args:
            filters: Number of filters/channels of the convolution.
            kernel_size: Size of the convolution's kernel. Defaults to (3,3).
            padding: Padding. Defaults to 'same'.
            activation: Activation after Batch Norm. Defaults to relu.
            **kwargs: Other arguments that are passed to the Layer.
        """
        super(ConvBlock, self).__init__(**kwargs)
        # --- store configuration for serialization
        self.filters = (filters,)
        self.kernel_size = (kernel_size,)
        self.padding = (padding,)
        self.activation = (activation,)
        self.kwargs = kwargs
        # --- create needed layers
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)
        self.batch_norm = BatchNormalization()
        self.activation_fn = get_activation(activation)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        return self.activation_fn(x)

    def get_config(self):
        return dict(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation,
            **self.kwargs,
        )


class AttentionBlock(Layer):
    """Implements the Attention-Block described in Fig 1. of the paper

    Args:
        units: number of units of the dense layer. Must be the same
         as the input filters/channels.
        **kwargs: Any other arguments that are passed to the Layer.
    """

    def __init__(self, units, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.kwargs = kwargs
        self.global_avg_pooling = GlobalAveragePooling2D(keepdims=True)
        self.dense = Dense(units=units)

    def call(self, inputs):
        x = self.global_avg_pooling(inputs)
        x = self.dense(x)
        x = relu(x)
        x = sigmoid(x)
        return Multiply()([x, inputs])

    def get_config(self):
        return dict(units=self.units, **self.kwargs)


class Midblock(Layer):
    def __init__(self, filters, steps, reducing_factor=2, **kwargs):
        """Implements the bottleneck described in Fig. 1 in the paper.

        :param filters: Initial and final number of filters/channels.
        :param steps: Number of AttentionBlocks. The output is reduced in channels after every step.
        :param reducing_factor: For each step the output of the AB is `filters / (reducing_factor**step)`, defaults to 2
        """
        super(Midblock, self).__init__(**kwargs)
        # --- store configuration for serialization
        self.filters = filters
        self.steps = steps
        self.reducing_factor = (reducing_factor,)
        self.kwargs = kwargs

        # --- create all necessary layers
        self.attn_blocks = []
        self.conv_blocks = []
        get_channels = lambda step: int(
            filters / (reducing_factor**step)
        )  # divided by 2**0, 2**1, 2**2 etc
        for step in range(self.steps):
            attn_units = get_channels(step)
            self.conv_blocks.append(ConvBlock(attn_units, name=f"conv_{step}"))
            self.attn_blocks.append(AttentionBlock(attn_units, name=f"attn_{step}"))
        self.conv_final = ConvBlock(filters=filters, name="conv_final")

    def get_config(self):
        return dict(
            filters=self.filters,
            steps=self.steps,
            reducing_factor=self.reducing_factor,
            **self.kwargs,
        )

    def call(self, inputs):
        x = inputs
        for step in range(self.steps):
            x = self.conv_blocks[step](x)
            if step == 0:  # store for the skip connection
                skip = x
            x = self.attn_blocks[step](x)
        x = self.conv_final(x)
        return tf.add(inputs, skip)


class LWBNAUnet(Model):
    def __init__(
        self, n_classes, filters, depth, midblock_steps, dropout_rate=0.3, **kwargs
    ):
        """Implements the Lightweight Bottle Neck Attention Unet architecture
        described in `A lightweight deep learning model for automatic segmentation
        and analysis of ophthalmic images` (https://doi.org/10.1038/s41598-022-12486-w)
        Input has shape `(Batch size, Height, Width, Channels)`
        Input has dtype float and is expected to be normalized to the range [0,1]
        Output has shape `(Batch size, Height, Width, n_classes)`

        Args:
            n_classes (int): number of output channels.
            filters (int): number of filters of the conv-blocks.
            depth (int): number of contraction/expansion levels.
            midblock_steps (int): number of attn-blocks in the bottle-neck.
            dropout_rate (float, optional): Dropout rate for training. Defaults to 0.3.
        """
        super(LWBNAUnet, self).__init__(**kwargs)
        # --- input shape is (height, width, channels)
        self.n_classes = n_classes
        self.filters = filters
        self.depth = depth
        self.midblock_steps = midblock_steps
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs

        self.conv_blocks = {}
        self.attn_blocks = {}
        for d in range(depth):
            self.conv_blocks[f"contract_{d}0"] = ConvBlock(
                filters=filters, name=f"conv_contract_{d}0"
            )
            self.conv_blocks[f"contract_{d}1"] = ConvBlock(
                filters=filters, name=f"conv_contract_{d}1"
            )
            self.attn_blocks[f"contract_{d}"] = AttentionBlock(
                filters, name=f"attn_contract_{d}"
            )
            self.conv_blocks[f"expand_{d}0"] = ConvBlock(
                filters=filters, name=f"conv_expand_{d}0"
            )
            self.conv_blocks[f"expand_{d}1"] = ConvBlock(
                filters=filters, name=f"conv_expand_{d}1"
            )
            self.attn_blocks[f"expand_{d}"] = AttentionBlock(
                filters, name=f"attn_expand_{d}"
            )

        self.conv_blocks["post_midblock_0"] = ConvBlock(
            filters=filters, name="conv_post_midblock_0"
        )
        self.conv_blocks["post_midblock_1"] = ConvBlock(
            filters=filters, name="conv_post_midblock_1"
        )
        self.attn_blocks["post_midblock"] = AttentionBlock(
            filters, name="attn_post_midblock"
        )
        self.midblock = Midblock(filters=filters, steps=midblock_steps)
        self.output_conv = Conv2D(
            filters=n_classes, kernel_size=1, activation="sigmoid"
        )

    def call(self, inputs, training=None):
        def dropout_conv_conv_attn(x, dir, depth):
            x = Dropout(self.dropout_rate)(x, training=training)
            x = self.conv_blocks[f"{dir}_{depth}0"](x)
            x = self.conv_blocks[f"{dir}_{depth}1"](x)
            return self.attn_blocks[f"{dir}_{depth}"](x)

        # store skip connections
        skip = {}

        # first contraction
        x = self.conv_blocks["contract_00"](inputs)
        x = self.conv_blocks["contract_01"](x)
        skip[0] = self.attn_blocks["contract_0"](x)
        # the first contraction comes directly from the second Conv-Block
        x = MaxPooling2D(name="max_pool_contract_0")(x)

        # intermediate contractions
        for d in range(1, self.depth):
            x = dropout_conv_conv_attn(x, "contract", d)
            skip[d] = x  # store for later
            x = MaxPooling2D(name="max_pool_contract_0")(x)

        # bottle-neck
        x = Dropout(self.dropout_rate)(x, training=training)
        x = self.midblock(x)
        x = self.conv_blocks["post_midblock_0"](x)
        x = self.conv_blocks["post_midblock_1"](x)
        x = self.attn_blocks["post_midblock"](x)
        x = UpSampling2D()(x)

        # intermediate expansions
        for d in reversed(range(1, self.depth)):
            x = tf.add(x, skip[d])  # add skip connection
            x = dropout_conv_conv_attn(x, "expand", d)
            x = UpSampling2D()(x)

        # final expansion
        x = tf.add(x, skip[0])  # add skip connection
        x = dropout_conv_conv_attn(x, "expand", 0)
        return self.output_conv(x)

    def get_config(self):
        return dict(
            n_classes=self.n_classes,
            filters=self.filters,
            depth=self.depth,
            midblock_steps=self.midblock_steps,
            dropout_rate=self.dropout_rate,
            **self.kwargs,
        )


if __name__ == "__main__":
    import numpy as np

    unet = LWBNAUnet(
        n_classes=1,
        filters=128,
        depth=4,
        midblock_steps=4,
        dropout_rate=0.3,
        name="my_unet",
    )
    unet.build(input_shape=(8, 320, 320, 3))
    unet.predict(np.random.rand(8, 256, 256, 3))
    unet.summary()

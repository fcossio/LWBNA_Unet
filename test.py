import lwbna_unet as unet
import numpy as np

# input has shape `(Batch size, Height, Width, Channels)`
# input has dtype float and is expected to be normalized to the range [0,1].
# output has shape `(Batch size, Height, Width, n_classes)`
my_unet = unet.LWBNAUnet(
    n_classes=1, 
    filters=128, 
    depth=4, 
    midblock_steps=4, 
    dropout_rate=0.3, 
    name="my_unet"
)

# the network is untrained. Dummy input.
my_unet.build(input_shape=(8,320,320,3))
my_unet.predict(np.random.rand(8,256,256,3))
my_unet.summary()
# Light Weight Bottle Neck Attention Unet
TF implementation of the architecture described in [A lightweight deep learning model for automatic segmentation and analysis of ophthalmic images](https://doi.org/10.1038/s41598-022-12486-w) by Sharma et al.

This is an independent implementation unrelated to the autors of the paper. I have used it for segmenting fibers in my own [project](https://wandb.ai/warm-kanelbullar/diameterY/reports/Measuring-the-diameter-of-nanofibers--VmlldzoyMjY2NTg4?accessToken=wvuavha9la5hd0vtt3h6h41fqgb9cdvywrmwwox7os2stkdrbh5vf23dzqq38cf5). Please leave a 
<a class="github-button" href="https://github.com/fcossio/LWBNA_Unet" data-icon="octicon-star" data-size="large" aria-label="Star fcossio/LWBNA_Unet on GitHub">Star</a> if this code is useful to you :smile:.

## Usage
```bash
# install your favorite version of tensorflow2
pip install tensorflow
# install this package
pip install lwbna-unet
```

```python
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
# you can now train `my_unet` as a regular `keras.Model`

```


<script async defer src="https://buttons.github.io/buttons.js"></script>

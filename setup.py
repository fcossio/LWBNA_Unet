# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
import os

def read(rel_path):
    here = pathlib.Path(__file__).parent.resolve()
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

setup(
    name='lwbna-unet',
    version='1.0.0',
    description='TF2 (Keras) implementation of LWBNA_Unet. Unrelated to the authors of the paper',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/fcossio/LWBNA_Unet',
    author='Fernando Cossio',
    author_email='fer_cossio@hotmail.com',
    keywords='tensorflow2, tf2, keras, image segmentation, computer vision, unet',
    packages=find_packages(where='tf2_lwbna_unet'),
    python_requires='>=3.6',

    install_requires=[
        'keras'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/fcossio/LWBNA_Unet/issues',
        'Source': 'https://github.com/pypa/sampleproject/',
    },
)
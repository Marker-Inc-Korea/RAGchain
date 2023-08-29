from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    name='KoPrivateGPT',
    version='0.1',
    packages=find_packages(where='KoPrivateGPT'),
    package_dir={'': 'KoPrivateGPT'},
    py_modules=[splitext(basename(path))[0] for path in glob('KoPrivateGPT/*.py')],
)

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension('deadwood_counter_revised',
              ["deadwood_counter_revised.pyx", 'deadwood_counter_revised.pxd'],
              include_dirs=[numpy.get_include()])
]

setup(
    ext_modules=cythonize(extensions)
)

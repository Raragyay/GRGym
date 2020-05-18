import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(["deadwood_counter_revised.pyx", 'deadwood_counter_revised.pxd']),
    include_dirs=[numpy.get_include()]
)

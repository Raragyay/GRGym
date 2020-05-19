import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension('*',
              ["*.pyx"],
              include_dirs=[numpy.get_include(), '.']),
    # Extension('meld',
    #           ["meld.pyx"],
    #           include_dirs=[numpy.get_include(), '.']),
    # Extension('set',
    #           ["set.pyx"],
    #           include_dirs=[numpy.get_include(), '.']),
    # Extension('run',
    #           ["run.pyx"],
    #           include_dirs=[numpy.get_include(), '.']),
]

setup(
    name='cython_environment',
    ext_modules=cythonize(extensions,
                          compiler_directives={
                              'language_level': '3'},
                          annotate=True),
    zip_safe=False
)

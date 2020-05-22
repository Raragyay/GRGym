import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup


def scan_dir(dir=None, files=[]):
    for file in (os.listdir(dir) if dir else os.listdir()):
        path = os.path.join(dir if dir else "", file)
        if os.path.isfile(path) and path.endswith(".pyx") and not file.startswith('__'):
            files.append(path.replace(os.path.sep, '.')[:-4])
        elif os.path.isdir(path):
            scan_dir(path, files)
    return files


def build_extension(ext_name):
    ext_path = ext_name.replace('.', os.path.sep) + '.pyx'
    return Extension(
        ext_name,
        [ext_path],
        include_dirs=[numpy.get_include(), '.'],
    )


os.chdir('tests')
ext_names = scan_dir()
extensions = [build_extension(name) for name in ext_names]

setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={
                              'language_level': '3',
                          }),
)

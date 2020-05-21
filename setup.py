import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, find_packages, setup


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


# os.path.join(os.getcwd(), path_to)
os.chdir('src')
ext_names = scan_dir()
print(ext_names)
extensions = [build_extension(name) for name in ext_names]
print(extensions)
setup(
    name="GRGym",
    packages=find_namespace_packages(),
    ext_modules=cythonize(extensions,
                          compiler_directives={
                              'language_level': '3',
                              'embedsignature': 'True'
                          },
                          annotate=True),
    zip_safe=False,
    package_data={
        'GRGym.environment': ['*.pxd']},
)

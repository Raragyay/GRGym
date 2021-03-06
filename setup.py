import os
import sys
import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


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


ext_names = scan_dir('GRGym')
extensions = [build_extension(name) for name in ext_names]

compiler_directives = {
    'language_level': '3',
    'embedsignature': True,
}
# Only add line tracing directive if distutils tracing macro is specified from CLI.
if 'CYTHON_TRACE_NOGIL' in sys.argv:
    print('true')
    compiler_directives['linetrace'] = True

setup(
    name="GRGym",
    packages=find_packages(),
    ext_modules=cythonize(extensions,
                          compiler_directives=compiler_directives,
                          annotate=True),
    zip_safe=False,
    package_data={
        'GRGym.environment': ['*.pxd']},
)

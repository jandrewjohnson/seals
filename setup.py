# setup.py — only needed to declare the Cython extension, which pyproject.toml
# cannot express (it needs numpy.get_include() resolved at build time).
from setuptools import setup
import numpy
from Cython.Build import cythonize
from setuptools.extension import Extension

ext_modules = [Extension('seals.seals_cython_functions',
                         ['seals/seals_cython_functions.pyx'],
                         include_dirs=[numpy.get_include()])]

# cythonize() rather than cmdclass={'build_ext': Cython.Distutils.build_ext}:
# that shim picks its base class from whatever happens to be in sys.modules at
# import time and inherits from distutils' build_ext, not setuptools'. On
# Python 3.12+ (no stdlib distutils) that resolution is import-order dependent,
# so the extension can silently go unbuilt and ship a pure-Python wheel.
setup(
    ext_modules=cythonize(ext_modules),
)
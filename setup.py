"""
To build, active the right conda environment and then run command:
python setup.py build_ext --inplace.
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='word2vec_inner_modified',
    ext_modules=cythonize("word2vec_inner_modified.pyx"),
    include_dirs=[numpy.get_include()]
)

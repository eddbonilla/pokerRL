from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("leduc_c.pyx"),
    include_dirs=[numpy.get_include()],
	package_data = {
        'leduc_c': ['*.pxd']}
)
setup(
	ext_modules = cythonize("MCTS_c.pyx"),
	include_dirs=[numpy.get_include()],
	package_data = {
        'leduc_c': ['*.pxd']}

)
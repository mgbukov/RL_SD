from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy,sys


file_name="environment"

if sys.platform == 'win32':
    setup(
        ext_modules = cythonize(
            Extension(
                file_name,
                sources=[file_name+".pyx"],
                extra_compile_args=["/std:c++latest"],
                extra_link_args=["/std:c++latest"],
                language="c++"
                      )
                                ),
        include_dirs=[numpy.get_include(),"./"]
        )
else:
    setup(
        ext_modules = cythonize(
            Extension(
                file_name,
                sources=[file_name+".pyx"],
                extra_compile_args=["-std=c++11"],
                extra_link_args=["-std=c++11"],
                language="c++"
                      )
                                ),
        include_dirs=[numpy.get_include(),"./"]
        )



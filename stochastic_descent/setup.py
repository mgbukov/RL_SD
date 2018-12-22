from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "qsp_opt.cython.cythonUtils",
        ["qsp_opt/cython/cythonUtils.pyx"]
    ),
]

setup(name='qsp_opti',
      version='1.0',
      description='Optimization of quantum state preparation',
      url='https://github.com/alexandreday/https://github.com/alexandreday/Optimize_QSP',
      author='Alexandre Day',
      author_email='alexandre.day1@gmail.com',
      license='MIT',
      packages=['qsp_opt'],
      zip_safe=False,
      include_package_data=True,
      ext_modules=cythonize(extensions)
)

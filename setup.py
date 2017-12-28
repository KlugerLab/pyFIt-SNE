from setuptools import setup, find_packages  # from distutils.core import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [Extension("fitsne.cppwrap",
                        ["fitsne/cppwrap.pyx", "fitsne/src/nbodyfft.cpp", "fitsne/src/sptree.cpp", "fitsne/src/tsne.cpp"],
                        language="c++",
                        # include_dirs=["fitsne", "fitsne/src"],
                        extra_compile_args=["-std=c++11", "-O3", '-pthread', "-lfftw3", "-lm"],
                        extra_link_args=['-lfftw3', '-lm'])]
extensions = cythonize(extensions, language="c++",  include_path=[])


# package_data = {}
__version__ = "0.0.0"
exec(open('fitsne/_version.py').read())

setup(name="fitsne",
      version=__version__,
      packages=find_packages(),
      install_requires=['numpy', 'cython'],
      ext_modules=extensions,
      # package_data=package_data,
      # metadata
      author="TODO", # TODO
      author_email="TODO", # TODO
      url="https://github.com/KlugerLab/pyFIt-SNE",
      download_url=f"https://github.com/KlugerLab/pyFIt-SNE/archive/{__version__}.tar.gz",
      keywords=["tSNE", "embedding"],
      description="TODO", # TODO
      license="BSD2") # TODO

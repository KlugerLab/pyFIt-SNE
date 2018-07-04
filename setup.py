from setuptools import setup, find_packages  # from distutils.core import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from os import path
from os import environ

__version__ = "0.0.0"
exec(open('fitsne/_version.py').read())
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

author="George Linderman, Gioele La Manno"
author_email="george.linderman@gmail.com, gioelelamanno@gmail.com"
url="https://github.com/KlugerLab/FIt-SNE"
download_url="https://github.com/KlugerLab/pyFIt-SNE/archive/%s.tar.gz"%__version__
keywords=["tSNE", "embedding"]
description="Fast Fourier Transform-accelerated Interpolation-based t-SNE (FIt-SNE)"
license="BSD3"

#Try...except because for some OS X setups, the compilation fails without -stdlib=libc++
try: 
    extensions = [Extension("fitsne.cppwrap",
                            ["fitsne/cppwrap.pyx", "fitsne/src/nbodyfft.cpp", "fitsne/src/sptree.cpp", "fitsne/src/tsne.cpp"],
                            language="c++",
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
          author=author, 
          author_email=author_email, 
          url=url,
          download_url=download_url,
          keywords=keywords,
          description=description, 
          long_description=long_description,
          license=license) 

except:
    extensions = [Extension("fitsne.cppwrap",
                            ["fitsne/cppwrap.pyx", "fitsne/src/nbodyfft.cpp", "fitsne/src/sptree.cpp", "fitsne/src/tsne.cpp"],
                            language="c++",
                            extra_compile_args=["-std=c++11","-stdlib=libc++", "-O3", '-pthread', "-lfftw3", "-lm"],
                            extra_link_args=['-lfftw3', '-lm'])]
    extensions = cythonize(extensions, language="c++",  include_path=[])

    setup(name="fitsne",
          version=__version__,
          packages=find_packages(),
          install_requires=['numpy', 'cython'],
          ext_modules=extensions,
          author=author, 
          author_email=author_email, 
          url=url,
          download_url=download_url,
          keywords=keywords,
          description=description, 
          long_description=long_description,
          license=license) 

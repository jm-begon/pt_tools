#!/usr/bin/env python

# Authors: Jean-Michel Begon
#
# License: BSD 3 clause

from distutils.core import setup

import pt_tools

if __name__ == '__main__':
    setup(name="pt_inspector",
          version=pt_tools.__version__,
          author="Jean-Michel Begon",
          author_email="jm.begon@gmail.com",
          url="https://github.com/jm-begon/pt_tools",
          description="A small toolkit for PyTorch",
          long_description=open("README.md").read(),
          license="BSD3",
          classifiers=[
              'Development Status :: 2 - Pre-Alpha',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Education',
              'License :: OSI Approved :: BSD License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3.5',
              'Topic :: Scientific/Engineering',
              'Topic :: Utilities',
              'Topic :: Software Development :: Libraries',
          ],
          platforms="any",
          packages=["pt_tools", "pt_tools/architectures", "pt_tools/datasets",
                    "pt_tools/training", "pt_tools/inspector"])
    # dependence on scikit-learn, numpy


from __future__ import print_function, division, absolute_import
import setuptools
from codecs import open

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open('src/version.py').read())
setuptools.setup(
    name="aggressive-ensemble.pytorch",
    version=__version__,
    author="Maciej Pajak",
    author_email="mpajak98@gmail.com",
    description="A package implementing aggressive ensemble methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mpajak98/aggressive-ensemble.pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aggressive-ensemble",
    version="0.1.5",
    author="Maciej Pajak",
    author_email="mpajak98@gmail.com",
    description="A package implementing aggressive ensemble methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mpajak98/aggressive-ensemble.pytorch",
    project_urls={
        "Documentation": "https://github.com/mpajak98/aggressive-ensemble.pytorch/docs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6, <9",
    install_requires=[
        "pandas~=1.3.4",
        "torch~=1.9.1",
        "pretrainedmodels~=0.7.4",
        "torchvision~=0.10.1",
        "matplotlib~=3.4.3",
        "numpy~=1.21.3",
        "imgaug~=0.4.0",
        "Pillow~=8.4.0",
        "scikit-learn~=1.0",
        "setuptools~=47.1.0",
        "pkbar~=0.5"],
)
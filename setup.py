from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name="autotonne",
    version="0.1.1",
    description="Auto machine learning, deep learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/toandaominh1997/autotonne",
    author="Tonne",
    author_email="toandaominh1997@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["autotonne"],
    include_package_data=True,
    install_requires=required
)

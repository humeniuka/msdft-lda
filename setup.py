#!/usr/bin/env python
from os import path
import re
from io import open
from setuptools import setup

def get_property(property, package):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(property),
        open(path.join('src', package, '__init__.py')).read(),
    )
    return result.group(1)

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.rst'), encoding='utf8') as f:
    long_description = f.read()

setup(
    name='mlmsdft',
    version=get_property('__version__', 'mlmsdft'),
    description='Local Matrix Density Functional Approximation for Ground and Excited States',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/humeniuka/msdft-lda',
    author='Alexander Humeniuk',
    author_email='alexander.humeniuk@gmail.com',
    license='LICENSE.txt',
    package_dir = {"": "src"},
    scripts = ["scripts/msdft.py"],
    install_requires=[
        'becke-multicenter-integration==0.0.2', # only for testing purposes
        'coverage==7.9.1', # only for testing purposes
        'jsonargparse[signatures]==4.32.1',
        'matplotlib==3.9.2',
        'numpy==2.1.1',
        'opt-einsum==3.4.0',
        'pandas==2.3.1',
        'prefect==3.6.20',
        'pyscf==2.9.0',
        'ruff==0.12.1', # only for testing purposes
        'scipy==1.14.1',
        'torch==2.7.1',
        'tqdm==4.66.5'],
    include_package_data=True,
    zip_safe=False,
)

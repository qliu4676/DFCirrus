import os, sys
import builtins
from setuptools import setup, find_packages
sys.path.append('src')

abspath = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(abspath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []


setup(

    name='dfcirrus',

    version='0.1',

    description='Background Modeling of Galactic Cirrus for Dragonfly',

    long_description=long_description,

    url='https://github.com/NGC4676/DFCirrus',

    author='Qing Liu',  

    author_email='qliu@astro.utoronto.ca',  

    keywords='astronomy',

    packages=find_packages(where="src"),
    
    package_dir={"": "src"},

    python_requires='>=3.7',

    install_requires=install_requires,

)

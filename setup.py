import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


requirements = [
    "numpy",
    "pandas",
    "seaborn",
    "torch>=1.6.0",
    "torchvision",
    "datasets",
    "tokenizers",
    "sentence_transformers",
    "recordlinkage",
    "scikit-learn",
    "scipy",
    "nltk",
    "sentencepiece",
    "huggingface-hub",
    "jupyter",
    "tqdm",
    "ipywidgets",
    "matplotlib",
    "xgboost==1.4.2",
    "dask[complete]",
    "pydantic"
]

setup(
    name="pyent",
    version="0.0.1",
    url="",
    license='MIT',
    author="Mustafa Waheed",
    author_email="mustafawaheed2013@u.northwestern.edu",
    long_description=read("README.md"),
    description="A transformer architecture based deep learning model to resolve real world entities in data sources",
    packages=find_packages(exclude=('tests',)),
    # entry_points={
    #     'console_scripts': [
    #         'trainer=sever.cli:cli'
    #     ]
    # },
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11'
    ],
)

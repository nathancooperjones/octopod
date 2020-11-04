from setuptools import find_packages, setup

with open('octopod/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='octopod',
    version=__version__,
    description='General purpose multi-task classification library',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'fastprogress',
        'joblib',
        'numpy',
        'Pillow<7.0.0',
        'transformers>=2.3.0',
        'sentencepiece!=0.1.92',
        'scikit-learn',
        'torch',
        'torchvision==0.2.1',
        'wildebeest',
    ],
    extras_requires={
        'dev': [
            'flake8',
            'flake8-docstrings',
            'flake8-import-order',
            'm2r',
            'pydocstyle<4.0.0',
            'pytest',
            'pytest-cov',
            'sphinx-rtd-theme==0.4.3'
        ]
    },
)

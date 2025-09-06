# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import setup, find_packages


NAME = 'encodec-training'
DESCRIPTION = 'A modular implementation of EnCodec training with SEANet encoder/decoder'

URL = 'https://github.com/yourusername/encodec-training'
AUTHOR = 'Your Name'
EMAIL = 'your.email@example.com'
REQUIRES_PYTHON = '>=3.8.0'

# Get version from __init__.py
for line in open('__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']
        break
else:
    VERSION = '1.0.0'

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Core dependencies
REQUIRED = [
    'torch>=2.0.0',
    'torchaudio>=2.0.0',
    'soundfile>=0.12.0',
    'librosa>=0.10.0',
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'omegaconf>=2.3.0',
    'tqdm>=4.64.0',
    'matplotlib>=3.5.0',
    'einops>=0.6.0',
    'wandb>=0.15.0',
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author_email=EMAIL,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require={
        'dev': ['coverage', 'flake8', 'mypy', 'pdoc3', 'pytest', 'black'],
        'audio': ['resampy>=0.4.0', 'pyworld>=0.3.0'],
        'distributed': ['torch-distributed>=2.0.0'],
    },
    packages=find_packages(),
    include_package_data=True,
    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'encodec-train=main:main',
        ],
    },
)

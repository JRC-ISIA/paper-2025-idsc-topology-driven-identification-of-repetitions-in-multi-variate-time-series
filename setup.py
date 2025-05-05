from distutils.core import setup

setup(name='cycle_detection',
    version='1.0',
    description='Topology-driven identification of repetitions in multi-variate time series',
    author='Elias Reich, Stefan Huber, Simon Schindler',
    author_email='eliassteffen.reich@fh-salzburg.ac.at',
    url='https://github.com/JRC-ISIA/paper-2025-idsc-topology-driven-identification-of-repetitions-in-multi-variate-time-series/',
    packages=[
        'cycle_detection', 
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.4',
        'scipy==1.10.1',
        'matplotlib>=3.7.5',
    ]
)

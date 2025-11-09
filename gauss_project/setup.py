from setuptools import setup, find_packages

setup(
    name='gauss-gnn',
    version='1.0.0',
    description='GrAph-customized Universal Self-Supervised Learning (GAUSS)',
    author='Research Team',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'torch-geometric>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',
        'networkx>=2.6.0',
        'tqdm>=4.62.0',
        'pandas>=1.3.0',
    ],
    python_requires='>=3.7',
)

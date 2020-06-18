from setuptools import find_packages, setup


with open('meezer/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='meezer',
    version=__version__,
    description='Supervised Siamese networks using hard negative examples.',
    long_description=readme,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'annoy>=1.15.2',
        'ipywidgets',
        'jupyterlab',
        'matplotlib',
        'numpy',
        'scikit-learn>0.20.0',
        'tensorflow==1.15.2',  # Tensorflow v2 is slower with many API-breaking changes
        'tqdm'
    ],
)

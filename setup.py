from setuptools import setup

setup(
    name='photdiag',
    version='0.1.0',
    author='Ivan Usov',
    author_email='ivan.usov@psi.ch',
    description='Photon diagnostics tools at SwissFel',
    license='GNU GPLv3',
    url='',
    packages=['photdiag'],
    install_requires=['h5py', 'numpy', 'scipy'],
)

from setuptools import setup

setup(
    name='photon-diag',
    version='0.1.0',
    author='Ivan Usov',
    author_email='ivan.usov@psi.ch',
    description='Photon diagnostics tools at SwissFel',
    license='',
    url='',
    packages=['photon_diag'],
    install_requires=['h5py', 'numpy', 'scipy'],
)

import re

from setuptools import find_packages, setup

with open("photodiag/__init__.py") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="photodiag",
    version=version,
    author="Ivan Usov",
    author_email="ivan.usov@psi.ch",
    description="Photon diagnostics tools at SwissFel",
    packages=find_packages(),
    package_data={"": ["static/*"]},
    include_package_data=True,
    license="GNU GPLv3",
)

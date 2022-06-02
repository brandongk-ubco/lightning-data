import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(name="lightningdata",
                 version=os.environ.get("RELEASE_VERSION", "alpha"),
                 author="Brandon Graham-Knight",
                 author_email="brandongk@alumni.ubc.ca",
                 description="Datasets for PyTorch Lightning.",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/brandongk-ubco/lightning-data",
                 packages=['lightningdata'],
                 install_requires=requirements,
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires='>=3.8',
                 include_package_data=True)

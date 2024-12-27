from setuptools import setup, find_packages

setup(
    name="PrepPy",
    version="0.1.0",
    author="Nathan Ayers",
    author_email="naprimarycontact@gmail.com",
    description="A Python library for preprocessing datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nayers17/preppy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "scikit-learn",
    ],
)

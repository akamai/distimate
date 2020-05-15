
from setuptools import setup, find_packages


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(
    name="distimate",
    version="0.0.dev",
    author="Miloslav Pojman",
    author_email="mpojman@akamai.com",
    description="Approximate statistical distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
    extras_require={
        "dev": ["flake8", "pytest"],
    },
    zip_safe=False,
    include_package_data=True,
)

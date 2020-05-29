from setuptools import setup, find_packages


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(
    name="distimate",
    version="0.0.dev",
    author="Miloslav Pojman",
    author_email="mpojman@akamai.com",
    description="Distributions visualized",
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
        "pandas": ["pandas>=1.0.0"],
    },
    classifiers=[
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Framework :: Matplotlib",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    zip_safe=False,
    include_package_data=True,
)

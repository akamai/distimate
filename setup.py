# Copyright 2020 Akamai Technologies, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages


with open("README.md", "r") as fp:
    long_description = fp.read()


setup(
    name="distimate",
    version="0.0.dev",
    author="Miloslav Pojman",
    author_email="mpojman@akamai.com",
    description="Distributions visualized",
    license="Apache License 2.0",
    url="https://github.com/akamai/distimate",
    project_urls={
        "Documentation": "https://distimate.readthedocs.io/",
        "Source Code": "https://github.com/akamai/distimate",
    },
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
        "License :: OSI Approved :: Apache Software License",
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

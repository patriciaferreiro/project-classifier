import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="projectclassifier",
    version="0.0.1",
    author="Patricia Ferreiro",
    author_email="patricia.ferreiro@databricks.com",
    description="Project name classifier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patriciaferreiro/project-classifier",
    project_urls={
        "Bug Tracker": "https://github.com/patriciaferreiro/project-classifier/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
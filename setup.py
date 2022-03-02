import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="parallel_shap",
    version="0.0.1",
    author="Name",
    author_email="micahmelling@gmail.com",
    description="Calculate SHAP values in parallel and auto-detection of what explainer to use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['parallel_shap'],
    install_requires=['shap>=0.35.0', 'pandas>=1.1.5', 'numpy>=1.19.4', 'matplotlib>=3.2.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

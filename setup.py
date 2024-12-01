import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="auto_shap",
    version="1.0.0",
    author="Micah Melling",
    author_email="micahmelling@gmail.com",
    description="Calculate SHAP values in parallel and automatically detect what explainer to use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/micahmelling/auto-shap',
    license="MIT",
    packages=['auto_shap'],
    install_requires=['shap>=0.46.0', 'pandas>=2.2.3', 'numpy>=1.26.4', 'matplotlib>=3.9.3', "numba>=0.60.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

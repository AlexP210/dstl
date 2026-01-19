from setuptools import setup, find_packages

setup(
    name="dstl",
    version="0.1.0",
    description="DSTL: Distilling Specialized Task Latents",
    author="TDMPC2 Authors",
    author_email="",  # Add author email if available
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'numpy',
        'gym',
        'pyyaml',
    ],
    include_package_data=True,
    package_data={
        'dstl': ['config.yaml'],
    }
)
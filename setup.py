from setuptools import setup, find_packages

setup(
    name="discovery_markets",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        # Example: 'requests==2.25.1',
    ],
    entry_points={
        "console_scripts": [
            "discovery_markets=main:main",
        ],
    },
)

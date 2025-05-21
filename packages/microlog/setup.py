from setuptools import find_packages, setup

setup(
    name="microlog",
    version="0.0.1",
    description="Centralized logging in microservice environments.",
    url="",
    author="Philipp Würfel",
    author_email="phi.wuerfel@gmail.com",
    license="TBD",
    packages=find_packages(),
    package_data={"microlog": ["dbs/db_files/*", "data/*"]},
    install_requires=[
        # add packages e.g. for elasticsearch
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ]
    },
    python_requires=">=3.11",
)

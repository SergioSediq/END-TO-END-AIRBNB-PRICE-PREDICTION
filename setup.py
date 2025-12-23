from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Read requirements from file and clean them."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove whitespace and newlines
        requirements = [req.strip() for req in requirements]
        # Remove empty lines and comments
        requirements = [req for req in requirements if req and not req.startswith('#')]
        # Remove -e . if present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_requirements = []
    for req in requirements:
        if req not in seen:
            seen.add(req)
            unique_requirements.append(req)
    
    return unique_requirements

setup(
    name="AirbnbPricePrediction",
    version="0.0.1",
    author="Sergio Sediq",
    author_email="tunsed11@gmail.com",
    description="End-to-end machine learning project for Airbnb price prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SergioSediq/END-TO-END-AIRBNB-PRICE-PREDICTION",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
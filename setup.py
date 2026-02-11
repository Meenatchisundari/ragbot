from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RAGBOT",
    version="0.1",
    author="Meenatchi Sundari",
    description="Updated medical information chatbot",
    packages=find_packages(),
    install_requires = requirements,
)
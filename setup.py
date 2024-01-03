from setuptools import setup

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='FoulDetection',
    version='1.0.0',
    description='Computer Vision model to detect football foul',
    author='Blackwoj',
    author_email='wojtek@nikiel.org',
    url='https://link.do.twojego/repozytorium',
    packages=[],
    install_requires=install_requires,
)

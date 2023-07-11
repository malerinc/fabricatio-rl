from setuptools import setup, find_packages
import unittest


with open("MANIFEST.in", "w") as f:
    f.write("recursive-include fabricatio_rl/visualization *")

with open("README.md", 'r') as f:
    long_description = f.read()


def fabricatio_test_suite():
    # TODO: move to pytest ;)
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


setup(
    name='fabricatio-rl',
    version='1.0.0',
    python_requires='>3.6.8',
    install_requires=[
      'gym==0.18.3', 'numpy==1.18.5', 'pandas==1.1.1', 'scipy==1.4.1',
      'Flask==1.1.1', 'Flask-RESTful==0.3.8'
    ],
    description="An Event Discrete Simulation Framework for "
                "Production Scheduling Problems.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('fabricatio_rl'),
    include_package_data=True,
    test_suite='setup.fabricatio_test_suite'
)

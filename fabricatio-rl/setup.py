import os

from setuptools import setup, find_packages
# import unittest


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


visualization_app_paths = package_files('./fabricatio_rl/visualization')


with open("MANIFEST.in", "w") as f:
    f.write("include")
    f.write(" ".join(visualization_app_paths))

try: 
    with open("README.md", 'r') as f:
        long_description = f.read()
except FileNotFoundError: 
    print("No README.md found")
    long_description = ""

# def fabricatio_test_suite():
#     # TODO: move to pytest ;)
#     test_loader = unittest.TestLoader()
#     test_suite = test_loader.discover('tests', pattern='test_*.py')
#     return test_suite


setup(
    name='fabricatio-rl',
    version='1.0.0',
    python_requires='>3.6.8',
    install_requires=[
      'gym==0.18.3', 'numpy~=1.19.0', 'pandas==1.1.1', 'scipy==1.5.4',
      'Flask==2.2.2', 'Flask-RESTful>0.3.8', 'protobuf==3.20.0'
    ],
    description="An Event Discrete Simulation Framework for "
                "Production Scheduling Problems.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'': visualization_app_paths},
    include_package_data=True #,
    # test_suite='setup.fabricatio_test_suite'
)

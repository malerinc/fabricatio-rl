from setuptools import setup, find_packages
# import unittest


with open("MANIFEST.in", "w") as f:
    f.write("recursive-include . *")

# with open("README.md", 'r') as f:
#     long_description = f.read()


setup(
    name='fabricatio-rl',
    version='1.0.0',
    python_requires='>3.6.8',
    install_requires=[
        'fabricatio-rl==1.0.0', 'tensorflow==2.3.0', 'tensorboard==2.3.0',
        'torch==1.10.2', 'scipy==1.5.4', 'pandas==1.1.1', 'seaborn==0.11.2',
        'pathos==0.2.6', 'ortools==7.8.7959', 'stable-baselines3==1.1.0'
    ],
    description="Scheduling algorithms for FabricatioRL.",
    # long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('fabricatio_rl'),
    include_package_data=True,
)

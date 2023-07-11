from setuptools import setup, find_packages

try:
    import fabricatio_rl
except ModuleNotFoundError:
    print("Before installing this package, you must first install fabricatio-rl!")
    quit()

setup(
    name='fabricatio-controls',
    version='1.0.0',
    python_requires='>3.6.8',
    install_requires=[
        'tensorflow==2.6.2', 'torch==1.10.2', 
        'scipy==1.5.4', 'pandas==1.1.1', 'seaborn==0.11.2',
        'pathos==0.2.6', 'ortools==7.8.7959', 'stable-baselines3==1.1.0'
    ],
    description="Scheduling algorithms for FabricatioRL.",
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
)

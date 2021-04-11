from setuptools import setup


setup(name='gym_fabrikatioRL',
      version='0.0.1',
      install_requires=[
          'gym',     # for gym interface
          'numpy',   # for rng, matrix transformations etc.
          'pandas',  # used with logging
          'scipy'    # for distributions, initial state sampling
      ],
      # And any other dependencies gym_fabrikatioRL needs
      include_package_data=True, zip_safe=False)

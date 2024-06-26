# from setuptools import setup
from distutils.core import setup

from setuptools import find_packages


# This setup must run first the requirement file.
setup(name='human_motion_prediction',
      version='0.0.1',
      packages=find_packages(),
      exclude=["__pycache__"],
      package_data={'': ['*.py', '*.yaml', '*.json', '*.sh']},
      install_requires=['numpy',
                        'gymnasium-robotics[all]',
                        'gymnasium[all]',
                        'mujoco==2.3.3',
                        # 'pandas',
                        # 'scalpl', # remove it
                        ]  # And any other dependencies foo needs
      )

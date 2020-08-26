from setuptools import setup

setup(
   name='cart',
   version='1.0.0',
   description='cart-regression-trees',
   author='Niall O\'Riordan',
   packages=['cart'],  # same as name
   install_requires=['pandas', 'numpy','sklearn'], # external packages as dependencies
)
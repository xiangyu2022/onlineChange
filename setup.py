# Setup Information for the package
from setuptools import setup, find_packages
from os.path import dirname, join, realpath

with open(join(dirname(realpath(__file__)), "README.md")) as f:
	long_description = f.read()

setup(
    name = 'onlineChange',
    version = '0.0.6',
    description='Quickest change detection algorithms for online streaming data in python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Yorkee2018/cpd.python',
    author = 'Xiangyu Zhang, Yi Rong, Jie Ding',
    author_email="zhan6004@umn.edu, dingj@umn.edu",
    packages = ['onlineChange'],
    license = 'GNU GPLv3',
	classifiers=[
	'Topic :: Education',
	'Topic :: Scientific/Engineering :: Information Analysis',
	'Topic :: Scientific/Engineering :: Mathematics',
	'Topic :: Scientific/Engineering',
	'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
	'Programming Language :: Python',
	'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8',
	'Intended Audience :: Education',
	'Intended Audience :: Developers',
	'Intended Audience :: Information Technology',
	'Intended Audience :: Science/Research',
    ],
)

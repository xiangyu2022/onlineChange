# The 'onlineChange' python package
The onlineChange python package is designed to quickest detect any change in distributions for online streaming data, supporting any user-specified distributions. It supports sequential Monte Carlo to perform Bayesian change point analysis, where the **parameters before or after the change can be unknown**. It also supports likelihood ratio based test to determine the stopping time (i.e. when change occurs).

This is still an ongoing project by a group of researchers from the University of Minnesota. The published software here is an initial version that is ready to use.

For more technical problems, please contact the author of the package Xiangyu Zhang at zhan6004@umn.edu.

For references, please contact Jie Ding at dingj@umn.edu.

## A Quick Setup Guide

### Getting Started

#### 1. Install the 'onlineChange' package using pip

```bash
# Installing test package
python -m pip install onlineChange

```
#### 2. Import the Model and Experiment API classes

```python
from onlineChange import stat, bayes, bayes_unknown_pre
```
## Using This Package

A quick guide of package can be found [here](https://github.com/JieGroup/onlineChange/blob/master/vignettes/user-guide.pdf).

## Acknowledgment

This research is funded by the Defense Advanced Research Projects Agency (DARPA) under grant number HR00111890040.

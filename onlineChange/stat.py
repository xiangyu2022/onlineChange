'''
    Statistic-based Stopping Rule for Continuous Data Changepoint detection for continuous data
    with known post-change distributions using the statistic-based stopping rule.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from math import *
from time import process_time

'''
    The introduction of the function parameters.

    GEN: A function of time that returns an observation.

    alpha: A numeric parameter in (0,1) that controls the probability of false alarm.

    nulower,nuupper: Optional nonnegative numerics, the earliest and latest time of changepoint
    based on prior belief. The default is nulower=0 and nuupper=18 which corresponds to the
    geometric prior distribution with p=0.1.

    score: An optional character specifying the type of score to be used:
    The default "hyvarinen" or the conventional "logarithmic".
    Case insensitive.

    ULP0,GLP0,LLP0,ULP1,GLP1,LLP1: Functions of an observation: The log unnormalized probability
    function, its gradient and its laplacian for the pre-change (ULP0,GLP0,LLP0) and post-change
    (ULP1,GLP1,LLP1) models. If score="hyvarinen", GLP0,LLP0,GLP1,LLP1,ULP0,ULP1 is required.
    If score="logarithmic", only ULP0,ULP1 is required.

    par0,par1 Numeric parameters for the pre-change (par0) and post-change (par1) models.

    lenx A positive numeric: The length of the variable of an obervation. Optional if score="hyvarinen"
    or if score="logarithmic" and par0,par1 are specified.

    lower0,upper0,lower1,upper1 Optional numeric vectors of length lenx:
    The lower and upper limits of an observation from the pre-change (lower0,upper0)
    and post-change (lower1,upper1) models. The defaults are infinite.

    return: A positive numeric: The stopping time.

'''


def detect_stat(GEN, alpha, nulower=None,nuupper=None,score="hyvarinen",
                ULP0=None,GLP0=None,LLP0=None,ULP1=None,GLP1=None,LLP1=None,
                par0=None,par1=None,
                lenx=None,lower0=None,upper0=None,lower1=None,upper1=None):

    # check the false alarm rate
    if alpha <= 0 or alpha >=1:
        raise Exception("'alpha' should be in (0,1)")

    # suppose the time of changepoint nu follows a geometric prior distribution, compute the parameter p
    # the default is nulower=0, nuupper=18, which corresponds to p=0.1
    if nulower is None:
        nulower = 0
    if nuupper is None:
        p = 0.1
    else:
        if nulower >= nuupper:
            raise Exception("need 'nulower' < 'nuupper'")
        p = 1/(1+ (nulower+nuupper)/2)

    # chose score and compute related functions
    score = score.lower()
    if score != "hyvarinen" and score != "logarithmic":
        raise Exception("Need score to be ")
    if score == "hyvarinen":
        # some checks, preparations, and transformations of the log probablity functions
        if GLP0 is None:
            raise Exception("Need gradient for the pre log unnormalized probability function")
        if LLP0 is None:
            raise Exception("Need laplacian for the pre log unnormalized probability function")
        if GLP1 is None:
            raise Exception("Need gradient for the post log unnormalized probability function")
        if LLP1 is None:
            raise Exception("Need laplacian for the post log unnormalized probability function")
        if par0 is None:
            par0 = 1
        elif par0 <=0:
            warnings.warn("'par0' should be positive for hyvarinen score. Use par0=1")
            par0 = 1
        if par1 is None:
            par1 = 1
        elif par1 <=0:
            warnings.warn("'par1' should be positive for hyvarinen score. Use par0=1")
            par1 = 1

        # compute the scores for the pre and post change models
        SC0 = lambda x: par0*(np.sum(GLP0(x)**2)/2 + LLP0(x))
        SC1 = lambda x: par1*(np.sum(GLP1(x)**2)/2 + LLP1(x))
    else:
        # when we choose logarithmic scores
        if par0 is None or par1 is None:
            if lenx is None:
                raise Exception("'lenx' is missing")
            if lenx <=0:
                raise Exception("'lenx' should be a positive integer")
            if lower0 is None:
                ninf = np.array([np.NINF])
                lower0 = np.repeat(ninf, lenx)
            if lower1 is None:
                ninf = np.array([np.NINF])
                lower1 = np.repeat(ninf, lenx)
            if upper0 is None:
                inf = np.array([np.inf])
                upper0 = np.repeat(inf, lenx)
            if upper1 is None:
                inf = np.array([np.inf])
                upper1 = np.repeat(inf, lenx)
        if par0 is None:
            fn = lambda x: exp(ULP0(x))
            par0 = log(integrate.quad(fn, lower0, upper0)[0])
        if par1 is None:
            fn = lambda x:exp(ULP1(x))
            par1 = log(integrate.quad(fn, lower1, upper1)[0])
        SC0 = lambda x: -ULP0(x) + par0
        SC1 = lambda x: -ULP1(x) + par1
    logA = log(1-alpha)-log(alpha)-log(p)
    logR = np.NINF
    t = 1
    while t<=nulower or logR<logA:
        x = GEN(t)
        z = SC0(x) - SC1(x)
        logR = log(1 + exp(logR))+z-log(1-p)
        t=t+1
    return t

'''
Statistic-based Stopping Rule for Binary Data
Changepoint detection for binary data with known post-change distributions
using the statistic-based stopping rule.

filp is a helper function.

    GEN: A function of time that returns an observation.

    alpha: A numeric parameter in (0,1) that controls the probability of false alarm.

    nulower,nuupper: Optional nonnegative numerics, the earliest and latest time of changepoint
    based on prior belief. The default is nulower=0 and nuupper=18 which corresponds to the
    geometric prior distribution with p=0.1.

    score: An optional character specifying the type of score to be used:
    The default "hyvarinen" or the conventional "logarithmic".
    Can be abbreviated. Case insensitive.

    ULP0, ULP1: Functions of an observation: The log unnormalized probability
    function for the pre-change and post-change models

    par0,par1: Optional numeric parameters for the pre-change (par0) and post-change (par1) models.

    lenx A positive numeric: The length of the variable of an obervation. Optional if score="hyvarinen"
    or if score="logarithmic" and par0,par1 are specified.

    tbin Optional numeric specifying the binary type: The default
    tbin=1 representing {1,0} or the alternative tbin=2 representing {1,-1}.

    return: A positive numeric: The stopping time.
'''
def flip(x, ULP, tbin):
  # Function to do binary flip
  #
  # Args:
  #   x: a vector
  #   ULP: a log unnormalized probability function
  #   tbin: a scalar
  #
  # Returns:
  #   Returns a vector, the ith component of which is ULP(tbin-x[i]).
    x = np.array([x])
    lenx = len(x)
    vec = np.zeros(lenx)
    for i in range(lenx):
        y = x
        y[i] = tbin - x[i]
        vec[i] = ULP(y)
    return vec



def detect_bin_stat(GEN, alpha, ULP0, ULP1, nulower=None,
                    nuupper=None,score="hyvarinen",
                    par0=None,par1=None,lenx=None,tbin=1):
    if alpha <= 0 or alpha >= 1:
        raise Exception("'alpha' should be in (0,1)")
    if nulower is None:
        nulower = 0
    if nuupper is None:
        p = 0.1
    else:
        if nulower >= nuupper:
            raise Exception("need 'nulower' < 'nuupper'")
        p = 1/(1 + (nulower +nuupper)/2)
    score = score.lower()
    if re.match(score, "hyvarinen"):
        score = "hyvarinen"
    if re.match(score, "logarithmic"):
        score = "logarithmic"
    if score == "hyvarinen":
        if par0 is None:
            par0 = 1
        elif par0 <=0:
            warnings.warn("'par0' should be positive for hyvarinen score. Use par0=1")
            par0 = 1
        if par1 is None:
            par1 = 1
        elif par1 <=0:
            warnings.warn("'par1' should be positive for hyvarinen score. Use par0=1")
            par1 = 1
        SC0 = lambda x: par0*np.sum((1+exp(ULP0(x) - flip(x, ULP0, tbin)))**(-2))
        SC1 = lambda x: par1*np.sum((1+exp(ULP1(x) - flip(x, ULP1, tbin)))**(-2))
    else:
        if par0 is None or par1 is None:
            dom = np.array(list(np.ndindex(tuple(np.repeat([2], lenx)))))
            if tbin ==2:
                dom[dom==0] =-1

        if par0 is None:
            n = np.shape(dom)[0]
            a = np.zeros(n)
            for i in range(n):
                a[i] = exp(ULP0(dom[i]))
            par0 = log(np.sum(a))
        if par1 is None:
            n = np.shape(dom)[0]
            a = np.zeros(n)
            for i in range(n):
                a[i] = exp(ULP1(dom[i]))
            par1 = log(np.sum(a))

        SC0 = lambda x: -ULP0(x) + par0
        SC1 = lambda x: -ULP1(x) + par1

    logA = log(1-alpha) -log(alpha)-log(p)
    logR = np.NINF
    t = 1
    while t<=nulower or logR<logA:
        x = GEN(t)
        z = SC0(x) - SC1(x)
        logR = log(1 + exp(logR))+z-log(1-p)
        t=t+1
    return t

'''
Simulation using Statistic-based Stopping Rule.

Simulation experiment of changepoint detection with known post-change
distributions using the statistic-based stopping rule.

Return is a named numeric vector with components
is.FA {A numeric of 1 or 0 indicating whether a false alarm has been raised.}
DD  {A positive numeric: The delay to detection.}
CT  {A positive numeric: The computation time.}

'''


def sim_detect_stat(GEN0, GEN1, detect_stat1, alpha, nulower = None, nuupper = None, score="hyvarinen",
                    ULP0=None,GLP0=None,LLP0=None,ULP1=None,GLP1=None,LLP1=None,
                    par0=None, par1=None,
                    lenx=None,lower0=None,upper0=None,lower1=None,upper1=None, tbin=1):
    if nulower is None:
        nulower = 0
    if nuupper is None:
        p = 0.1
    else:
        if nulower >= nuupper:
            raise Exception("need 'nulower' < 'nuupper'")
        p = 1/(1 + (nulower +nuupper)/2)
    nu = np.random.geometric(p, 1) + nulower
    def GEN(t):
        if nu >= t:
            return GEN0()
        else:
            return GEN1()

    CT0 = process_time()
    if detect_stat1 is detect_stat:
        t = detect_stat1(GEN=GEN, alpha=alpha, nulower=nulower, nuupper=nuupper, score=score,
                        ULP0=ULP0, GLP0=GLP0,LLP0=LLP0,ULP1=ULP1,GLP1=GLP1,LLP1=LLP1,
                        par0=par0, par1=par1,
                        lenx=lenx, lower0=lower0,upper0=upper0,lower1=lower1,upper1=upper1)
    else:
        t = detect_stat1(GEN=GEN, alpha=alpha, ULP0=ULP0, ULP1=ULP1, nulower=nulower,
                        nuupper=nuupper,score=score,
                        par0=par0, par1=par1, lenx=lenx, tbin=1)
    CT1 = process_time()
    CT = CT1 - CT0
    fa = -1
    if nu>=t:
        fa = 1
    else:
        fa = 0
    out={}
    out["is.FA"]=fa
    out["DD"]=max(t-nu, [0])[0]
    out["CT"]=CT
    print("is.FA:", fa, "     DD:", max(t-nu, [0])[0], "     CT:", CT)
    return out

'''
Example 1 about detect_stat.

Change from N(0,1) to N(1,1) at t=15.
Prior knowledge suggests change occurs between 10 and 25.
We choose:
def GEN(t):
    if t<=15:
        return (np.random.normal(0, 1, 1)[0])
    else:
        return (np.random.normal(0, 1, 1)[0] + 1)
ULP0=lambda x: -x**2/2
ULP1=lambda x: -(x-1)**2/2
GLP0=lambda x: (-1) *x
GLP1=lambda x: -(x-1)
LLP0=lambda x: -1
LLP1=lambda x: -1
par0=log(2*pi)/2
par1=par0

# using hyvarinen score
print(detect_stat(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1))

# using log score. normalizing constant is unknown
print(detect_stat(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,lenx=1))

# using log score. normalizing constant is known
print(detect_stat(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1))

For your convenience, you could just copy and use these codes to show the example1:
def GEN(t):
    if t<=15:
        return (np.random.normal(0, 1, 1)[0])
    else:
        return (np.random.normal(0, 1, 1)[0] + 1)

ULP0=lambda x: -x**2/2
ULP1=lambda x: -(x-1)**2/2
GLP0=lambda x: (-1) *x
GLP1=lambda x: -(x-1)
LLP0=lambda x: -1
LLP1=lambda x: -1
par0=log(2*pi)/2
par1=par0

print(detect_stat(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1))
print(detect_stat(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,lenx=1))
print(detect_stat(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1))
'''

'''
Example 2 about detect_bin_stat:

Change from 3 iid Bernoulli(0.2) to 3 iid Bernoulli(0.8) at t=10.
Prior knowledge suggests change occurs before 20.

def GEN(t):
    if(t <= 10):
        return np.random.binomial(3, 0.2, 1)[0]
    else:
        return np.random.binomial(3, 0.8, 1)[0]
ULP0 = lambda x: np.sum(x)*(log(0.2)-log(1-0.2))
ULP1 = lambda x: np.sum(x)*(log(0.8)-log(1-0.8))
par0 = -3*log(1-0.2)
par1 = -3*log(1-0.8)

# using hyvarinen score
print(detect_bin_stat(GEN=GEN,alpha=0.1,nuupper=20,ULP0=ULP0,ULP1=ULP1))

# using log score. normalizing constant is unknown
print(detect_bin_stat(GEN=GEN,alpha=0.1,nuupper=20,score="log",ULP0=ULP0,ULP1=ULP1,lenx=3))

# using log score. normalizing constant is known
print(detect_bin_stat(GEN=GEN,alpha=0.1,nuupper=20,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1))

For your convenience, you could just copy and use these codes to show the example2:
def GEN(t):
    if(t <= 10):
        return np.random.binomial(3, 0.2, 1)[0]
    else:
        return np.random.binomial(3,0.8,1)[0]
ULP0 = lambda x: np.sum(x)*(log(0.2)-log(1-0.2))
ULP1 = lambda x: np.sum(x)*(log(0.8)-log(1-0.8))
par0 = -3*log(1-0.2)
par1 = -3*log(1-0.8)

print(detect_bin_stat(GEN=GEN,alpha=0.1,nuupper=20,ULP0=ULP0,ULP1=ULP1))
print(detect_bin_stat(GEN=GEN,alpha=0.1,nuupper=20,score="log",ULP0=ULP0,ULP1=ULP1,lenx=3))
print(detect_bin_stat(GEN=GEN,alpha=0.1,nuupper=20,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1))
'''


# # Example 3 about sim_detect_stat using detect_stat

# def GEN0():
#     return (np.random.normal(0, 1, 1)[0])
# def GEN1():
#     return (np.random.normal(0, 1, 1)[0] + 1)

# ULP0=lambda x: -x**2/2
# ULP1=lambda x: -(x-1)**2/2
# GLP0=lambda x: (-1) *x
# GLP1=lambda x: -(x-1)
# LLP0=lambda x: -1
# LLP1=lambda x: -1
# par0=log(2*pi)/2
# par1=par0
# # using hyvarinen score

# for i in range(1000):
#     sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_stat,nulower=10,nuupper=25,alpha=0.1,GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1)


# # using log score. normalizing constant is unknown

# for i in range(1000):
#     sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_stat,nulower=10,nuupper=25,alpha=0.1,score="log",ULP0=ULP0,ULP1=ULP1,lenx=1)


# #using log score. normalizing constant is known

# for i in range(1000):
#     sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_stat,nulower=10,nuupper=25,alpha=0.1,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)




'''
Example 4 about sim_detect_stat using detect_bin_stat
Change from 3 iid Bernoulli(0.2) to 3 iid Bernoulli(0.8) occurs before 20.

def GEN0():
    return (np.random.binomial(1, 0.2, 3))
def GEN1():
    return (np.random.binomial(1, 0.8, 3))

ULP0= lambda x: np.sum(x)*(log(0.2)-log(1-0.2))
ULP1= lambda x: np.sum(x)*(log(0.8)-log(1-0.8))
par0=-3*log(1-0.2)
par1=-3*log(1-0.8)

# using hyvarinen score
sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=0.1,ULP0=ULP0,ULP1=ULP1)
# using log score. normalizing constant is unknown
sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=0.1,score="log",ULP0=ULP0,ULP1=ULP1,lenx=3)
# using log score. normalizing constant is known
sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=0.1,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)

For your convenience, you could just copy and use these codes to show the example3:

def GEN0():
    return (np.random.binomial(1, 0.2, 3))
def GEN1():
    return (np.random.binomial(1, 0.8, 3))

ULP0= lambda x: np.sum(x)*(log(0.2)-log(1-0.2))
ULP1= lambda x: np.sum(x)*(log(0.8)-log(1-0.8))
par0=-3*log(1-0.2)
par1=-3*log(1-0.8)

sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=0.1,ULP0=ULP0,ULP1=ULP1)
sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=0.1,score="log",ULP0=ULP0,ULP1=ULP1,lenx=3)
sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=0.1,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)
'''

'''
def GEN0():
    return (np.random.binomial(1, 0.2, 3))
def GEN1():
    return (np.random.binomial(1, 0.8, 3))

ULP0= lambda x: np.sum(x)*(log(0.2)-log(1-0.2))
ULP1= lambda x: np.sum(x)*(log(0.8)-log(1-0.8))
par0=-3*log(1-0.2)
par1=-3*log(1-0.8)


sum = 0
alpha=np.zeros(9)
for i in range(9):
    alpha[i]=0.1*(i+1)

hfa=np.zeros(9); hdd=np.zeros(9)
lfa=np.zeros(9); ldd=np.zeros(9)
hpt=np.zeros(9); lpt=np.zeros(9)

for i in range(9):
    sumhfa=0;sumhdd=0;sumhpt=0;
    sumlfa=0;sumldd=0;sumlpt=0
    for j in range(1000):
        a=sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=alpha[i],ULP0=ULP0,ULP1=ULP1)
        b=sim_detect_stat(GEN0=GEN0,GEN1=GEN1,detect_stat1=detect_bin_stat,nuupper=20,alpha=alpha[i],score="log",ULP0=ULP0,ULP1=ULP1,lenx=3)
        sumhfa=sumhfa+a["is.FA"]
        sumhdd=sumhdd+a["DD"]
        sumhpt=sumhpt+a["CT"]
        sumlfa=sumlfa+b["is.FA"]
        sumldd=sumldd+b["DD"]
        sumlpt=sumlpt+b["CT"]
    hfa[i]=sumhfa/1000
    hdd[i]=sumhdd/1000
    hpt[i]=sumhpt/1000
    lfa[i]=sumlfa/1000
    ldd[i]=sumldd/1000
    lpt[i]=sumlpt/1000

x=hfa
y=hdd
y1=ldd
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, 'r-')
lns2 = ax.plot(x, y1, 'b-')
ax.grid()
ax.set_xlabel("false alarm rate")
ax.set_ylabel(r"average delay")

plt.show()

x=alpha
y=hfa
y1=lfa
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, 'r-')
lns2 = ax.plot(x, y1, 'b-')
lns3 = ax.plot(x, x, 'y-')
ax.grid()
ax.set_xlabel("alpha")
ax.set_ylabel(r"prob of false alarm")

plt.show()

x=hfa
y=hdd
y1=ldd
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, 'r-')
lns2 = ax.plot(x, y1, 'b-')
ax.grid()
ax.set_xlabel("false alarm rate")
ax.set_ylabel(r"average delay")

plt.show()

x=alpha
y=hdd
y1=ldd
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, 'r-')
lns2 = ax.plot(x, y1, 'b-')
ax.grid()
ax.set_xlabel("alpha")
ax.set_ylabel(r"prob of false alarm")

plt.show()
'''

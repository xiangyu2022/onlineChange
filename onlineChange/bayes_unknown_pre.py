import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import re
from math import *
from time import process_time

def f2i(x, a, b):
    '''
    A helper function which transform from finite to infinite.
    look at https://mc-stan.org/docs/2_18/reference-manual/variable-transforms-chapter.html
    to get more details.
    '''
    if isinf(a) and isinf(b):
        return x
    elif isinf(b):
        return (log(x-a))
    elif isinf(a):
        return (log(b-x))
    else:
        return (log(x-a)-log(b-x))





def i2f(y, a, b):
    '''
    A helper function which transform from infinite to finite.
    look at https://mc-stan.org/docs/2_18/reference-manual/variable-transforms-chapter.html
    to get more details.
    '''

    if isinf(a) and isinf(b):
        return y
    elif isinf(b):
        return (exp(y) + a)
    elif isinf(a):
        return (b-exp(y))
    else:
        return (a+(b-a)/(1+exp(-y)))



def logJ(y,a,b):
    ''' Look at https://mc-stan.org/docs/2_18/reference-manual/variable-transforms-chapter.html for more details

        Args:
        x, a, b: float numbers which might be infinite.

        Returns:
        A float or integer number.

        Examples:
        print(logJ(3, np.inf, np.inf))
            0
        print(logJ(3, 4, np.inf))
            3
        print(logJ(5, 3, np.inf))
            5
    '''
    if isinf(a) and isinf(b):
        return 0
    elif isinf(b) or isinf(a):
        return y
    else:
        return (-y-2*log(1+exp(-y)))


def v2(xoy, a, b, FUN):
    ''' Apply transformation to three vectors

        Args:
        xoy, a, b: vectors which include float numbers.

        Returns:
        A float number.

        Examples:
        print(v2([1],[2], [3], i2f))
            2.731058578630005
        print(v2([1,2], [2,3], [3,4], i2f))
            [2.731058578630005, 3.880797077977882]
    '''
    n = len(xoy)
    if n==1:
        return FUN(xoy[0], a[0], b[0])
    yox = np.zeros(n)
    for i in range(n):
        yox[i] = FUN(xoy[i], a[i], b[i])
    return yox



def m2(mat, a, b, FUN):
      ''' Function to apply transformation to a matrix.

        Args:
        mat: a matrix
        a: a vector
        b: a vector
        FUN: an already speified function that have 3 inputs
        FUN:
        an already speified function that have 3 inputs

        Returns:
        returns a matrix, the ith row of which is v2(mat[i,],a,b,FUN).
      '''
      if(len(a) == 1):
          newmat = mat
          for i in range(len(mat)):
              newmat[i] = FUN(mat[i], a[0], b[0])
          return newmat
      out = mat
      # all three are vectors
      for i in range(np.shape(mat)[0]):
          out[i, :] = v2(mat[i, :], a, b, FUN)
      return out



def f2(y, a, b, ULP):
    ''' Function to apply transformation to variables and give dditional Jacobian to prob. functions.

          Args:
            y: a vector
            a: a vector that no shorter than y
            b: a vector that no shorter than y
            ULP: an already speified function, the log unnormalized probability function of a certain distribution

          Returns:
            Returns a vector that equals to the prob. functions of the transformed variables plus an additional Jacobian.
  '''
    x = v2(y, a, b, i2f)
    return (ULP(x)+np.sum(v2(y, a, b, logJ)))

def mult(n, w):
    '''Multinomial resampling

        Args:
        w: numeric non-negative vector of length K, specifying the probability for the K classes,
        is internally normalized to sum 1.

        n: integer equals to K, specifying the total number of objects that are put into K boxes
        in the typical  multinomial experiment

        Returns:
        Returns a vector, which is the result of multinomial resampling.

        Examples:
        print(mult(10, [1/6.]*6))
            [1 1 2 2 3 4 5 5 6 6]
        print(mult(20, [1/4, 1/4, 1/4, 1/4]))
            [1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 4 4 4 4]
    '''
    a = np.arange(1, len(w)+1)  # Here should be len(w)+1 rather than n+1.
    b = np.random.multinomial(n, w, size = 1)
    return np.repeat(a, b[0])   # b[0] is to change an array to a list.

'''
    Bayesian Stopping Rule for Continuous Data with Unknown Pre and Post
    Changepoint detection for continuous data with unknown pre-change and post-change
    distributions using the Bayesian stopping rule.

    GEN: A function of time that returns an observation.

    alpha: A numeric parameter in (0,1) that controls the probability
    of false alarm.

    nulower,nuupper: Optional nonnegative numerics: The earliest and latest
    time of changepoint based on prior belief. The default is nulower=0
    and nuupper=18 which corresponds to the geometric prior distribution
    with p=0.1

    score: An optional character specifying the type of score to be used:
    The default "hyvarinen" or the conventional "logarithmic".
    Can be abbreviated. Case insensitive.

    c,n: Optional parameters of the Sequentital Monte Carlo algorithm: ESS
    threshold c(0<c<1) and sample size n(n>0). Default is c=0.5 and n=1000.
    The resample-move step is triggered whenever the ESS goes below c*n.

    lenth1: A positive numeric: The length of the variable of the unknown
    parameter in the post-change model.

    lenth0: A positive numeric: The length of the variable of the unknown
    parameter in the pre-change model.

    thlower1,thupper1 Optional numeric vectors of length lenth1: The
    lower and upper limits of the unknown parameter in the post-change model.
    The defaults are infinite.

    thlower0,thupper0: Optional numeric vectors of length lenth0: The lower and upper
    limits of the unknown parameter in the pre-change model.
    The defaults are infinite(negative infinite to positive finite).

    GENTH1: An optional function that takes a sample size and returns a
    random sample from the prior distribution of the unknown parameter in the
    post-change model. Default is standard normal on the unconstrained
    space. Required if ULPTH1 is specified.

    ULPTH1: An optional function: The log unnormalized probability function
    of the prior distribution of the unknown parameter in the post-change
    model. Default is standard normal on the unconstrained space. Required
    if GENTH1 is specified.

    GENTH0 An optional function that takes a sample size and returns a
    random sample from the prior distribution of the unknown parameter in the
    pre-change model. Default is standard normal on the unconstrained
    space. Required if ULPTH0 is specified.

    ULPTH0 An optional function: The log unnormalized probability function
    of the prior distribution of the unknown parameter in the pre-change
    model. Default is standard normal on the unconstrained space. Required
    if GENTH0 is specified.

    ULP0,GLP0,LLP0,ULP1,GLP1,LLP1: Functions of an observation: The log unnormalized probability
    function, its gradient and its laplacian for the pre-change (ULP0,GLP0,LLP0) and post-change
    (ULP1,GLP1,LLP1) models. If score="hyvarinen", GLP0,LLP0,GLP1,LLP1,ULP0,ULP1 is required.
    If score="logarithmic", only ULP0,ULP1 is required.

    par0,par1: Numeric parameters for the pre-change
    par0 and post-change par1 models, except if score="logarithmic"
    that par1 is a function of the unknown parameter in the post-change model.
    If score="hyvarinen", the positive tuning parameter with a default of 1.
    If score="logarithmic", the negative log normalizing constant.

    lower0,upper0,lower1,upper1 Optional numeric vectors of length lenx:
    The lower and upper limits of an observation from the pre-change (lower0,upper0)
    and post-change (lower1,upper1) models. The defaults are infinite.

    return: A named numeric vector with components
    t: A positive numeric: The stopping time.
    LER: A numeric in (0,1): The low ESS rate, i.e., the proportion of iterations that ESS drops below c*n.
    AAR: A numeric in (0,1): The average acceptance rate of the Metropolis-Hastings sampling in the move step.
    NaN if ESS never drops below c*n.

    # Examples
    # Change from N(0,1) to 2*N(0,1)+1 at t=15.
    # Prior knowledge suggests change occurs between 10 and 25.
    # The mean and standard deviation of both the pre-change and the post-change normal distribution are unknown.

    # def GEN(t):
    #     if(t<=15):
    #         return np.random.normal(0,1,1)[0]
    #     else:
    #         return 2*np.random.normal(0,1,1)[0]+1
    # ULP1=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
    # GLP1=lambda x, th: -(x-th[0])/(th[1]**2)
    # LLP1=lambda x, th: -1/(th[1]**2)
    # ULP0=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
    # GLP0=lambda x, th: -(x-th[0])/(th[1]**2)
    # LLP0=lambda x, th: -1/(th[1]**2)
    # par0=lambda th: log(2*pi)/2+log(th[1])
    # par1=lambda th: log(2*pi)/2+log(th[1])
    #
    # detect_bayes_unknown_pre(GEN=GEN, alpha=0.1, nulower=5, nuupper=30, lenth1=2, lenth0=2, thlower1=[np.NINF,0],thlower0=[np.NINF,0],
    #                          score="hyvarinen",GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1)
    #
    # detect_bayes_unknown_pre(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,lenth1=2, lenth0=2, thlower1=[np.NINF,0],thlower0=[np.NINF,0],
    #                 score="logarithmic",ULP0=ULP0,ULP1=ULP1,par1=par1,par0=par0,lenx=1)
'''

def detect_bayes_unknown_pre(GEN, alpha, lenth0, lenth1, nulower=None,nuupper=None, score= "hyvarinen", c=0.5, n=1000,
                 thlower1=None,thupper1=None,
                 thlower0=None,thupper0=None,
                 GENTH1=None,ULPTH1=None,GENTH0=None,ULPTH0=None,
                 ULP0=None,GLP0=None,LLP0=None,ULP1=None,GLP1=None,LLP1=None,
                 par0=None,par1=None,
                 lenx=None,lower0=None,upper0=None,lower1=None,upper1=None):
    # check the false alarm rate
    if alpha <=0 or alpha >=1:
        raise Exception("'alpha' should be in (0, 1)")

    # choose score
    score = score.lower()
    if re.match(score, "hyvarinen"):
        score = "hyvarinen"
    if re.match(score, "logarithmic"):
        score = "logarithmic"

    # suppose the time of changepoint nu follows a geometric prior distribution, compute the parameter p
    # the default is nulower=0, nuupper=18, which corresponds to p=0.1
    if nulower is None:
        nulower = 0
    if nuupper is None:
        p = 0.1
    else:
        if(nulower >= nuupper):
            raise Exception("need 'nulower' < 'nuupper'")
        p = 1/(1 + (nulower + nuupper)/2)

    # suppose the unknown parameters in the post-change model th = c(th[1], th[2], ...) follows some known prior distribution
    # if not specified, the defaults of the lower and upper limits of th are -Inf and Inf
    if thlower1 is None:
        thlower1 = np.repeat([np.NINF], lenth1)
    if thupper1 is None:
        thupper1 = np.repeat([np.inf], lenth1)

    if thlower0 is None:
        thlower0 = np.repeat([np.NINF], lenth0)
    if thupper0 is None:
        thupper0 = np.repeat([np.inf], lenth0)

  # create a matrix nuth that combines 3 parts by column:
  # a random sample of the time of changepoint nu, a vector of length n
  # a random sample of the unknown parameters in the post-change model th1, a matrix with n rows and lenth1 columns
  # a random sample of the unknown parameters in the pre-change model th0, a matrix with n rows and lenth0 columns
    if GENTH1 is None and ULPTH1 is None:
        ULPTH1 = lambda th1: -1 * (np.sum([elem**2 for elem in th1])) /2 ###default prior N(0,1)
        GENTH1 = lambda n: np.random.normal(0, 1, size=(n, lenth1))
        # t=0 sample
        # Written geometric as size = (n, 1) to stack them easily. nulower should be a constant. Just add them up.
        geo = np.random.geometric(p, size=n)
        geo_nulower = [nulower] + geo
        nuth1 = np.column_stack((geo_nulower, GENTH1(n)))
    else:
        # GENTH and ULPTH should always exist or be missing at the same time
        # Can't be one existing while the other missing
        if GENTH1 is None:
            raise Exception("'GENTH' is missing")
        if ULPTH1 is None:
            raise Exception("'ULPTH' is missing")
        fULPTH1 = ULPTH1
        # apply transformation f2 to ULPTH, which adds an additional Jacobian to ULPTH
        ULPTH1 = lambda th1: f2(th1, thlower1, thupper1, fULPTH1)
        geo =  np.random.geometric(p, size=n)
        # when building nuth, apply transformation m2 to GENTH(n), which makes finite to infinite transformations to GENTH(n)
        geo_nulower = [nulower] + geo
        nuth1=np.column_stack((geo_nulower, m2(GENTH1(n), thlower1, thupper1,f2i)))

    if GENTH0 is None and ULPTH0 is None:
        ULPTH0 = lambda th0: -1 * (np.sum([elem**2 for elem in th0])) /2 ###default prior N(0,1)
        GENTH0 = lambda n: np.random.normal(0, 1, size=(n, lenth0))
        nuth=np.column_stack((nuth1, GENTH0(n)))
    else:
        # GENTH and ULPTH should always exist or be missing at the same time
        # Can't be one existing while the other missing
        if GENTH0 is None:
            raise Exception("'GENTH' is missing")
        if ULPTH0 is None:
            raise Exception("'ULPTH' is missing")
        fULPTH0 = ULPTH0
        # apply transformation f2 to ULPTH, which adds an additional Jacobian to ULPTH
        ULPTH0 = lambda th0: f2(th0, thlower0, thupper0, fULPTH0)
        nuth=np.column_stack((nuth1, m2(GENTH0(n), thlower0, thupper0,f2i)))

    def LPQ(nuth, numean, thmean1, thsd1, thmean0, thsd0):
        # sample nu from its prior distribution
        nu=nuth[0]
        # sample th1 and th0 from their prior distributions
        th1=nuth[1:lenth1+1]; th0=nuth[len(nuth)-lenth0:len(nuth)]
        for i in range(len(thsd1)):
            if thsd1[i] == 0:
                thsd1[i]=1
        for i in range(len(thsd0)):
            if thsd0[i] == 0:
                thsd0[i]=1
        # print(thsd1, thsd0)
        return ((nu - nulower)*(log(1-p)-log(1-1/(1+numean)))+np.sum((th1-thmean1)**2/(thsd1**2)) + ULPTH1(th1)+np.sum((th0-thmean0)**2/(thsd0**2))+ULPTH0(th0))
        # when choose hyvarinen score
    if score == "hyvarinen":
        # some checks, preparations, and transformations of the log probablity functions
        # functions for pre-change:
        if GLP0 is None:
            raise Exception("Need gradient for the pre log unnormalized probability function")
        else:
            fGLP0=GLP0
            GLP0=lambda x,th: fGLP0(x, v2(th,thlower0,thupper0,i2f))
        if LLP0 is None:
            raise Exception("Need laplacian for the pre log unnormalized probability function")
        else:
            fLLP0=LLP0
            LLP0=lambda x,th: fLLP0(x, v2(th,thlower0,thupper0,i2f))
        # functions for post-change:
        if GLP1 is None:
            raise Exception("Need gradient for the post log unnormalized probability function")
        else:
            fGLP1 = GLP1
            GLP1 = lambda x, th: fGLP1(x, v2(th, thlower1, thupper1, i2f))
        if LLP1 is None:
            raise Exception("Need laplacian for the post log unnormalized probability function")
        else:
            fLLP1 = LLP1
            LLP1 = lambda x, th: fLLP1(x, v2(th, thlower1, thupper1, i2f))
        # check the optional parameters for the pre and post change models
        if par0 is None:
            par0 = 1
        elif par0<= 0:
            warnings.warn("'par0' should be positive for hyvarinen score. Use par0 = 1")
            par0 = 1
        if par1 is None:
            par1 = 1
        elif par1 <= 0:
            warnings.warn("'par1' should be positive for hyvarinen score. Use par1 = 1")
            par1 = 1
    # compute the scores for the pre and post change models
        SC0 = lambda x,th: par0 * (np.sum(np.array(GLP0(x,th))**2))/2 + LLP0(x,th)
        def hSC0(x, th):
            if lenth0 == 1:
                newth0 = th
                for i in range(np.shape(th)[0]):
                    newth0[i] = SC0(x, th[i])
                return newth0
            else:
                newth0 = np.zeros(np.shape(th)[0])
                for i in range(np.shape(th)[0]):
                    newth0[i] = SC0(x, th[i,:])
                return newth0

        SC1 = lambda x, th: par1 * (np.sum(np.array((GLP1(x, th) **2)))/2 + LLP1(x, th))
        def hSC1(x, th):
            if lenth1 == 1:
                newth1 = th
                for i in range(np.size(th)):
                    newth1[i] = SC1(x, th[i])
                return newth1
            else:
                newth1 = np.zeros(np.shape(th)[0])
                for i in range(np.shape(th)[0]):
                    newth1[i] = SC1(x, th[i,:])
                return newth1
    # compute the log of the difference of the pre-change and post-change scores
        def hLZSUM(nuth, t, x):
            out = np.zeros(n)
            nu = nuth[:, 0]
            nu = [int(elem) for elem in nu]
            id = []
            for i in range(len(nu)):
                if nu[i]<t:
                    id.append(i)
            for i in id:
                xtail = x
                if nu[i] >0:
                    xtail=x[nu[i]:]
                a = xtail
                b = xtail
                for j in range(np.size(xtail)):
                    a[j] = SC0(xtail[j], nuth[i,np.shape(nuth)[1]-lenth0:np.shape(nuth)[1]])
                    b[j] = SC1(xtail[j], nuth[i,1:(1+lenth1)])
                out[i] = np.sum(a) - np.sum(b)
            return out

    else:
        # when choose logarithmic score
        fULP0 = ULP0
        ULP0 = lambda x, th: fULP0(x, v2(th, thlower0, thupper0, i2f))
        fULP1 = ULP1
        ULP1 = lambda x, th: fULP1(x, v2(th, thlower1, thupper1, i2f))
        if par0 is None or par1 is None:
            if lenx is None:
                raise Exception("'lenx' is missing")
            if lenx <= 0:
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
        if par1 is None:
            raise Exception("Need par1 for log score.")
            # fn = lambda x, th: exp(ULP0(x, th))
            # if lenx == 1:
            #     par1=lambda th: log(integrate.nquad(fn, [lower1[0],upper1[0]],args=(th))[0])
            # else:
            #     rgs=[]
            #     for i in range(lenx):
            #         rgs.append([lower1[i], upper1[i]])
            #     par1 =lambda th: log(integrate.nquad(fn, rgs, args=th)[0])
        else:
            fpar1 = par1
            par1 = lambda th: fpar1(v2(th, thlower1, thupper1, i2f))

        if par0 is None:
            raise Exception("Need par0 for log score.")
            # fn = lambda x:exp(ULP0(x))
            # if lenx == 1:
            #     par0 = log(integrate.quad(fn, lower0[0], upper0[0])[0])
            # else:
            #     rgs=[]
            #     for i in range(lenx):
            #         rgs.append([lower0[i], upper0[i]])
            #     par0 = log(integrate.nquad(fn, rgs)[0])
        else:
            fpar0 = par0
            par0 = lambda th: fpar0(v2(th, thlower0, thupper0, i2f))

        if lenth1 ==1:
            lpar1 = np.zeros(n)
            for i in range(n):
                lpar1[i] = par1(nuth[i, 1:(1+lenth1)])
        else:
            lpar1 = np.zeros(n)
            for i in range(n):
                lpar1[i]=par1(nuth[i, 1:(1+lenth1)])

        if lenth0 ==1:
            lpar0 = np.zeros(n)
            for i in range(n):
                lpar0[i] = par0(nuth[i, np.shape(nuth)[1]-lenth0:np.shape(nuth)[1]])
        else:
            lpar0 = np.zeros(n)
            for i in range(n):
                lpar0[i]=par0(nuth[i, np.shape(nuth)[1]-lenth0:np.shape(nuth)[1]])

        SC0 = lambda x, th0, par0th: -ULP0(x, th0) + par0th
        SC1 = lambda x, th1, par1th: -ULP1(x, th1) + par1th

        def lSC1(x, th, lpar1):
            if len(lpar1) == 1:
                out=SC1(x, th, lpar1)
            else:
                out=np.zeros(len(lpar1))
                if lenth1 == 1:
                    for i in range(len(lpar1)):
                        out[i] = SC1(x, th[i], lpar1[i])
                else:
                    for i in range(len(lpar1)):
                        out[i] = SC1(x, th[i,:], lpar1[i])
            return out

        def lSC0(x, th, lpar0):
            if len(lpar0) == 1:
                out=SC0(x, th, lpar0)
            else:
                out=np.zeros(len(lpar0))
                if lenth0 == 1:
                    for i in range(len(lpar0)):
                        out[i] = SC0(x, th[i], lpar0[i])
                else:
                    for i in range(len(lpar0)):
                        out[i] = SC0(x, th[i,:], lpar0[i])
            return out

        def lLZSUM(nuth, t, x, lpar1,lpar0):
            out = np.zeros(n)
            nu = nuth[:,0]
            nu = [int(elem) for elem in nu]
            id = []
            for i in range(n):
                if nu[i] < t:
                    id.append(i)
            for i in id:
                xtail = x
                if nu[i] >0:
                    xtail = x[nu[i]:]
                a = xtail
                b = xtail
                for j in range(np.size(xtail)):
                    a[j] = SC0(xtail[j], nuth[i, np.shape(nuth)[1]-lenth0:np.shape(nuth)[1]],lpar0[i])
                for j in range(np.size(xtail)):
                    b[j] = SC1(xtail[j], nuth[i, 1:(1+lenth1)] , lpar1[i])
                out[i] = np.sum(a) - np.sum(b)
            return out
    # At step t=0:
    t=0
    # initialize some useful variables
    # w: weights
    w = np.repeat([1/n], n)
    # lw: the log of weights
    lw = np.repeat([-log(n)], n)
    # lrat: the log hastings ratio
    lrat = np.zeros(n)
    # ngsc: the negative scores
    ngsc = np.zeros(n)
    # lowESS: low Effective Sample Size
    lowESS = 0
    # x: an observation sampled from the pre-change mode
    # ar: acceptance rate
    x = []; ar = []
      #length.unique.id = 1000
      # i.acc=1
      # At step t = 2, 3, ...:

    while True:
        # if ESS = 1/sum(w^2) < c*n, do the resample and move step and draw (nu,th) for time = t
        if (1/(np.sum(w**2))) < c*n:
            lowESS =  lowESS+1
            # resample step:
            # use a multinomial distribution to obtain equally weighted samples
            id = mult(n, w) -[1]
            # print(id)
            nuth = nuth[id,: ]

            if score == "logarithmic":
                lpar1 = lpar1[id]
                lpar0 = lpar0[id]
            # set the weights to be equal
            w = np.repeat([1/n], n)
            lw = np.repeat([-log(n)], n)
            # move step:
            # consider an MCMC kernel targeting the joint posterior of (nu,th) conditional on the t-1 observations
            # compute the mean of randomly sampled nu
            numean = np.mean(nuth[:,0])
            # draw a new nu from the MCMC kernel
            nu_p = nulower + np.random.geometric(1/(1+(numean-nulower)), n)
            if lenth1 ==1:
                thmean1 = np.mean(nuth[:, 1:(1+lenth1)])
                thsd1 = np.std(nuth[:, 1:(1+lenth1)])
                th_p1 = np.random.normal(0, 1, n)
                th_p1 = [elem*thsd1+thmean1 for elem in th_p1]
            else:
                a = nuth[:, 1:(1+lenth1)]
                thmean1 = np.zeros(lenth1)
                thsd1 = np.zeros(lenth1)
                for i in range(lenth1):
                    thmean1[i] = np.mean(a[:,i])
                    thsd1[i] = np.std(a[:,i])
                th_p1 = np.random.normal(0, 1, n*lenth1)
                th_p1 = th_p1 *np.tile(thsd1,n)+np.tile(thmean1, n)
                th_p1 = th_p1.reshape(n, lenth1)

            if lenth0 ==1:
                thmean0 = np.mean(nuth[:, np.shape(nuth)[1]-lenth0 :np.shape(nuth)[1]])
                thsd0 = np.std(nuth[:, np.shape(nuth)[1]-lenth0 :np.shape(nuth)[1]])
                th_p0 = np.random.normal(0, 1, n)
                th_p0 = [elem*thsd0+thmean0 for elem in th_p0]
            else:
                b = nuth[:, np.shape(nuth)[1]-lenth0 :np.shape(nuth)[1]]
                thmean0 = np.zeros(lenth0)
                thsd0 = np.zeros(lenth0)
                for i in range(lenth0):
                    thmean0[i] = np.mean(b[:,i])
                    thsd0[i] = np.std(b[:,i])
                th_p0 = np.random.normal(0, 1, n*lenth0)
                th_p0 = th_p0 *np.tile(thsd0,n)+np.tile(thmean0, n)
                th_p0 = th_p0.reshape(n, lenth0)
            # combine the new nu and th1, th0 together
            nuth_p=np.column_stack((nu_p,th_p1,th_p0))
            # compute the log Hastings ratio
            if score =="logarithmic":
                if lenth1 ==1:
                    lpar1_p = np.zeros(n)
                    for i in range(n):
                        lpar1_p[i] = par1(th_p1[i])
                else:
                    lpar1_p = np.zeros(n)
                    for i in range(n):
                        lpar1_p[i] = par1(th_p1[i,: ])

                if lenth0 ==1:
                    lpar0_p = np.zeros(n)
                    for i in range(n):
                        lpar0_p[i] = par0(th_p0[i])
                else:
                    lpar0_p = np.zeros(n)
                    for i in range(n):
                        lpar0_p[i] = par0(th_p0[i,: ])
            lrat = np.zeros(n)
            for i in range(n):
                lrat[i]=LPQ(nuth_p[i,:], numean, thmean1, thsd1, thmean0, thsd0)-LPQ(nuth[i,:],numean, thmean1, thsd1, thmean0, thsd0)
            if score == "hyvarinen":
                lrat = lrat + hLZSUM(nuth_p,t,x)-hLZSUM(nuth,t,x)
            else:
                lrat = lrat + lLZSUM(nuth_p,t,x,lpar1_p,lpar0_p)-lLZSUM(nuth,t,x,lpar1,lpar0)
                # accept the new (nu,th1,th0) if the accept ratio smaller than a random sample from Unif(0,1)
            uni = np.random.uniform(0, 1, n)
            uni = [log(elem) for elem in uni]
            # accept the new (nu,th) if the Hastings ratio smaller than a random sample from Unif(0,1)
            acc = (lrat >= uni)
            if np.sum(acc) >0:
                nuth[acc] = nuth_p[acc]
                if score == "logarithmic":
                    lpar1[acc] = lpar1_p[acc]
                    lpar0[acc] = lpar0_p[acc]
            ar.append(np.mean(acc))
        t = t+1
        xt = GEN(t)
        if xt is None:
            raise Exception("Did not detect any change.")

        x.append(xt)
        xg = (nuth[:, 0] <t) # boolen array now
        if np.sum(~xg) >0:
            if score == "hyvarinen":
                ngsc[~xg] = -hSC0(xt, nuth[~xg, np.shape(nuth)[1]-lenth0 :np.shape(nuth)[1]])
            else:
                ngsc[~xg] = -lSC0(xt, nuth[~xg, np.shape(nuth)[1]-lenth0 :np.shape(nuth)[1]], lpar0[~xg])

        if np.sum(xg) >0:
            if score == "hyvarinen":
                ngsc[xg] = -hSC1(xt, nuth[xg,1:(1+lenth1)])
            else:
                ngsc[xg] = -lSC1(xt, nuth[xg,1:(1+lenth1)],lpar1[xg])
        lw_ngsc = lw+ngsc
        mx =np.max(lw_ngsc)
        diff = lw_ngsc-[mx]
        # calclute the new weights w
        exp1 = [exp(elem) for elem in diff]
        lsm=log(np.sum(exp1))
        lw=[elem-lsm for elem in diff]
        w=[exp(elem) for elem in lw]
        w=np.array(w)
        # now the new (w, nu, th) approximate the joint posterior of (nu, th) conditional on the t observations
        sum_w = np.sum(w[xg])
        if t> nulower and sum_w  >= 1-alpha:
            break
    if ar == []:
        out = {}
        out["t"]=t
        out["LER"]=round(lowESS/t, 4)
        out["AAR"]="NAN"
        print(out)
        return out
        # print("t:", t, "     LER:", round(lowESS/t, 4), "     AAR: NAN", )
    else:
        out = {}
        out["t"]=t
        out["LER"]=round(lowESS/t, 4)
        out["AAR"]=round(np.mean(ar), 4)
        print(out)
        return out



'''
Simulation using Bayesian Stopping Rule with unknown post-change distributions using the Bayesian stopping rule.

GEN0: A function that takes a value of the unknown parameter in the pre-change model and return an observation.

GEN1: A function that takes a value of the unknown parameter in the post-change model and return an observation.

th0 Numeric: True value of the unknown parameter in the pre-change model.

th1 Numeric: True value of the unknown parameter in the post-change model.

detect_bayes_unknown_pre: A function that implements the Bayesian stopping rule.

nulower,nuupper Optional nonnegative numerics: The earliest and latest time of changepoint based on prior belief. The default is nulower=0
    and nuupper=18 which corresponds to the geometric prior distribution with p=0.1.

return A named numeric vector with components
is.FA: A numeric of 1 or 0 indicating whether a false alarm has been raised.
DD: A positive numeric: The delay to detection.
CT: A positive numeric: The computation time.
LER: A numeric in (0,1): The low ESS rate, i.e., the proportion of iterations that ESS drops below c*n.
AAR: A numeric in (0,1): The average acceptance rate of the Metropolis-Hastings sampling in the move step. NaN if ESS never drops below c*n.

# Examples:

# Change from N(0,1) to 2*N(0,1)+1 occurs between 10 and 25.
# The mean and standard deviation of the post-change normal distribution are unknown.
# GEN1=lambda th: th[1]*np.random.normal(0,1,1)+th[0]
# GEN0=lambda th: th[1]*np.random.normal(0,1,1)+th[0]
# ULP1=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
# GLP1=lambda x, th: -(x-th[0])/(th[1]**2)
# LLP1=lambda x, th: -1/(th[1]**2)
# ULP0=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
# GLP0=lambda x, th: -(x-th[0])/(th[1]**2)
# LLP0=lambda x, th: -1/(th[1]**2)
# par0=lambda th: log(2*pi)/2+log(th[1])
# par1=lambda th: log(2*pi)/2+log(th[1])
#
# sim_detect_bayes_unknown_pre(GEN0, GEN1, th0=[0,1], th1=[1,2], detect_bayes_unknown_pre=detect_bayes_unknown_pre, lenth0=2, lenth1=2, alpha=0.3,nulower=5, nuupper=30,
#     score= "hyvarinen",thlower1=[np.NINF,0],thlower0=[np.NINF,0],GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1,par0=0.1,par1=0.1)
#
# sim_detect_bayes_unknown_pre(GEN0,GEN1,th0=[0,1],th1=[1,2],detect_bayes_unknown_pre=detect_bayes_unknown_pre,lenth1=2,lenth0=2,alpha=0.1,nulower=20,
#             nuupper=50,thlower1=[np.NINF,0],thlower0=[np.NINF,0],score="log",lenx=1,ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)
'''
def sim_detect_bayes_unknown_pre(GEN0, GEN1, th0, th1, detect_bayes_unknown_pre, lenth0, lenth1, alpha,
                                 nulower=None, nuupper=None,
                                 score= "hyvarinen", c=0.5, n=1000,
                                 thlower1=None, thupper1=None,
                                 thlower0=None, thupper0=None,
                                 GENTH1=None,ULPTH1=None,GENTH0=None,ULPTH0=None,
                                 ULP0=None,GLP0=None,LLP0=None,ULP1=None,GLP1=None,LLP1=None,
                                 par0=None,par1=None,
                                 lenx=None,lower0=None,upper0=None,lower1=None,upper1=None):
    if nulower is None:
        nulower = 0
    if nuupper is None:
        p = 0.1
    else:
        if nulower >= nuupper:
            raise Exception("need 'nulower' < 'nuupper'")
        p = 1/(1+(nulower+nuupper)/2)
    nu = np.random.geometric(p, 1) + nulower
    def GEN(t):
        if nu >= t:
            return GEN0(th0)
        else:
            return GEN1(th1)
    CT0 = process_time()

    #
    # def detect_bayes_unknown_pre(GEN, alpha, lenth0, lenth1, nulower=None,nuupper=None, score= "hyvarinen", c=0.5, n=1000,
    #                  thlower1=None,thupper1=None,
    #                  thlower0=None,thupper0=None,
    #                  GENTH1=None,ULPTH1=None,GENTH0=None,ULPTH0=None,
    #                  ULP0=None,GLP0=None,LLP0=None,ULP1=None,GLP1=None,LLP1=None,
    #                  par0=None,par1=None,
    #                  lenx=None,lower0=None,upper0=None,lower1=None,upper1=None):
    db = detect_bayes_unknown_pre(GEN=GEN,alpha=alpha,lenth0=lenth0,lenth1=lenth1,nulower=nulower, nuupper=nuupper, score=score, c=c, n=n,
                      thlower1=thlower1, thupper1=thupper1,
                      thlower0=thlower0, thupper0=thupper0,
                      GENTH1=GENTH1, ULPTH1=ULPTH1, GENTH0=GENTH0, ULPTH0=ULPTH0,
                      ULP0=ULP0, GLP0=GLP0, LLP0=LLP0, ULP1=ULP1, GLP1=GLP1, LLP1=LLP1,
                      par0=par0, par1=par1,
                      lenx=lenx, lower0=lower0, upper0=upper0, lower1=lower1, upper1=upper1)
    CT1 = process_time()
    CT = CT1 - CT0
    fa = -1
    if nu>=db["t"]:
        fa = 1
    else:
        fa = 0
    print("is.FA:", fa, "   DD:", max((db["t"]-nu)[0],0),   "   CT:", round(CT,4), "   LER:", db["LER"], "   AAR:", db["AAR"])
    out = {}
    out["is.FA"]=fa
    out["DD"]=max((db["t"]-nu)[0],0)
    out["CT"]=round(CT,4)
    out["LER"]=db["LER"]
    out["AAR"]=db["AAR"]
    return out

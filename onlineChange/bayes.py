import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
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
    Bayesian Stopping Rule for Continuous Data.
    Changepoint detection for continuous data with unknown post-change
    distributions using the Bayesian stopping rule.

    GEN: A function of time that returns an observation.

    alpha: A numeric parameter in (0,1) that controls the probability of false alarm.

    nulower,nuupper: Optional nonnegative numerics, the earliest and latest time of changepoint
    based on prior belief. The default is nulower=0 and nuupper=18 which corresponds to the
    geometric prior distribution with p=0.1.

    score: An optional character specifying the type of score to be used:
    The default "hyvarinen" or the conventional "logarithmic".
    Can be abbreviated. Case insensitive.

    c,n: Optional parameters of the Sequentital Monte Carlo algorithm: ESS
    threshold c(0<c<1) and sample size n(n>0). Default is c=0.5 and n=1000.
    The resample-move step is triggered whenever the ESS goes below c*n.

    lenth: A positive numeric: The length of the variable of the unknown
    parameter in the post-change model.

    thlower,thupper: Optional numeric vectors of length lenth: The lower and upper
    limits of the unknown parameter in the post-change model.
    The defaults are infinite(negative infinite to positive finite).

    GENTH: An optional function that takes a sample size and returns a random sample from the
    prior distribution of the unknown parameter in the post-change model. Default is standard normal
    on the unconstrained space. Required if ULPTH is specified.

    ULPTH: An optional function: The log unnormalized probability function of the prior distribution
    of the unknown parameter in the post-change model. Default is standard normal on the unconstrained
    space. Required if GENTH is specified.

    ULP0,GLP0,LLP0,ULP1,GLP1,LLP1: Functions of an observation: The log unnormalized probability
    function, its gradient and its laplacian for the pre-change (ULP0,GLP0,LLP0) and post-change
    (ULP1,GLP1,LLP1) models. If score="hyvarinen", GLP0,LLP0,GLP1,LLP1,ULP0,ULP1 is required.
    If score="logarithmic", only ULP0,ULP1 is required.

    par0,par1 Optional numeric parameters for the pre-change (par0) and post-change (par1) models.

    lenx A positive numeric: The length of the variable of an obervation. Optional if score="hyvarinen"
    or if score="logarithmic" and par0,par1 are specified.

    lower0,upper0,lower1,upper1 Optional numeric vectors of length lenx:
    The lower and upper limits of an observation from the pre-change (lower0,upper0)
    and post-change (lower1,upper1) models. The defaults are infinite.

    return: A named numeric vector with components
    t: A positive numeric: The stopping time.
    LER: A numeric in (0,1): The low ESS rate, i.e., the proportion of iterations that ESS drops below c*n.
    AAR: A numeric in (0,1): The average acceptance rate of the Metropolis-Hastings sampling in the move step.
    NaN if ESS never drops below c*n.

'''
def detect_bayes(GEN, alpha, lenth, score= "hyvarinen", c=0.5, n=1000,
                 nulower=None,nuupper=None,
                 thlower=None,thupper=None,
                 GENTH=None,ULPTH=None,
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
    if thlower is None:
        thlower = np.repeat([np.NINF], lenth)
    if thupper is None:
        thupper = np.repeat([np.inf], lenth)

    # if not specified, the default prior distribution of th is standard normal N(0,1) on the unconstrained space
    if GENTH is None and ULPTH is None:
        ULPTH = lambda th: -1 * (np.sum([elem**2 for elem in th])) /2 ###default prior N(0,1)
        GENTH = lambda n: np.random.normal(0, 1, size=(n, lenth))
        # t=0 sample
        # Written geometric as size = (n, 1) to stack them easily. nulower should be a constant. Just add them up.
        geo = np.random.geometric(p, size=n)
        geo_nulower = [nulower] + geo
        nuth = np.column_stack((geo_nulower, GENTH(n)))
    else:
        # GENTH and ULPTH should always exist or be  missing at the same time
        # Can't be one existing while the other missing
        if GENTH is None:
            raise Exception("'GENTH' is missing")
        if ULPTH is None:
            raise Exception("'ULPTH' is missing")
        fULPTH = ULPTH
        # apply transformation f2 to ULPTH, which adds an additional Jacobian to ULPTH
        ULPTH = lambda th: f2(th, thlower, thupper, fULPTH)
        geo =  np.random.geometric(p, size=n)
        # when building nuth, apply transformation m2 to GENTH(n), which makes finite to infinite transformations to GENTH(n)
        geo_nulower = [nulower] + geo
        nuth=np.column_stack((geo_nulower, m2(GENTH(n), thlower, thupper,f2i)))

    # compute the log of Metropols-Hastings kernal with independent proposal from Geom(1/(1+numean)) * N(thmean, thsd^2)
    # the input nuth here is a vector of length lenth+1
    def LPQ(nuth,numean,thmean,thsd):
        nu=nuth[0]; th=nuth[1:]
        return ((nu - nulower)*(log(1-p)-log(1-1/(1+numean)))+ULPTH(th)+np.sum([(elem-thmean)**2 /(thsd**2) for elem in th]) /2)

    # when choose hyvarinen score
    if score == "hyvarinen":
        # some checks, preparations, and transformations of the log probablity functions
        # functions for pre-change:
        if GLP0 is None:
            raise Exception("Need gradient for the pre log unnormalized probability function")
        if LLP0 is None:
            raise Exception("Need laplacian for the pre log unnormalized probability function")
        # functions for post-change:
        if GLP1 is None:
            raise Exception("Need gradient for the post log unnormalized probability function")
        else:
            fGLP1 = GLP1
            GLP1 = lambda x, th: fGLP1(x, v2(th, thlower, thupper, i2f))
        if LLP1 is None:
            raise Exception("Need laplacian for the post log unnormalized probability function")
        else:
            fLLP1 = LLP1
            LLP1 = lambda x, th: fLLP1(x, v2(th, thlower, thupper, i2f))
        # check the optional parameters for the pre and post change models
        if par0 is None:
            par0 = 0.01
        elif par0<= 0:
            warnings.warn("'par0' should be positive for hyvarinen score. Use par0 = 1")
            par0 = 0.01
        if par1 is None:
            par1 = 0.01
        elif par1 <= 0:
            warnings.warn("'par1' should be positive for hyvarinen score. Use par1 = 1")
            par1 = 0.01

        # compute the scores for the pre and post change models
        SC0 = lambda x: par0 * (np.sum(np.array(GLP0(x)**2))/2 + LLP0(x))
        SC1 = lambda x, th: par1 * (np.sum(np.array((GLP1(x, th) **2)))/2 + LLP1(x, th))
        def hSC1(x, th):
            if lenth == 1:
                newth1 = th
                for i in range(np.size(th)):
                    newth1[i] = SC1(x, th[i])
                return newth1
            else:
                newth2 = np.zeros(np.shape(th)[0])
                for i in range(np.shape(th)[0]):
                    newth2[i] = SC1(x, th[i,:])
                return newth2


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
                    a[j] = SC0(xtail[j])
                    b[j] = SC1(xtail[j], nuth[i,1:])
                out[i] = np.sum(a) - np.sum(b)
            return out

    else:
        # when choose logarithmic score
        fULP1 = ULP1
        ULP1 = lambda x, th: fULP1(x, v2(th, thlower, thupper, i2f))
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
        if par0 is None:
            raise Exception("Need par0.")
            # fn = lambda x:exp(ULP0(x))
            # if lenx == 1:
            #     par0 = log(integrate.quad(fn, lower0[0], upper0[0])[0])
            # else:
            #     rgs=[]
            #     for i in range(lenx):
            #         rgs.append([lower0[i], upper0[i]])
            #     par0 = log(integrate.nquad(fn, rgs)[0])
        if par1 is None:
            raise Exception("Need par1.")
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
            par1 = lambda th: fpar1(v2(th, thlower, thupper, i2f))

        if lenth ==1:
            lpar1 = np.zeros(n)
            for i in range(n):
                lpar1[i] = par1(nuth[i, 1])
        else:
            lpar1=np.zeros(n)
            for i in range(n):
                lpar1[i]=par1(nuth[i, 1:])

        SC0 = lambda x: -ULP0(x) + par0
        SC1 = lambda x, th, par1th: -ULP1(x, th)+par1th

        def lSC1(x, th, lpar1):
            if len(lpar1) == 1:
                out=SC1(x, th, lpar1)
            else:
                out=np.zeros(len(lpar1))
                if lenth == 1:
                    for i in range(len(lpar1)):
                        out[i] = SC1(x, th[i], lpar1[i])
                else:
                    for i in range(len(lpar1)):
                        out[i] = SC1(x, th[i,:], lpar1[i])
            return out

        def lLZSUM(nuth, t, x, lpar1):
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
                    a[j]    = SC0(xtail[j])
                for j in range(np.size(xtail)):
                    b[j] = SC1(x, nuth[i, 1:] , lpar1[i])
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

    # At step t = 2, 3, ...:
    while True:
        # if ESS = 1/sum(w^2) < c*n, do the resample and move step and draw (nu,th) for time = t
        if (1/(np.sum([elem**2 for elem in w]))) < c*n:
            lowESS =  lowESS+1
            # resample step:
            # use a multinomial distribution to obtain equally weighted samples
            id = mult(n, w) -[1]
            nuth = nuth[id,: ]
            if score == "logarithmic":
                lpar1 = lpar1[id]
            # set the weights to be equal
            w = np.repeat([1/n], n)
            lw = np.repeat([-log(n)], n)
            # move step:
            # consider an MCMC kernel targeting the joint posterior of (nu,th) conditional on the t-1 observations
            # compute the mean of randomly sampled nu
            numean = np.mean(nuth[:,0])
            # draw a new nu from the MCMC kernel
            nu_p = nulower + np.random.geometric(1/(1+(numean-nulower)), n)
            # draw a new th from the MCMC kernel
            if lenth ==1:
                thmean = np.mean(nuth[:, 1])
                thsd = np.std(nuth[:, 1])
                th_p = np.random.normal(0, 1, n)
                th_p = [elem*thsd+thmean for elem in th_p]
            else:
                a = nuth[:, 1:]
                thmean = np.zeros(lenth)
                thsd = np.zeros(lenth)
                for i in range(lenth):
                    thmean[i] = np.mean(a[:,i])
                    thsd[i] = np.std(a[:,i])
                th_p = np.random.normal(0, 1, n*lenth)
                th_p = th_p *np.tile(thsd,n)+np.tile(thmean, n)
                th_p = th_p.reshape(n, lenth)
            # combine the new nu and th together
            nuth_p=np.column_stack((nu_p,th_p))
            # compute the log Hastings ratio
            if score =="logarithmic":
                if lenth ==1:
                    lpar1_p = np.zeros(n)
                    for i in range(n):
                        lpar1_p[i] = par1(th_p[i])
                else:
                    lpar1_p = np.zeros(n)
                    for i in range(n):
                        lpar1_p[i] = par1(th_p[i,: ])
            lrat = np.zeros(n)
            for i in range(n):
                lrat[i]=LPQ(nuth_p[i,:],numean,thmean,thsd)- LPQ(nuth[i,:],numean,thmean,thsd)
            if score == "hyvarinen":
                lrat = lrat + hLZSUM(nuth_p,t,x)-hLZSUM(nuth,t,x)
            else:
                lrat = lrat + lLZSUM(nuth_p,t,x,lpar1_p)-lLZSUM(nuth,t,x,lpar1)
            uni = np.random.uniform(0, 1, n)
            uni = [log(elem) for elem in uni]
            # accept the new (nu,th) if the Hastings ratio smaller than a random sample from Unif(0,1)
            acc = (lrat >= uni)
            if np.sum(acc) >0:
                nuth[acc] = nuth_p[acc]
                if score == "logarithmic":
                    lpar1[acc] = lpar1_p[acc]
            ar.append(np.mean(acc))
        t = t+1
        xt = GEN(t)

        if xt is None:
            raise Exception("Did not detect any change.")

        x.append(xt)
        xg = (nuth[:, 0] <t) # boolen array now
        if np.sum(~xg) >0:
            ngsc[~xg] = -SC0(xt)

        if np.sum(xg) >0:
            if score == "hyvarinen":
                ngsc[xg] = -hSC1(xt, nuth[xg,1:])
            else:
                ngsc[xg] = -lSC1(xt, nuth[xg,1:],lpar1[xg])

        lw_ngsc = lw+ngsc
        mx =np.max(lw_ngsc)
        diff = lw_ngsc-mx
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
        return out
        # print("t:", t, "     LER:", round(lowESS/t, 4), "     AAR: NAN", )
    else:
        out = {}
        out["t"]=t
        out["LER"]=round(lowESS/t, 4)
        out["AAR"]=round(np.mean(ar), 4)
        return out
        # print("t:", t, "     LER:", round(lowESS/t, 4), "     AAR:", round(np.mean(ar), 4))

'''
Simulation using Bayesian Stopping Rule

Simulation experiment of changepoint detection with unknown post-change
distributions using the Bayesian stopping rule.

GEN0: A function that take no argument and return an observation from
the pre-change model.

GEN1: A function that takes a value of the unknown parameter in the post-change model
and return an observation.

th0 Numeric: True value of the unknown parameter in the post-change model.

detect.bayes: A function that implements the Bayesian stopping rule.
Currently only detect.bayes for continuous data is supported.

nulower,nuupper: Optional nonnegative numerics: The earliest and latest time of
changepoint based on prior belief. The default is nulower=0 and nuupper=18 which
corresponds to the geometric prior distribution with p=0.1.

return: A named numeric vector with components
is.FA: A numeric of 1 or 0 indicating whether a false alarm has been raised.
DD: A positive numeric: The delay to detection.
CT: A positive numeric: The computation time.
LER: A numeric in (0,1): The low ESS rate, i.e., the proportion of iterations that ESS drops below c*n.
AAR: A numeric in (0,1): The average acceptance rate of the Metropolis-Hastings sampling in the move step.
NaN if ESS never drops below c*n.

'''

def sim_detect_bayes(GEN0, GEN1, th0, detect_bayes, alpha, score= "hyvarinen", c=0.5, n=1000,
                     nulower=None, nuupper=None,
                     thlower=None,thupper=None,
                     GENTH=None,ULPTH=None,
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
            return GEN0()
        else:
            return GEN1(th0)
    CT0 = process_time()
    db = detect_bayes(GEN=GEN, alpha=alpha, lenth=len(th0), score=score, c=c, n=n,
                      nulower=nulower, nuupper=nuupper,
                      thlower=thlower, thupper=thupper,
                      GENTH=GENTH, ULPTH=ULPTH,
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
'''
Example1:
Change from N(0,1) to 2*N(0,1)+1 at t=15.
Prior knowledge suggests change occurs between 10 and 25.
#The mean and standard deviation of the post-change normal distribution are unknown.

def GEN(t):
    if(t<=15):
        return np.random.normal(0,1,1)[0]
    else:
        return 2*np.random.normal(0,1,1)[0]+1
ULP1=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
GLP1=lambda x, th: -(x-th[0])/(th[1]**2)
LLP1=lambda x, th: -1/(th[1]**2)
ULP0=lambda x: ULP1(x,[0,1])
GLP0=lambda x: GLP1(x,[0,1])
LLP0=lambda x: LLP1(x,[0,1])
par0=log(2*pi)/2
par1=lambda th: log(2*pi)/2+log(th[1])

#using hyvarinen score
detect_bayes(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,lenth=2,GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1)

#using log score, normalizing constant known
detect_bayes(GEN=GEN,alpha=0.1,nulower=10,nuupper=25,lenth=2,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)
'''

'''
# Example 2:
# Change from N(0,1) to 2*N(0,1)+1 occurs between 10 and 25.
# The mean and standard deviation of the post-change normal distribution are unknown.

def GEN0():
    return np.random.normal(0,1,1)[0]
def GEN1(th):
    return th[1]*np.random.normal(0,1,1)[0]+th[0]
ULP1=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
GLP1=lambda x, th: -(x-th[0])/(th[1]**2)
LLP1=lambda x, th: -1/(th[1]**2)
ULP0=lambda x: ULP1(x, [0,1])
GLP0=lambda x: GLP1(x, [0,1])
LLP0=lambda x: LLP1(x, [0,1])
par0=log(2*pi)/2
par1=lambda th: log(2*pi)/2 + log((th[1]))

#using hyvarinen score
sim_detect_bayes(GEN0=GEN0,GEN1=GEN1,th0=[1,2],detect_bayes=detect_bayes,alpha=0.1,nulower=10,nuupper=25,GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1)

#using log score, normalizing constant known
sim_detect_bayes(GEN0=GEN0,GEN1=GEN1,th0=[1,2],detect_bayes=detect_bayes,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)
'''

'''
def GEN0():
    return np.random.normal(0,1,1)[0]
def GEN1(th):
    return th[1]*np.random.normal(0,1,1)[0]+th[0]
ULP1=lambda x, th: -((x-th[0])**2)/2/(th[1]**2)
GLP1=lambda x, th: -(x-th[0])/(th[1]**2)
LLP1=lambda x, th: -1/(th[1]**2)
ULP0=lambda x: ULP1(x, [0,1])
GLP0=lambda x: GLP1(x, [0,1])
LLP0=lambda x: LLP1(x, [0,1])
par0=log(2*pi)/2
par1=lambda th: log(2*pi)/2 + log((th[1]))

# sim_detect_bayes(GEN0=GEN0,GEN1=GEN1,th0=[1,2],thlower=[np.NINF,0],detect_bayes=detect_bayes,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)

# using hyvarinen score
sum = 0
alpha=np.zeros(9)
for i in range(9):
    alpha[i]=0.1*(i+1)

hfa=np.zeros(9); hdd=np.zeros(9)
lfa=np.zeros(9); ldd=np.zeros(9)
hpt=np.zeros(9); lpt=np.zeros(9)
sumlfa=0;sumldd=0;sumlpt=0
for i in range(9):
    sumhfa=0;sumhdd=0;sumhpt=0;
    for j in range(100):
        a=sim_detect_bayes(GEN0=GEN0,GEN1=GEN1,th0=[1,2],thlower=[np.NINF,0],detect_bayes=detect_bayes,alpha=alpha[i],nulower=10,nuupper=25,GLP0=GLP0,LLP0=LLP0,GLP1=GLP1,LLP1=LLP1)
        sumhfa=sumhfa+a["is.FA"]
        sumhdd=sumhdd+a["DD"]
        if a["CT"] == 0:
            sumhpt = sumhpt
        else:
            sumhpt=sumhpt+log(a["CT"])
    hfa[i]=sumhfa/100
    hdd[i]=sumhdd/100
    hpt[i]=sumhpt/100



# sum = 0
# for i in range(50):
#     sim_detect_bayes(GEN0=GEN0,GEN1=GEN1,th0=[1,2],thlower=[np.NINF,0],detect_bayes=detect_bayes,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1,par0=par0,par1=par1)
# print(sum/50)


#using log score, normalizing constant unknown
# sum = 0
# for i in range(100):
#     sum+=sim_detect_bayes(GEN0=GEN0,GEN1=GEN1,th0=[1,2],thlower=[np.NINF,0],detect_bayes=detect_bayes,alpha=0.1,nulower=10,nuupper=25,score="log",ULP0=ULP0,ULP1=ULP1, lenx=1)
# print(sum/100)
# using log score, normalizing constant known

x=hfa
y=hdd
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, '-', label = '1')
ax.grid()
ax.set_xlabel("false alarm rate")
ax.set_ylabel(r"average delay")

plt.show()

x=hfa
y=hpt
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, '-', label = '1')
ax.grid()
ax.set_xlabel("prob of false alarm")
ax.set_ylabel(r"log computation time")

plt.show()

x=alpha
y=hfa
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, '-', label = '1')
lns2 = ax.plot(x, x, '-', label = '1')
ax.grid()
ax.set_xlabel("alpha")
ax.set_ylabel(r"prob of false alarm")

plt.show()

x=alpha
y=hdd
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(x, y, '-', label = '1')
ax.grid()
ax.set_xlabel("alpha")
ax.set_ylabel(r"average delay")

plt.show()
'''

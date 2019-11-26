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

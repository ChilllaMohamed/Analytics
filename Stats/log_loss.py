from sklearn.datasets import make_classification


X,y = make_classification(n_features=4)


from scipy.optimize import minimize


import numpy as np

def prob(X , b):
    return 1/(1+ np.exp(-1*np.sum(X*b , axis=1)))

x = X
betas = np.zeros(X.shape[1])

def logloss(betas,x,y):
    #print(betas)
    probs = prob(x,betas)
    probs = np.clip(probs , 0.000001 , .99999)
    loss = -1*np.sum(y*np.log(probs) + (1-y)*np.log(1-probs))
    return loss

def hingeLoss(betas , x,y):
    probs = prob(x,betas)
    scores = np.sum(x*betas , axis=1)
    print(scores)
    yl = np.where(y == 0 , -1 , 1)
    #labels = np.where(probs > 0.5 ,1 ,0)
    vals = np.clip(1- yl*scores , 0 , np.Inf)
    loss = np.sum(vals)
    return loss

logit = minimize(logloss ,betas , args=(X,y))
hingeLogit = minimize(hingeLoss ,betas , args=(X,y))

logit.x
hingeLogit.x


from sklearn.metrics import accuracy_score , roc_auc_score


def MeasurePerf(X,y, betas):
    predProbs = prob(X,betas)
    predLabels = np.where(predProbs > 0.5  , 1 , 0)
    ac = accuracy_score(y , predLabels)
    auc = roc_auc_score(y , predProbs)
    print(f"ac : {ac} , auc : {auc}")



MeasurePerf(X ,y, logit.x)
MeasurePerf(X ,y, hingeLogit.x)

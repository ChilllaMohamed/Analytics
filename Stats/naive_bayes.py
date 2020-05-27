import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs


X,y =make_blobs(n_samples=100, centers=2,cluster_std=5)

from plotnine import *

df = pd.DataFrame({"x1" : X[:,0], "x2" : X[:,1] , "y" : y})
df["y"] = df["y"].astype("category")

p9.ggplot(df , p9.aes(x = "x1" , y="x2" , color="y")) + p9.geom_point()

from scipy.stats import norm

from collections import defaultdict


class NaiveBayes():

    def fit(self,X,y):
        dists = {}
        yval = 1
        for yval in np.unique(y):
            xwithYval = X[np.argwhere(y == yval)[:,0]]
            # here is an assumption
            # for each column we can independently estimate the PDF
            estimatedMeans = np.apply_along_axis(np.mean ,0,xwithYval)
            estimatedStd = np.apply_along_axis(np.std ,0,xwithYval)
            params = zip(estimatedMeans , estimatedStd)
            # Here is one assumption
            # Numerical Variables are assumed to be normal , or any other distribution
            # Normal is WRONG! because probabilities can't be negative
            # but what the hell
            dist = [norm(u,s) for u,s in params]
            dists[yval] = dist


        yprobs = [np.sum(np.where(y == yval)) /y.shape[0] for yval in np.unique(y)]

        self.dists = dists
        self.yprobs = yprobs
        self.yuniques = np.unique(y)
        return self

    def predict(self,X):
        dists = self.dists
        yprobs = self.yprobs
        preds = []
        for i,row in enumerate(X):
            maxsofar = 0
            predClass = -1
            for k,yval in enumerate(self.yuniques):
                probs = []

                for j,col in enumerate(row):
                    dist = dists[yval]
                    prob = dist[j].pdf(col)
                    probs.append(prob)

                finalProb = np.prod(probs) * yprobs[k]
                if finalProb > maxsofar:
                    maxsofar = finalProb
                    predClass = yval
            preds.append(predClass)

        return preds


model = NaiveBayes()
model = model.fit(X,y)
preds = model.predict(X)


from sklearn.metrics import confusion_matrix

confusion_matrix(y,preds)


df["misclassfied"] = np.where(preds != y , 1 , 0)
df["misclassfied"] =df["misclassfied"].astype("category")

ggplot(df , aes(x = "x1" , y="x2" , color="misclassfied"  , shape="y")) +geom_point()


def expand_grid(x, y):
    xG, yG = np.meshgrid(x, y) # create the actual grid
    xG = xG.flatten() # make the grid 1d
    yG = yG.flatten() # same
    return pd.DataFrame({'x1':xG, 'x2':yG})

grid = expand_grid(np.arange(-20,20,0.2) , np.arange(-20,20,0.2))


decisionBoundaryDf = grid

decisionBoundaryDf["y"] = model.predict(decisionBoundaryDf[["x1" , "x2"]].as_matrix())
decisionBoundaryDf["color"] = np.where(decisionBoundaryDf["y"] == 1 , "#00880004" , "#88000004")

ggplot(decisionBoundaryDf , aes(x = "x1" , y="x2" , color="color")) \
    +geom_point() \
    +geom_point(df , aes(x = "x1" , y="x2" , color="y"))


# Caveats ?
# 1. you need a big data set in order to make reliable estimations of the probability of each class.
# 2. if you have no occurrences of a class label and a certain attribute value together (e.g. class="nice", shape="sphere")
#    then the frequency-based probability estimate will be zero.
#    Given Naive-Bayes' conditional independence assumption,
#    when all the probabilities are multiplied you will get zero
#    and this will affect the posterior probability estimate.
# 3. Maybe not use normal ?
#   John, G. H., & Langley, P. (1995). Estimating continuous distributions in Bayesian classifiers. Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence (pp. 338-345). Montreal, Quebec: Morgan Kaufmann.

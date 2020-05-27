%load_ext autoreload
%autoreload 2

import numpy as np

from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000 , n_features=1 ,n_informative=1 , n_redundant=0 , n_clusters_per_class=1)
w = np.ones(X.shape[1]).T

def loss_function(w,X,y):
    wtx = np.sum(w.T*X , axis=1)
    loss = np.abs(wtx-y)
    #print(wtx , loss)
    return np.mean(loss)


def stochastic(w,X,y):
    indice = int(np.random.random()*X.shape[0])
    xtiny = [X[indice , :]]
    ytiny = [y[indice]]
    #print(w,xtiny , ytiny)
    return loss_function(np.array([w]) ,xtiny , ytiny)

wposs = np.linspace(0,100,100)
gradientdescent = [ loss_function(w,X,y) for w in wposs]
stochasticloss =  [stochastic(w ,X,y ) for w in wposs]

import pandas as pd

df = pd.DataFrame({"w" : wposs , "gd" : gradientdescent , "sgd":stochasticloss})

from plotnine import *

ggplot(df , aes(x= "w" , y="gd")) + geom_point(color="green") + geom_point(aes(x="w" , y="sgd") ,color="red")

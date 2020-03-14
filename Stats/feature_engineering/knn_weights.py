import pandas as pd

from sklearn.datasets import make_blobs,make_classification


import numpy as np
from scipy.optimize import minimize , fmin_cg , fmin
from sklearn.metrics import hinge_loss , log_loss , confusion_matrix , accuracy_score , roc_auc_score
from plotnine import *
from sklearn.neighbors import KNeighborsClassifier
import logging

logging.basicConfig(level=logging.INFO)

class Experiment:

    def Run(self):
        logging.info("Creating Dataset")
        ninfo = 5
        X,y = make_classification(n_samples=100,class_sep=0.8, n_features=20,n_informative=ninfo ,n_redundant=0, n_classes=2)

        df = pd.DataFrame(X , columns=[f"x{i+1}" for i  in range(X.shape[1]) ])
        df["y"] = y
        #
        # from plotnine import *
        #
        # for i,x1 in enumerate(df.columns):
        #     for j,x2 in enumerate(df.columns):
        #         if j > i and (x1 != "y" and x2 != "y"):
        #             #print(x1,x2)
        #             p = ggplot(df , aes(x=x1 , y=x2 , color="y")) + geom_point()
        #             print(p)


        weights = np.zeros(X.shape[1])
        k = 5
        kdf = df.copy()

        def knnWeights(weights , kdf , k):
            df = kdf.copy()
            epsi = 0.0001
            df["predicted"] = 1
            indepCols = df.drop(["y" , "predicted"] ,axis=1).columns
            distmat = np.ones((df.shape[0],df.shape[0]))*10000
            for a,rowa in df.iterrows():
                for b,rowb in df.iterrows():
                    if a == b:
                        continue
                    #print(weights,rowa , rowb)
                    dist = 0
                    for i,col in enumerate(indepCols):
                        dist += weights[i]*np.abs(rowa[col]-rowb[col])

                    distmat[a,b] = dist

                kneighbors = np.argpartition(distmat[a], k)[:k]
                #print(distmat[a,kneighbors])
                #counts = np.bincount(df.iloc[kneighbors,:]["y"])
                #classMaj = np.argmax(counts)
                ys = df.iloc[kneighbors,:]["y"]
                prob = len(ys[ys==1])/k
                #print(prob)
                df.loc[a,"predicted"] = prob

            #loss = -1*hinge_loss(df["y"] , df["predicted"])
            #loss = 1-accuracy_score(df["y"] , df["predicted"])
            loss = -log_loss(df["y"] , df["predicted"])
            #print(loss,confusion_matrix(df["y"] , df["predicted"]))
            print(loss)
            return loss

        logging.info("Learning Feature Weights")
        lweights = fmin(knnWeights , weights ,args=(df,k) ,maxiter=10, disp=True)


        logging.info("Calculating Performance")

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X,y)

        #confusion_matrix(y,knn.predict(X))

        woa = accuracy_score(y , knn.predict(X))
        woauc = roc_auc_score(y, knn.predict_proba(X)[:,1])

        indices = np.argsort( np.abs(lweights))[-ninfo:]
        subX = X[:,indices]
        knn.fit(subX,y)

        #confusion_matrix(y,knn.predict(subX))

        wa = accuracy_score(y , knn.predict(subX))
        wauc = roc_auc_score(y, knn.predict_proba(subX)[:,1])

        rdf = pd.DataFrame({"woa":[woa] , "woauc":[woauc] , "wa":[wa] , "wauc":[wauc]})

        return rdf



dfs = [Experiment().Run() for i in range(10)]

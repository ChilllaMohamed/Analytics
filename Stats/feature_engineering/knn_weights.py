import pandas as pd

from sklearn.datasets import make_blobs,make_classification


import numpy as np
from scipy.optimize import minimize , fmin_cg , fmin
from sklearn.metrics import hinge_loss , log_loss , confusion_matrix , accuracy_score , roc_auc_score
from plotnine import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
import logging

from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO)


class Experiment:

    def Run(self):
        logging.info("Creating Dataset")
        ninfo = 5
        X,y = make_classification(n_samples=300,
                                    n_features=ninfo,
                                    n_classes=2)

        #np.random.seed(100)
        E = np.random.normal(0, 1, size=(len(X), 10))
        X = np.hstack((X, E))
        print(X.shape)

        weights = np.zeros(X.shape[1])
        k = 5

        def knnWeights(weights , X,y , k):

            def dist(a,b):
                return np.sum(np.multiply(weights, np.abs(a-b)))

            distMat = pairwise_distances(X,metric=dist)
            kneighbors = np.argsort(distMat , axis=1)[:,1:k+1]

            # don't do probability because its a bad estimation to begin with
            #probs = np.count_nonzero(y[kneighbors] == 1 , axis=1)/k
            pred = np.round(np.sum(y[kneighbors] , axis=1)/k)
            #loss = 1-roc_auc_score(y,probs) #+ np.linalg.norm(weights , ord=2)
            loss = 1-accuracy_score(y,pred)
            #loss = log_loss(y,probs)
            return loss

        logging.info("Learning Feature Weights")

        Xtrain,X_test,y_train,y_test = train_test_split(X,y)

        #lweights = fmin(knnWeights , weights ,args=(df,k) ,maxiter=10, disp=True)
        lweights = minimize(knnWeights , weights ,args=(Xtrain,y_train,k) ,options={"maxiter":10})
        lweights = lweights.x
        print(lweights)
        logging.info("Calculating Performance")

        knn = KNeighborsClassifier()

        knn.fit(Xtrain,y_train)

        #print(rfc.feature_importances_)
        #confusion_matrix(y,knn.predict(X))

        woa = accuracy_score(y_test, knn.predict(X_test))
        woauc = roc_auc_score(y_test, knn.predict_proba(X_test)[:,1])

        randomIndices = np.random.choice(len(lweights) , ninfo,replace =False)

        selectedIndices = np.argsort( np.abs(lweights))[-ninfo:]
        print(randomIndices, selectedIndices)
        accs = []
        aucs = []
        for indices in [randomIndices , selectedIndices]:
            #indices = np.argsort( np.abs(lweights))[:ninfo]
            subX = Xtrain[:,indices]

            def dist(a,b):
                return np.sum(np.multiply(lweights[indices], np.abs(a-b)))

            knn = KNeighborsClassifier(metric = dist)
            knn.fit(subX,y_train)

            #confusion_matrix(y,knn.predict(subX))
            subtestX = X_test[:,indices]
            wa = accuracy_score(y_test , knn.predict(subtestX))
            wauc = roc_auc_score(y_test, knn.predict_proba(subtestX)[:,1])
            accs.append(wa)
            aucs.append(wauc)

        rdf = pd.DataFrame({ "rfa":[accs[0]] , "rfauc":[aucs[0]],"woa":[woa] , "woauc":[woauc] , "wa":[accs[1]] , "wauc":[aucs[1]]})

        return rdf



dfs = [Experiment().Run() for i in range(10)]

d = pd.concat(dfs)
d

d.describe()

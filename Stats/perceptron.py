from plotnine import *

import pandas as pd
import numpy as np


from sklearn.datasets import make_classification , make_blobs

#X,y = make_classification(n_samples=100, n_features=2 , n_redundant=0,n_informative=2)
X,y = make_blobs(n_samples=20, centers=2, n_features=2 ,cluster_std=2.8)
y = np.where(y == 0 , -1 , 1)

df = pd.DataFrame(dict(x1 = X[:,0],x2 = X[:,1], x3 = 1, y = y))
df["y"] = df["y"].astype("category")
ggplot(df , aes(x="x1" , y="x2" , color="y")) + geom_point()


from sklearn.preprocessing import StandardScaler
df[["x1" , "x2"]] = StandardScaler().fit_transform(df[["x1" , "x2"]])


ggplot(df , aes(x="x1" , y="x2" , shape="y", color="y")) + geom_point()



def gprint(wdf,rdf):
    p = (ggplot(wdf)
    + geom_segment(aes(x="x1", y="x2" ),xend=0 , yend=0 )
    + geom_segment(rdf, aes(x="x1" , y="x2" ) , xend=0 , yend=0)
     + geom_point(ndf , aes(x="x1" , y="x2" , color="y",shape="y") , size=5)
     + geom_text(ndf,aes(x="x1" , y="x2" , label="misclassfiedCount" ),color="green"))
    print(p)

W = np.array([0.5,0.5,0.5])

weights = [W]

misclassfiedCount = np.zeros(df.shape[0]) -1
iter = 0
#np.cross(np.array(weights[0]) ,weights[1].astype("float"))
while True:
    m = 0
    #row = df.iloc[0]
    for i,row in df.iterrows():
        isCorrectlyClassified = row["y"]*(np.dot(W, row.drop("y").values)) > 0
        if not isCorrectlyClassified:
            wdf = pd.DataFrame([W] , columns =df.drop("y" , axis=1).columns.values)
            rdf = pd.DataFrame([row.drop("y").values] , columns =df.drop("y" , axis=1).columns.values)
            gprint(wdf , rdf)
            W = W + row["y"]*row.drop("y").values
            weights.append(W)
            misclassfiedCount[i] = iter
            iter += 1
            m += 1
            #print(W ,df.drop("y" , axis=1).columns.values )


    if m == 0:
        break

weights


#weights = np.apply_along_axis(lambda a : a / a[2] ,1, weights)
#weights = np.apply_along_axis(lambda a : a / 1  ,1, weights)
#weights = weights-1
#np.divide(np.array(weights) , np.array(weights[0]) , axis=1)



weightChange = pd.DataFrame(weights , columns=df.drop("y" , axis=1).columns)
weightChange["iteration"] = weightChange.index.values
weightChange["iteration"] = weightChange["iteration"].astype("str")
weightChange[["x1" , "x2" , "x3"]] = weightChange[["x1" , "x2" , "x3"]].astype("float")

ndf = df.copy()
ndf.loc[:,["x1" , "x2" , "x3"]] = ndf.loc[:,["x1" , "x2" , "x3"]].astype("float")
ndf["misclassfiedCount"] = misclassfiedCount
ndf["index"] = ndf.index
ndf["color"] = np.where(ndf["misclassfiedCount"] == -1 , "#00007708" , "#556677ff")

(ggplot(weightChange)
+ geom_segment(aes( y="-x3/x2" ,xend="-x3/x1" ),x=0 , yend=0 )
 + geom_point(ndf , aes(x="x1" , y="x2" , color="color",shape="y") , size=5)
 + geom_text(ndf,aes(x="x1" , y="x2" , label="misclassfiedCount" ),color="green")
+ geom_text(aes( y="-x3/x2" , label="iteration" ),color="red" , size=20 ,x=0)
 #+ scale_x_continuous(limits=(-3,3))
 #+ scale_y_continuous(limits=(-5,5))
 + theme(aspect_ratio = 1))


# Bias is important here
# [array([0.5, 0.5, 0.5]),
#  array([-0.13363456714334288, 0.9068832653928369, -0.5], dtype=object),
#  array([-0.8544795891705356, 0.8527281195677334, 0.5], dtype=object),
#  array([-1.5145657382641953, -1.0273904262927762, 1.5], dtype=object),
#  array([-1.9327834786398084, -1.7134430192886536, 0.5], dtype=object),
#  array([-2.421149233768465, -0.38282945988386685, -0.5], dtype=object)]
# x3 +ve means that the side with origin is postive side
# x3 -ve means that the side with origin is negative side

# that is if i know nothing about the features x1 and x2 what would i guess -1 or 1 ?
# that is bias

import numpy as np
import pandas as pd
from plotnine import *


x1 = np.linspace(-1 , 1 , 100)
dist = 1
df = pd.DataFrame({"x" : x1 })

for d in [1,2,4,8,16]:

    y = dist-np.power(x1,d)
    df[d] = np.power(y , 1/d)

df = df.melt(id_vars="x")

ggplot(df , aes(x="x" , y="value" , color="variable"))  + geom_point()

# Since x1 , x2 are bound between -1,1 we invert the top left quad and
# then symmeterically get the lower quads as well

# this means that if manhattan distance , euclidean distance = 1 then points for whom
# manhattan distance was used are closer in space
# as opposed to those for which euclidean distance was use
# higher order gives room for viggle and is not susiptible to smaller changes

import math

x1 = np.linspace(-3 , 3 , 100)
x2 = np.linspace(-3 , 3 , 100)

recs = []
for d in [1,2,6,12,24]:
    for xa in x1:
        for xb in x2:
            v = math.pow(xa , d) + math.pow(xb , d)
            #v = math.pow((xa-xb) , d) #+ math.pow(xb , d)
            #print(v)
            dist = abs(math.pow(v , 1/d))
            recs.append({"x1" : xa , "x2" : xb , "dist":dist , "d" : d})



df = pd.DataFrame(recs)

ggplot(df , aes(x="x1" , y="x2" ,color="dist"))  + geom_point() + facet_grid("~d")

p = 6
x = np.linspace(-3,3 ,100)
y = np.power(1- x**p , 1/p)

curveDf = pd.DataFrame({"x" : x , "y":y})

ggplot(curveDf , aes(x="x",y="y")) + geom_point()

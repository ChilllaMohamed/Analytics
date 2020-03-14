from scipy.stats import norm
from plotnine import *
import numpy as np
import pandas as pd

xv = np.linspace(100,200,200)
yv = norm.pdf(xv , 150 , 10)

df = pd.DataFrame({"x" : xv , "y" : yv})

(ggplot( df , aes(x = "x" , y="y")) + geom_point())


# Ideally we won't have to sum np.sum(yv) part as xv would include all posible values
# But in this case we only have 200 values between 100-200 instead of the infinite possible values
# therefore we sum up the probabilities

np.sum(xv*yv) / np.sum(yv)


xv = np.linspace(100,200,20000)
yv = norm.pdf(xv , 150 , 10)
np.sum(xv*yv)

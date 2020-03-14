import pandas as pd
import pathlib

import os
from plotnine import *

path = pathlib.Path().resolve()
projPath = os.path.abspath(os.path.join(path , "DSBA6156" , "Loans"))
csvPath  = os.path.abspath(os.path.join(projPath ,"loans.csv"))
df = pd.read_csv(csvPath)


(ggplot(df ,  aes(x='lender_count' , y='loan_amount'))
+ geom_point()
+ geom_smooth(method='lm'))


(ggplot(df ,  aes(x='lender_count' , y='loan_amount' ,color="status"))
+ geom_point()
+ geom_smooth(method='lm' , color='red')
+ scale_x_continuous(limits=(0,600))
+ scale_y_continuous(limits=(0,12000)))

df["gotEverything"] = df["funded_amount"] == df["loan_amount"]

df["lender_count"].describe()
bins = list(df["lender_count"].describe())[-5:]
#bins = [0,41,2665]
df["lender_bin"] = pd.cut(df["lender_count"] ,bins)


(ggplot(df[df["lender_count"] < 50] ,  aes(x='repayment_term' , y='loan_amount' ,color="lender_bin"))
+ geom_point(alpha=0.5)
+ geom_smooth(method='lm' , color='red')
+ scale_x_continuous(limits=(0,150))
+ scale_y_continuous(limits=(0,20000)))

loanAmountMeanByStatus  = df.groupby(["repayment_term" , "lender_bin"]).agg({"loan_amount" : "mean"}).reset_index()

(ggplot(loanAmountMeanByStatus , aes(x="repayment_term" , y="loan_amount" , color="lender_bin"))
+ geom_line())
df["lender_count"].describe()
(ggplot(df , aes(x='status' , y = "repayment_term" , color='lender_count')) + geom_boxplot())

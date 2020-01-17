from bs4 import BeautifulSoup
import requests
import re

from pymagnitude import *
import os

import pandas as pd
import numpy as np
from nltk import ngrams, FreqDist , word_tokenize
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

stpwrds = None
tokenizer = None

import json

class Info:

    def __init__(self,title,funding,valuation,text):
        self.title = title.split("|")[1].strip(" ")
        self.rank = title.split("|")[0].strip(" ")
        self.funding = re.findall("([\d\.]+ (billion|million))" ,  funding)[0][0]
        self.valuation = re.findall("([\d\.]+ (billion|million))" ,  funding)[0][0]
        self.text = text.lower()
        self.tokens = [ t for t in tokenizer.tokenize(self.text) if t not in stpwrds]

    def SetRankForKW(self , kwrank):
        self.kwrank = kwrank

    def GetFundingNum(self):
        bOrM =  (1000000000 if self.funding.find( "billion") >= 0 else 1000000)
        return float(re.findall("[\d.]+" , self.funding)[0]) *bOrM

    def GetTokens(self):
        return self.tokens

    def ToJson(self):
        return {
            "title" : self.title,
            "rank" : self.rank,
            "kwrank" : str(self.kwrank),
            "funding" : self.funding,
            "fundingNum" : str(self.GetFundingNum()),
            "valuation" : self.valuation,
            "text" : self.text
        }

    def Print(self):
        print(f"Title : {self.title} \n; Funding : {self.funding} \n; Valuation : {self.valuation} \n; Text : {self.text}\n\n")

class DoIt:

    def __init__(self):
        self.magModel = "glove-lemmatized.6B.50d.magnitude"
        global stpwrds,tokenizer
        stpwrds = stopwords.words("english")
        tokenizer = RegexpTokenizer(r'\w+')

    def GetData(self):
        url = "https://www.forbes.com/sites/jilliandonfro/2019/09/17/ai-50-americas-most-promising-artificial-intelligence-companies/#3f011b46565c"
        page = requests.get(url)
        content = page.text
        soup = BeautifulSoup(content)
        h2s = soup.find_all("h2" , class_= "subhead-embed")
        allH4s = soup.find_all("h4" , class_= "subhead4-embed")

        #print(h2s)
        texts = [h2 for h2 in h2s if h2.text.find("|") >= 0]

        validH4s = [h2 for h2 in allH4s if h2.find("strong") is not None and h2.text.find("READ MORE") < 0]
        infos = []
        print(len(validH4s))
        for i,t in enumerate(texts):
            #print(t)
            h4s = validH4s[i*4:i*4+4]
            #print(h4s)
            infos.append(Info(t.text , h4s[2].text , h4s[3].text , t.find_next("p").text))
        #print(texts)

        for inf in infos:
            inf.Print()

        self.SetInfos(infos)

    def SetInfos(self, infos):
        self.infos = infos

    def GetInfos(self):
        return self.infos

    def SetFreqDist(self,dist):
        self.freqDist = dist

    def GetFreqDist(self):
        return self.freqDist

    def FreqDistribution(self):
        tokens = [t for inf in self.GetInfos() for t in inf.GetTokens()]
        self.SetFreqDist(FreqDist(tokens))

    def PrintCommon(self):
        dist = self.GetFreqDist()
        print(dist.most_common(n=50))

    def DocSimiliarity(self,kw):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__)  ,self.magModel))
        vectors = Magnitude(path)
        simis = []
        for inf in self.GetInfos():
            me = np.max(vectors.similarity(kw , inf.GetTokens()))
            inf.SetRankForKW(me)
            #simis.append((inf , me))

        #return sorted(simis , key=lambda x : x[1])

    def Run(self , file):
        self.GetData()
        self.FreqDistribution()
        self.PrintCommon()
        self.DocSimiliarity("data")

        with open(file, "w") as fp:
            json.dump([inf.ToJson() for inf in self.GetInfos()] , fp)



class Loader:

    def __init__(self,file):
        with open(file) as fp:
            self.jsstr = json.dumps(json.load(fp))

    def Load(self):
        self.df = pd.read_json(self.jsstr, orient="records")
        print(self.df.head())

    def Describe(self):
        print("TOP 10 Data related start ups: ")
        print(self.df.columns)
        data =  self.df[ self.df["kwrank"] >= 1.0]
        notData =   self.df[ self.df["kwrank"] < 1.0]
        print("TOP 5 Data related start ups: ")
        print(data.sort_values("fundingNum" , ascending=False).head(5))
        print("TOP 5 Non- Data related start ups: ")
        print(notData.sort_values("fundingNum" , ascending=False).head(5))
        datafunding = data["fundingNum"].sum()
        notDataFunding = notData["fundingNum"].sum()

        print("Data Automation Startups")
        print(data["fundingNum"].describe())
        print("Non-Data  Automation Startups")
        print(notData["fundingNum"].describe())
        #print(f"Total Valuation of these data start ups( {data.shape[0]}) -  {datafunding/1000000} Millions")
        #print(f"Total Valuation of Non-data start ups ( {data.shape[1]}) {notDataFunding/1000000} Millions")

    def Run(self):
        self.Load()
        self.Describe()

if __name__ == "__main__":

    #DoIt().Run("out.json")
    Loader("out.json").Run()

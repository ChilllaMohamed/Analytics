
from pymongo import MongoClient
from IndianMedia.constants import MongoConsts

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class DBConnection():

    def __init__(self):
        self.client = MongoClient("192.168.99.100" , 27017)
        self.indianMediaVideoCollection = self.client[MongoConsts.DB][MongoConsts.VIDEO_COLLECTION]

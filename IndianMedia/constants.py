

class Creds:

    KEY_FILE = "_keyfile"

class MongoConsts:
    DB = "indianmedia"
    VIDEO_COLLECTION = "videos"

class Channels:
    WIRE= "UChWtJey46brNr7qHQpN6KLQ"
    THE_PRINT = "UCuyRsHZILrU7ZDIAbGASHdA"
    INDIA_TODAY = "UCYPvAwZP8pZhSMW8qs7cVCw"
    REPUBLIC_WORLD = "UCwqusr8YDwM-3mEYTDeJHzw"

    LOOKUP = {
        "WIRE" : WIRE,
        "PRINT": THE_PRINT,
        "INDIA_TODAY" : INDIA_TODAY,
        "REPUBLIC_WORLD" : REPUBLIC_WORLD
    }

    MAX_VIDEOS = 2000

    VID_TYPE_LIST = "youtube#videoListResponse"

    @staticmethod
    def reverseLookup(channelId):
        for k,v in Channels.LOOKUP.items():
            if v == channelId:
                return k

# %load_ext autoreload
# %autoreload 2

import pyyoutube

import os
import pathlib
import json
import logging
import datetime
import math

import numpy as np

logging.basicConfig(level=logging.INFO)

from IndianMedia.constants import Channels , MongoConsts , Creds
from IndianMedia.pymongoconn import DBConnection


try:
    __file__
except:
    __file__ = os.path.abspath(os.path.join("." ,"..","Analytics","IndianMedia",Creds.KEY_FILE))

connection = DBConnection()


f = os.path.abspath(os.path.join(os.path.dirname(__file__) ,Creds.KEY_FILE))
key = open(f,"r").read().strip("\n")

api = pyyoutube.Api(api_key=key)


#p = pl.items[0]

def GetChannelVideoInfo(channelId):
    pl = api.get_playlists( channel_id =channelId , count=200)

    #all_videos = []
    for j,i in enumerate(pl.items):
        logging.info(f"-- Loading - {j} Playlist")

        pljs = json.loads(i.to_json())

        videos = api.get_playlist_items(playlist_id=i.id , count=200)

        #vid = videos.items[0]
        for k,vid in enumerate(videos.items):
            logging.info(f"-- Loading - {k} - Video")
            vidId = vid.snippet.resourceId.videoId

            tjs = json.loads(vid.to_json())
            tjs["playlist"] = pljs
            tjs["created_date_time"] = datetime.datetime.now()
            connection.client[MongoConsts.DB][MongoConsts.VIDEO_COLLECTION].insert_one(tjs)

            #all_videos.append(tjs)


    #chosen_videos = np.random.choice(all_videos ,max(Channels.MAX_VIDEOS , len(all_videos)) , replace=False)

    #for k,vid in enumerate(chosen_videos):
        #connection.client[MongoConsts.DB][MongoConsts.VIDEO_COLLECTION].insert_one(tjs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id" , help="Youtube Channel Id or Channels Lookup Name from IndianMedia.constants import Channels")

    args = parser.parse_args()
    if args.channel_id in Channels.LOOKUP:
        args.channel_id = Channels.LOOKUP[args.channel_id]

    GetChannelVideoInfo(args.channel_id)

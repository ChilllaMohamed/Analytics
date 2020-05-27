# %load_ext autoreload
# %autoreload 2

from IndianMedia.pymongoconn import DBConnection
from IndianMedia.utils import getCurrentDIR
from IndianMedia.constants import Channels

connection  =DBConnection()

collection  = connection.indianMediaVideoCollection

import csv
import os


rows = []
for vid in collection.find():
    typeList = vid["kind"] == Channels.VID_TYPE_LIST

    #print(vid)
    info = vid["items"][0] if typeList else vid
    info = info["snippet"]

    channelId = Channels.reverseLookup(info["channelId"])
    ptitle = vid["playlist"]["snippet"]["title"]
    title = info["title"]
    desc = info["description"]
    date = info["publishedAt"]
    rows.append([channelId,ptitle,date,title,desc])

    #if ('items' in vid and len( vid["items"]) > 0) or vid["kind"].lower().find("playlistitem") > -1:



dir = getCurrentDIR()
f = os.path.abspath(os.path.join(dir , "text.csv"))

header = ["Channel Id" , "Playlist Title" , "Date" , "Title" , "Description"]

with open(f,"w" , encoding="utf-8", newline='') as fobj:
    writer = csv.writer(fobj)
    writer.writerow(header)
    writer.writerows(rows)

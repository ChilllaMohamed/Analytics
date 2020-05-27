import os

def getCurrentDIR():
    try:
        __file__
    except:
        # Using DT virtual environment in Github/Datathon
        __file__ = os.path.abspath(os.path.join("." ,"..","Analytics","IndianMedia","_keyfile"))

    return os.path.abspath(os.path.dirname(__file__))

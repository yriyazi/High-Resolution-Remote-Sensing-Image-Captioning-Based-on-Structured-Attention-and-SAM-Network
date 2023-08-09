import py7zr

def extract(sourse:str,target:str):
    with py7zr.SevenZipFile(sourse, mode='r') as z:
        z.extractall(path = target)
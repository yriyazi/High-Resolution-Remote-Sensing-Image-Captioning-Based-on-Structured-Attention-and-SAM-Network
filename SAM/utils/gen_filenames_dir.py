import os

# Run only once
def Filename_generator(path):
    with open(os.path.join(path, 'filneame.txt'), "w") as file:  
        for item in dir:
            # write each item on a new line
            file.write("%s\n" % item)
        print('Done')
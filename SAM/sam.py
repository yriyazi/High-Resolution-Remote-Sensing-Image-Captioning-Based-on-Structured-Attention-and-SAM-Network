import  glob,   tqdm,   os, cv2,    time
import  utils,  nets
import  numpy             as np
from    operator          import itemgetter
#%%
# ###### Change dataset in here #########
path = os.path.join('datasets', 'UCM')
start = 0
end   = -1
#%%
if not os.path.isfile(os.path.join(path,'filneame.txt')):
    utils.Filename_generator(path=path)
    print("filename generated" )
    
dirg = []
with open(os.path.join(path, 'filneame.txt'), "r") as file:
    for item in file:
        # write each item on a new line
        dirg.append(item)    
    
#%%
cc = [os.path.join(path,'imgs',os.path.split(item)[-1][:-1]) for item in dirg]

i = 0
for file in tqdm.tqdm(cc[start:end]):
  name = os.path.split(file)[-1].split('.')[0]
  # Loading image

  if not os.path.isfile(os.path.join(path,'masks',name+'.npz')):
    image = cv2.imread(file)
    image = cv2.resize(image, (utils.dim_input,utils.dim_input), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # generate ROI
    masks = nets.mask_generator.generate(image)
    # Sorting dictionary by area
    masks = sorted(masks, key=itemgetter('area'),reverse=True) 

    Dump = np.zeros(shape=[utils.channel_input,utils.dim_input,utils.dim_input])
    for i in range(min(len(masks),utils.channel_input)):
      Dump[i,:,:] = masks[i]['segmentation'].astype(int)

    np.savez_compressed(os.path.join(path,'masks',name+'.npz'), my_array=Dump)
  
    # i += 1
    # if i%10 == 0: time.sleep(5)
  else:
    print(f"{file=} exist")

from .graber            import *
from .JSONloaders       import *
from .Data_loader       import *
from .dataset           import *
from .Target_token      import *
from .extract           import extract
import  os
import  torchvision.transforms      as      transforms
from    torch.utils.data            import  DataLoader

def remove_invalid_files(file_list):
    valid_files = []
    for file_address in file_list:
        if os.path.isfile(file_address):
            valid_files.append(file_address)
        else:
            print(f"Invalid file address: {file_address}. File not found.")
    
    return valid_files

import py7zr
def extract(sourse:str,target:str):
    with py7zr.SevenZipFile(sourse, mode='r') as z:
        z.extractall(path = target)
for dirpath, dirnames, filenames in os.walk(os.path.join('datasets')):
    for filename in filenames:   
            
        if not os.path.exists(os.path.join(dirpath,filename.split('.')[0])) :
            if filename.split('.')[-1] == '7z':
                print(f"extracting {os.path.join(dirpath,filename)}")
                extract(sourse = os.path.join(dirpath,filename), target = os.path.join(dirpath))


UCM_JSON   = os.path.join('datasets', 'UCM'  ,'dataset.json')
RSICD_JSON = os.path.join('datasets', 'RSICD','dataset.json')

source = Countainer()


# Defieing all datasets
RSICD       = dataloaders.JSONloaders(RSICD_JSON)
data_frame  = dataloaders.read_dataset(RSICD._all_Sentences())
source      = dataloaders.process_data(data_frame,source)

UCM         = dataloaders.JSONloaders(UCM_JSON)
data_frame  = dataloaders.read_dataset(UCM._all_Sentences())
source      = dataloaders.process_data(data_frame,source)

transform = transforms.Compose([
                                transforms.Resize((512, 512)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])

if utils.DatasetInUse == "UCM":
    root_directory          = os.path.join('datasets', 'UCM')
    train,validation,test = dataloaders.UCM.splitter()
    train       = [os.path.join(root_directory, 'imgs',index) for index in train]
    validation  = [os.path.join(root_directory, 'imgs',index) for index in validation]
    test        = [os.path.join(root_directory, 'imgs',index) for index in test]

    # train       = remove_invalid_files(train)
    # validation  = remove_invalid_files(validation)
    # test        = remove_invalid_files(test)

    tarin_dataset = dataloaders.CustomDataset(jpg_files = train     ,transformation = transform,JSON = dataloaders.UCM,root_directory = root_directory)
    valid_dataset = dataloaders.CustomDataset(jpg_files = validation,transformation = transform,JSON = dataloaders.UCM,root_directory = root_directory)
    test_dataset  = dataloaders.CustomDataset(jpg_files = test      ,transformation = transform,JSON = dataloaders.UCM,root_directory = root_directory)

elif utils.DatasetInUse == "RSICD":
    root_directory          = os.path.join('datasets', 'RSICD')
    train,validation,test = dataloaders.RSICD.splitter()
    train       = [os.path.join(root_directory, 'imgs',index) for index in train]
    validation  = [os.path.join(root_directory, 'imgs',index) for index in validation]
    test        = [os.path.join(root_directory, 'imgs',index) for index in test]

    train       = remove_invalid_files(train)
    validation  = remove_invalid_files(validation)
    test        = remove_invalid_files(test)
    
    tarin_dataset = dataloaders.CustomDataset(jpg_files = train     ,transformation = transform,JSON = dataloaders.RSICD,root_directory = root_directory)
    valid_dataset = dataloaders.CustomDataset(jpg_files = validation,transformation = transform,JSON = dataloaders.RSICD,root_directory = root_directory)
    test_dataset  = dataloaders.CustomDataset(jpg_files = test      ,transformation = transform,JSON = dataloaders.RSICD,root_directory = root_directory)





import  torch
import  dataloaders
import  os
import  utils
import  numpy               as      np
from    torch.utils.data    import  Dataset
from    PIL                 import  Image
from    torch.nn.utils.rnn  import  pad_sequence
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
def collate_fn(batch):
    """
    This is a function for collating a batch of variable-length sequences into a single tensor, which is
    useful for training a neural network with PyTorch.

    The input to this function is a batch of samples, each containing a source and target sequence. 
    The function extracts the source and target sequences from each sample, and then pads them to ensure
    that all sequences in the batch have the same length. This is necessary because PyTorch requires all
    inputs to a neural network to have the same shape.

    The function uses the PyTorch pad_sequence function to pad the sequences. pad_sequence is called with
    the batch_first=True argument to ensure that the batch dimension is the first dimension of the output
    tensor. The padding_value argument is set to 0 to pad with zeros.

    """
    _one = [item[0] for item in batch]
    _two = [item[1] for item in batch]
    _thr = [item[2] for item in batch]
              
    _1 = torch.stack(_one)
    _2 = torch.stack(_two)
    _3 = torch.nn.utils.rnn.pad_sequence(_thr, batch_first=True)

    return _1, _2 , _3

class CustomDataset(Dataset):
    # def __init__(self, npz_files ,jpg_files,JSON, transformation=None):
    def __init__(self,jpg_files,JSON, transformation,root_directory,Sort_masks=True):

        self.original_picture   = jpg_files
        self.mask_dir           = [os.path.join(root_directory, 'masks',os.path.split(index)[-1].split('.')[0]+'.npz') for index in jpg_files]

        self.transformation     = transformation
        self.JSON               =  JSON
        self.Sort_masks         = Sort_masks
        
        self.N = utils.N_ROI
    
    
        self.resize = transforms.Resize(size=(512, 512))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform(self, image, mask):
        """
        custum transformation 
        #https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/6

        """

        # Random horizontal flipping
        if random.random() > 0.5:
            image   = TF.hflip(image)
            mask    = TF.hflip(mask)
        # Random vertical flipping
        if random.random() > 0.5:
            image   = TF.vflip(image)
            mask    = TF.vflip(mask)
        
        
        # Resize
        image = self.resize(image)
        # Transform to tensor
        image       = TF.to_tensor(image)
        #Normalize
        image = self.normalize(image)
        
        return image, mask
    
    
    def __len__(self):
        return len(self.original_picture)
    
    def __getitem__(self, index):
        adress = self.original_picture[index]
        #-----------------------------image-----------------------------
        image = Image.open(adress).convert("RGB")
        
        #------------------------------mask-------------------------------
        mask = np.load(self.mask_dir[index])['my_array']
        mask = torch.from_numpy(mask).to(torch.float32)[0:self.N,:,:]
        
        if self.Sort_masks==utils.sorted_mask:
            row = mask.sum(dim=1).sum(dim=1)
            sSort = row.sort(descending=True).indices
            _mask = torch.zeros_like(mask)
            for i in range(mask.shape[0]):
                _mask[i,:,:] = mask[sSort[i].item(),:,:]
            mask = _mask
        #------------------------------transformation--------------------    
        if self.transformation != None:
            image, mask = self.transform(image, mask)
        #---------------------------sentences-----------------------------
        _,lsss = self.JSON.item(os.path.split(adress)[-1])
        data = []
        for item in lsss :
            item = item.split(" ")
            sentence = [1] # 'SOS'
            for word in item:
                try:
                    sentence.append(dataloaders.source.word2index[word])
                except:
                    # ignore none token types
                    continue
            sentence.append(2) # 'EOS'
            data.append(sentence)
        # Convert the lists to tensors
        tensor_list = [torch.tensor(lst).to(torch.int32) for lst in [*data]]
        # Pad the sequences
        padded_sequence = pad_sequence(tensor_list, batch_first=True)
        #---------------------------------------------------------------

        if utils.rand_sent:
            return image,mask,padded_sequence[torch.randint(low=0,high=5,size=[1]).item()]
        
        return image,mask,padded_sequence[0]



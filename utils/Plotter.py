import torch
import utils
import dataloaders
import numpy as np
import matplotlib.pyplot as plt
import  torchvision.transforms      as      transforms

class plotter():
    def __init__(self,
                 device = 'cpu' ) -> None:
        self.device = device
        # Assuming you have a normalized image stored in the variable 'normalized_image'
        # Define the normalization parameters used in transforms.Normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        
        self.dataset = dataloaders.test_dataset
    
    def unnormalize(self,
                    imgs:torch.tensor):
        
        # Create the unnormalize transform
        unnormalize_transform = transforms.Normalize(
                                                        mean=[-m / s for m, s in zip(self.mean, self.std)],
                                                        std=[1 / s for s in self.std]
                                                    )
        # Convert the normalized image back to the original scale
        unnormalized_image = unnormalize_transform(imgs.cpu())
        # Convert the tensor to a numpy array
        return np.transpose(unnormalized_image.numpy(), (1, 2, 0))
    
    @staticmethod
    def show_anns(anns:np.array):
        # ax = plt.gca()
        # ax.set_autoscale_on(False)
        img = np.zeros((512,512, 4))
        for ann in range(utils.N_ROI):
            img[anns[ann]] = np.concatenate([np.random.random(3), [0.95]])
        # ax.imshow(img)
        return img
    
    @staticmethod
    def visualize_mask(mask_array:np.array):
        num_channels = mask_array.shape[0]
        rows = int(np.ceil(np.sqrt(num_channels)))
        cols = int(np.ceil(num_channels / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.3)

        for i, ax in enumerate(axes.flatten()):
            if i < num_channels:
                ax.imshow(mask_array[i], cmap='gray')
                ax.set_title(f'Channel {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        plt.show()
    
    def data(self,
               index:int):
        
        # image,mask,padded_sequence  = self.dataset[index]
        # image                       = image.unsqueeze(0).to( self.device)
        # mask                        = mask.unsqueeze(0).to( self.device)
        # padded_sequence             = padded_sequence.unsqueeze(0).to( self.device)
        return self.dataset[index] #image,mask,padded_sequence
    
    def ind2word(self,input):
        temp = torch.tensor(input).clone().detach()[torch.tensor(input)!=0 ]
        temp = temp[temp!=1]
        temp = temp[temp!=2]
        
        return " ".join([dataloaders.source.index2word[item.item()] for item in temp])
    
    def plot(self,
             index:int,
             caption1:str,
             save_adress = None,
             save_iamge = False,show_iamge = True):
        img,masked_img,padded_sequence = self.data(index)
        
        image = self.unnormalize(img)
        masks = self.show_anns(masked_img.cpu().numpy().astype(bool))


        
        fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(10,5) , dpi=150)
        fig.suptitle(f"{index=} of test dataset ",y=0.95+0.05 , fontsize=20)
        # Plot the first image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title("Original Picture", pad=10)

        # Plot the second image
        axes[1].imshow(image)
        axes[1].imshow(masks)
        axes[1].axis('off')
        axes[1].set_title("segmented Picture", pad=10)
        
        # Add the caption
        for _index in range(len(caption1)):
            fig.text(0.5,  0.075 -0.09*_index, f"Prediction {_index+1} :{caption1[_index]}", ha='center',
                fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
        
        fig.text(0.5, -0.37, f"Ground Truth :{self.ind2word(padded_sequence)}", ha='center',
             fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8))
        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.1)
        if save_iamge==True:
            plt.savefig(save_adress+str(index)+'.png', bbox_inches='tight')
        
        if show_iamge==True:
            # Show the figure
            plt.show()
            
            
    def plot_SAM(self,
            index:int,
            caption1:str,
            save_adress = None,
            save_iamge = False,show_iamge = True):
        img,masked_img,padded_sequence = self.data(index)
        
        image = self.unnormalize(img)
        masks = self.show_anns(masked_img.cpu().numpy().astype(bool))


        
        fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize=(10,5) , dpi=150)
        fig.suptitle(f"{index=} of test dataset ",y=0.95+0.05 , fontsize=20)
        
        # Plot the first image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title("Original Picture", pad=10)

        # Plot the second image
        axes[1].imshow(image)
        axes[1].imshow(masks)
        axes[1].axis('off')
        axes[1].set_title("segmented Picture", pad=10)
        
        if save_iamge==True:
            plt.savefig(save_adress+str(index)+'.png', bbox_inches='tight')
        
        if show_iamge==True:
            # Show the figure
            plt.show()

    def plot_attention(self,
                       predicted:str,
                       index:int,
                       attention_outputs,save_adress = None,save_iamge = False,show_iamge = True):
        cols = int(np.sqrt(len(predicted.split(' '))))+1
        rows = cols

        fig, axes = plt.subplots(rows, cols,figsize=(20, 10))
        fig.suptitle(f"{index=} of test dataset ",y=0.95+0.05 , fontsize=20)

        image,masked_img,_ = self.data(index)

        image = self.unnormalize(image)
        masks = masked_img.cpu().numpy().astype(bool)

        # argmaxc = attention_outputs.argmax(dim=2).cpu()[0]
        values , indices =  attention_outputs.cpu().topk(3,dim=2)
        for i, ax in enumerate(axes.flatten()):
            if i < len(predicted.split(' ')):
                _img = np.zeros((512,512, 4))
                ax.imshow(image)
                
                
                
                _img[masks[indices[0][i][0]]] = np.concatenate([np.random.random(3), [0.90]])
                _img[masks[indices[0][i][1]]] = np.concatenate([np.random.random(3), [0.50]])
                _img[masks[indices[0][i][2]]] = np.concatenate([np.random.random(3), [0.30]])

                ax.imshow(_img)
                ax.set_title(predicted.split(' ')[i])
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
                
        if save_iamge:
            plt.savefig(save_adress+str(index)+'.png', bbox_inches='tight')
        
        if show_iamge:
            # Show the figure
            plt.show()
import  torch
import  utils
import  torch.nn                    as      nn
from    torchvision.models          import  resnet50, ResNet50_Weights
from    torchvision.transforms      import  Resize

class ResNetFeatureExtractor_proposedStructuredPooling(nn.Module):
    """
    ResNet-based feature extractor module for proposed structured pooling.

    Inherits from nn.Module.

    """
    def __init__(self,
                 cut    :int= utils.resnet_Cut,
                 resize :int= utils.resize_ROI):
        
        super(ResNetFeatureExtractor_proposedStructuredPooling, self).__init__()
        """
        Initializes an instance of ResNetFeatureExtractor_proposedStructuredPooling.

        Args:
            cut (int): The number of layers to include in the feature extractor.
            resize (int): The size to resize the region of interest (ROI).

        """

        self.resize_ROI = resize
        # Load the pre-trained ResNet50 model
        self.resnet  = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        # Remove the classification layer (the last fully connected layer)
        # and pooling layer
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:cut])
        
        self.fc1 = nn.Linear(utils.resnet_OutPut,utils.decoder_hiddenState)
        self.fc2 = nn.Linear(utils.resnet_OutPut,utils.decoder_hiddenState)
        
        self.resize_transform = Resize((self.resize_ROI,self.resize_ROI),antialias=True)
        
        if utils.fine_tune_resnet:
            for name, param in self.resnet.named_parameters():
                if name.startswith('0') or name.startswith('1') or name.startswith('4'):
                    param.requires_grad = False
        else :
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def forward(self,
                Images,
                Masks,
                ):
        """
        Performs a forward pass of the ResNetFeatureExtractor_proposedStructuredPooling.
        --------------------------------------------------------------------------------
        in matrix multiplication part 
            tensor_a = torch.randn(32,  100 ,1      ,7  ,7) # ROI
            tensor_b = torch.randn(32,  1   ,2048   ,7  ,7) # feature sform resnet
        --------------------------------------------------------------------------------
        Args:
            Images (torch.Tensor): The input images.
            Masks (torch.Tensor): The input masks.

        
        Returns:
            torch.Tensor: The structured attention outputs.
            torch.Tensor: The initial hidden state.
            torch.Tensor: The initial cell state.

        """
        
        self.resnet.eval()
        F             = self.resnet(Images)
        R_prime       = self.resize_transform(Masks)

        S = torch.matmul(R_prime.unsqueeze(2), F.unsqueeze(1)).sum(dim=-1).sum(dim=-1)/(self.resize_ROI*self.resize_ROI)

        h_0 = self.fc1(S.mean(dim=1))
        c_0 = self.fc2(S.mean(dim=1))
        
        return S,h_0,c_0
    
    def check(self,):
        """
        Prints the requires_grad attribute for each parameter in the ResNet model.

        """
        for name, param in self.resnet.named_parameters():
            print(name, param.requires_grad)
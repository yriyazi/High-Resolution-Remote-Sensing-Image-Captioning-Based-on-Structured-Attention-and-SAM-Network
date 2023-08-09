import torch
import torch.nn as nn
import utils

class Attention_Module(nn.Module):
    """
    Attention module for structured attention.

    Inherits from nn.Module.

    """
    def __init__(self,
                 intermediate_weight    :int,
                 size_S                 :int,
                 size_h                 :int):
        """
        Initializes an instance of Attention_Module.

        Args:
            intermediate_weight (int): The size of the intermediate weight.
            size_S (int): The size of S.
            size_h (int): The size of h.

        """
        super(Attention_Module, self).__init__()
        
        self.FC1    = nn.Linear(in_features=size_S, out_features=intermediate_weight)
        self.FC2    = nn.Linear(in_features=size_h, out_features=intermediate_weight)
        self.FC3    = nn.Linear(in_features=intermediate_weight, out_features=1)
        self.relu   = nn.ReLU()

    def forward(self,
                Si      :torch.Tensor,
                ht_1    :torch.Tensor,):
        """
        Performs a forward pass of the Attention_Module.

        Args:
            Si (torch.Tensor): The input Si.
            ht_1 (torch.Tensor): The input ht_1.

        Returns:
            torch.Tensor: The output.

        """
        out_FC1 = self.FC1(Si)
        out_FC2 = self.FC2(ht_1)
        fused_feature = self.relu(out_FC1 + out_FC2)
        output = self.FC3(fused_feature)
        return output


class Structured_Attention(nn.Module):
    """
    Structured attention module.

    Inherits from nn.Module.

    """

    def __init__   (self,
                    intermediate_weight     :int = utils.intermediate_weight,
                    size_S                  :int = utils.resnet_OutPut,
                    size_h                  :int = utils.decoder_hiddenState,
                    device                  :str = 'cuda'):

        super(Structured_Attention, self).__init__()
        """
        Initializes an instance of Structured_Attention.

        Args:
            intermediate_weight (int): The size of the intermediate weight.
            size_S (int): The size of S.
            size_h (int): The size of h.
            device (str): The device to be used for computation.

        """
        self.device =device
        self.Attention_Module = Attention_Module(intermediate_weight,size_S, size_h).to(device=device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                S       :torch.Tensor,
                h_t_1   :torch.Tensor):
        """
        Performs a forward pass of the Structured_Attention.

        Args:
            S (torch.Tensor): The input S.
            h_t_1 (torch.Tensor): The input h_t_1.

        Returns:
            torch.Tensor: The structured attention outputs.
            torch.Tensor: The attention weights.

        """
        N = S.shape[1]
        batch_size = S.shape[0]
        a_t = torch.zeros(batch_size, N).to(device=self.device)
        for i in range(N):
            a_t[:, i] = self.Attention_Module.forward(S[:, i, :], h_t_1).squeeze(1)
        a_t = self.softmax(a_t)
        z_t = torch.zeros(batch_size, S.shape[2]).to(device=self.device)
        for i in range(N):
            z_t += S[:, i, :] * a_t[:, i].unsqueeze(1).expand_as(S[:, i, :])

        return z_t,a_t

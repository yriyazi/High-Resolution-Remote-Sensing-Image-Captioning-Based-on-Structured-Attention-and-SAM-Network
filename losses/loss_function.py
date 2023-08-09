import torch
import torch.nn as nn
import utils

class custum_loss(nn.Module):
    """
    Custom loss module for a neural network model.

    Inherits from nn.Module.

    """
    def __init__(self,)-> None:
        """
        Initializes an instance of custum_loss.

        Initializes the base loss as CrossEntropyLoss.

        """
        super(custum_loss, self).__init__()
        self.base = nn.CrossEntropyLoss()
    
    def DSR(self,
            attention       :torch.Tensor   ) -> torch.Tensor:
        """
        Calculates the Doubly Stochastic Regularization (DSR) loss term.

        Args:
            attention (torch.Tensor): The attention tensor.

        Returns:
            torch.Tensor: The DSR loss term.

        """
        return torch.pow(1 - attention.sum(dim=1), 2).sum(dim=1).mean(dim=0)

    def AVR(self,
            attention       :torch.Tensor,
            Region_count    :int            ) -> torch.Tensor:
        """
        Calculates the Attention Variance Regularization (AVR) loss term.

        Args:
            attention (torch.Tensor): The attention tensor.
            Region_count (int): The number of regions.

        Returns:
            torch.Tensor: The AVR loss term.

        """
        # torch.norm(a_t, p=2, dim=2, keepdim=False)
        return (torch.linalg.norm((attention- (1/Region_count)),dim=2, ord=2)).sum(dim=1).mean()
    
    def forward(self, 
                predicted   :torch.Tensor,
                ground_truth:torch.Tensor,
                attention   :torch.Tensor,
                Region_count:int            ) -> torch.Tensor:
        """
        Computes the overall loss.

        Args:
            predicted (torch.Tensor): The predicted outputs of the model.
            ground_truth (torch.Tensor): The ground truth labels.
            attention (torch.Tensor): The attention tensor.
            Region_count (int): The number of regions.

        Returns:
            torch.Tensor: The overall loss.

        """
        loss = self.base(predicted,ground_truth)
        if utils.loss_DSR:
            loss += self.DSR(attention)*utils.loss_Beta
        if utils.loss_AVR:
            loss += self.AVR(attention,Region_count)*utils.loss_Gamma
        return loss

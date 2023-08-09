import torch
import dataloaders
import utils
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder module for a sequence-to-sequence model.

    Inherits from nn.Module.

    """
    def __init__(self,
                hidden_size      :int = utils.decoder_hiddenState,
                vocab_size       :int = dataloaders.source.n_words,
                num_layers       :int = utils.decoder_num_layer,
                y_t_outputSize   :int = utils.word_score,
                featureSize      :int = utils.resnet_OutPut):
        """
        Initializes an instance of Decoder.

        Args:
            hidden_size (int)   : The hidden size of the LSTM layers.
            vocab_size (int)    : The size of the vocabulary.
            num_layers (int)    : The number of LSTM layers.
            y_t_outputSize (int): The output size of y_t.
            featureSize (int)   : The size of the input features.

        """
        super(Decoder, self).__init__()

        self.num_layers      = num_layers
        self.hidden_size     = hidden_size
        self.vocab_size      = vocab_size
        self.y_t_outputSize  = y_t_outputSize
        self.featureSize     = featureSize
        self.embed_size      = self.hidden_size + self.featureSize + self.y_t_outputSize 

        self.embed  = nn.Embedding(self.vocab_size, utils.decoder_embbeding_size)
        self.lstm   = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True)
        
        self.L_o    = nn.Linear(self.y_t_outputSize, self.vocab_size)
        self.L_h    = nn.Linear(hidden_size, self.y_t_outputSize)
        self.L_y    = nn.Linear(self.y_t_outputSize, self.y_t_outputSize)
        self.L_z    = nn.Linear(self.featureSize, self.y_t_outputSize)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the embedding weights.

        """
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self,
                y_t_1  :torch.Tensor,
                z_t    :torch.Tensor,
                h_t_1  :torch.Tensor,
                c_t_1  :torch.Tensor):
            """
            Performs a forward pass of the decoder.

            Args:
                y_t_1 (torch.Tensor): The previous word token.
                z_t (torch.Tensor): The input features.
                h_t_1 (torch.Tensor): The previous hidden state.
                c_t_1 (torch.Tensor): The previous cell state.

            Returns:
                torch.Tensor: The predicted word token.
                torch.Tensor: The current hidden state.
                torch.Tensor: The current cell state.

            """
            # y_t_1 is one word token i.e. torch.tensor(12)

            Py_t_1        = self.embed(y_t_1)
            x_t           = torch.cat((h_t_1, Py_t_1, z_t), dim=1).unsqueeze(1)

            _ , (h_t,c_t) = self.lstm(x_t,(h_t_1.unsqueeze(0),c_t_1.unsqueeze(0)))
            y_t           = self.L_o(self.L_h(h_t.squeeze(0))+self.L_y(Py_t_1)+self.L_z(z_t))

            return y_t, h_t.squeeze(0), c_t.squeeze(0)
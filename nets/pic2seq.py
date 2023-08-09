import  dataloaders
import  random
import  nets
import  torch
import  torch.nn        as  nn
import  utils


class pic2Seq(nn.Module):
    """
    pic2Seq model for image captioning.

    Inherits from nn.Module.

    """
    def __init__(self,
                 embeddings_dim     :int = 3072,
                 hidden_Dimention   :int = 512,
                 Num_layer = 1,
                 device:str='cuda'):
        """
        Initializes an instance of pic2Seq.

        Args:
            embeddings_dim (int): The dimensionality of the image embeddings.
            hidden_Dimention (int): The hidden dimensionality of the LSTM layers.
            Num_layer (int): The number of LSTM layers.
            device (str): The device to be used for computation.

        """
        super(pic2Seq, self).__init__()
        self.device = device
        self.res50                  = nets.ResNetFeatureExtractor_proposedStructuredPooling().to(device=self.device)
        self.Structured_Attention   = nets.Structured_Attention(device = self.device).to(device=self.device)
        self.decoder                = nets.Decoder().to(device=self.device)
        
    def forward(self, 
                pictures_captions           :torch.Tensor,
                image                       :torch.Tensor,
                mask                        :torch.Tensor,
                teacher_forcing_ratio                       =utils.teacher_forcing):
        """
        Performs a forward pass of the pic2Seq model.

        Args:
            pictures_captions (torch.Tensor): The input image captions.
            image (torch.Tensor): The input image.
            mask (torch.Tensor): The input mask.
            teacher_forcing_ratio (float): The probability of using teacher forcing during training.

        Returns:
            torch.Tensor: The decoder outputs.
            torch.Tensor: The predicted captions.
            torch.Tensor: The attention outputs.

        """
        batch_size          = pictures_captions.shape[0]
        lenght_sentences    = pictures_captions.shape[1]

        S,h_t,c_t   = self.res50(image,mask)
        y_t         = pictures_captions[:,0]

        predicted           = torch.ones( size=[batch_size,lenght_sentences                            ], device=self.device)
        decoder_outputs     = torch.zeros(size=[batch_size,lenght_sentences, dataloaders.source.n_words], device=self.device)
        attention_outputs   = torch.zeros(size=[batch_size,lenght_sentences, utils.N_ROI               ], device=self.device)

        for i in range(lenght_sentences):
            
            z_t,a_t        = self.Structured_Attention.forward(S,h_t)
            attention_outputs[:,i,:] = a_t
            temp, h_t, c_t = self.decoder.forward(   y_t_1 = y_t,
                                                    z_t   = z_t,
                                                    h_t_1 = h_t,
                                                    c_t_1 = c_t)
            decoder_outputs[:,i,:] = temp
            y_t = temp.argmax(1)
            predicted[:,i] = y_t

            # teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            y_t = pictures_captions[:,i] if teacher_force else y_t

        return decoder_outputs,predicted,attention_outputs

    def wordPredict(self,
                    S       :torch.Tensor,
                    candid  :torch.Tensor,
                    h_t     :torch.Tensor,
                    c_t     :torch.Tensor,):
        """
        Predicts the next word given the current state.

        Args:
            S (torch.Tensor): The input image features.
            candid (torch.Tensor): The current word token.
            h_t (torch.Tensor): The current hidden state.
            c_t (torch.Tensor): The current cell state.

        Returns:
            torch.Tensor: The predicted word score.
            torch.Tensor: The updated hidden state.
            torch.Tensor: The updated cell state.

        """
        
        z_t,_        = self.Structured_Attention.forward(S,h_t)
        score, h_t, c_t = self.decoder.forward( y_t_1 = candid,
                                                z_t   = z_t,
                                                h_t_1 = h_t,
                                                c_t_1 = c_t)
        return score, h_t , c_t
            
    def beam_search(self,
                image,
                mask,
                pictures_captions,
                beam_width = utils.Beam_width,
                ):
        """
        Performs beam search decoding to generate captions for the given image.

        Args:
            image (torch.Tensor): The input image.
            mask (torch.Tensor): The input mask.
            pictures_captions (torch.Tensor): The input image captions.
            beam_width (int): The width of the beam search.

        Returns:
            list: A list of beams containing the caption sequences and their scores.

        """

        with torch.no_grad():
            lenght_sentences    = pictures_captions.shape[1]
            #Encoder neural net
            S,h_t, c_t          = self.res50(image,mask)
            y_t                 = pictures_captions[:,0]

            # Initialize the initial beam
            beams = [([y_t], h_t , c_t , 0)]
            keep = []
            # keep = torch.zeros(size=[1,lenght_sentences,beam_width,beam_width])
            # Decoder and attention
            for word in range(lenght_sentences):
                # Expand each beam
                new_beams = []
                
                for beam in beams:
                    
                    # temp_sent.append(y_t)
                    y_t, h_t , c_t , score = beam
                    scores_t, h_t , c_t = self.wordPredict(S    = S,
                                                        candid  = y_t[-1],
                                                        h_t     = h_t,
                                                        c_t     = c_t)
                    # Apply log softmax to get log probabilities
                    log_probs = torch.log_softmax(scores_t, dim=1)

                    # Get the top k candidates
                    topk_probs, topk_indices = torch.topk(log_probs, beam_width)

                    # Create new beams for each candidate
                    for i in range(beam_width):
                        tt_Temp =[]
                        tt_Temp=y_t.copy()
                        new_decoder_input = topk_indices[:, i]
                        
                        # tt_Temp.append(y_t)
                        tt_Temp.append(new_decoder_input)
                        
                        new_score = score + topk_probs[:, i]
                        new_beams.append((tt_Temp, h_t , c_t, new_score))
                        
                        tt_Temp =[]

                # Sort the beams by score
                new_beams.sort(key=lambda x: x[3], reverse=True)

                # Select top k beams
                beams = new_beams[:beam_width]
                keep.extend(new_beams[0][0])
            # Select the best beam as the final output
            # *_, best_score = beams[0]
            # best_sequence = beams[0][0].squeeze().tolist()

            return beams
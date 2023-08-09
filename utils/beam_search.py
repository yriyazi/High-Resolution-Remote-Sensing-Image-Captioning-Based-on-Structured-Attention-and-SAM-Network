import torch
import nets,dataloaders


class beamvis():
    def __init__(self,
                 model_adress:str,
                 device = 'cuda' ) -> None:
        self.device = device
        self.Pic2Seq = nets.pic2Seq()
        self.Pic2Seq.load_state_dict(torch.load(model_adress))
        self.Pic2Seq.eval()
        
        self.dataset = dataloaders.test_dataset
    
    def __data(self,
               index:int):
        
        image,mask,padded_sequence  = self.dataset[index]
        image                       = image.unsqueeze(0).to( self.device)
        mask                        = mask.unsqueeze(0).to( self.device)
        padded_sequence             = padded_sequence.unsqueeze(0).to( self.device)
        return image,mask,padded_sequence
    
    def ind2word(self,input):
        temp = torch.tensor(input).clone().detach()[torch.tensor(input)!=0 ]
        if temp[2]==temp[1]:
            temp[1]=1
    
        if temp[1]==temp[0]:
            temp[1]=1
            
        temp = temp[temp!=1]
        temp = temp[temp!=2]
        
        return " ".join([dataloaders.source.index2word[item.item()] for item in temp])

         
    def beam(self,
             index:int,
             ):
        with torch.inference_mode():
            image,mask,padded_sequence = self.__data(index)
            
            self.sbeams = self.Pic2Seq.beam_search(image = image,
                        mask = mask,
                        pictures_captions = padded_sequence,
                        )
            sent = []
            for data in (self.sbeams):
                y_t, *_ = data
                sent.append(self.ind2word(y_t))

            return sent
    
    def prediction(self,
                   index:int):
        with torch.inference_mode():
            image,mask,padded_sequence = self.__data(index)
            decoder_outputs,predicted,attention_outputs = self.Pic2Seq.forward(image = image,
                                                                            mask = mask,
                                                                            pictures_captions = padded_sequence,)
            return decoder_outputs,self.ind2word(predicted),attention_outputs
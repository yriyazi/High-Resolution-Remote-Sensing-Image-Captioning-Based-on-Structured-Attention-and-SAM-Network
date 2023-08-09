import  torch
import  utils
# utils.set_seed(42)
import  os ,tqdm
import  nets, dataloaders , deeplearning
import  numpy                               as      np
import  matplotlib.pyplot                   as      plt
import  torchvision.models                  as      models
import  torchvision.transforms              as      transforms
from    torch.utils.data                    import  DataLoader
from    nltk.translate.bleu_score           import  sentence_bleu, SmoothingFunction
chencherry = SmoothingFunction()
#%%
directory_path      = r"models\UCM-WarmUped\loss ablation\DSR=True AVR=False FT_Resnet=False N_ROI=8"
model               = r"DSR=True AVR=False FT_Resnet=False N_ROI=8_020_025_valid_acc 0.5312788486480713.pt"
#%%
test_dataloader = DataLoader(dataloaders.test_dataset, batch_size=32, shuffle=True,collate_fn=dataloaders.collate_fn)
Pic2Seq = nets.pic2Seq()
Pic2Seq.load_state_dict(torch.load(os.path.join(directory_path,model)))
Pic2Seq.eval()
scores          = [0,0,0,0]
count_failed    = [0,0,0,0]
count_sussec    = [0,0,0,0]
senteses = []
senteses_groundTruth = []
def ind2word(input):
    temp = (input).clone().detach()[input!=0 ]
    temp = temp[temp!=1]
    temp = temp[temp!=2]
    return " ".join([dataloaders.source.index2word[item.item()] for item in temp])
#%%
loop = tqdm.tqdm(
                            enumerate(test_dataloader, 1),
                            total=len(test_dataloader),
                            desc="BLEU",
                            position=0,
                            leave=True
                        )

#%%
with torch.inference_mode():
    for batch_idx,  (image,mask,padded_sequence) in loop:
        image = image.to('cuda')
        mask = mask.to('cuda')
        padded_sequence = padded_sequence.to('cuda')

        sent = []
        for i in range(len(image)):
            sbeams = Pic2Seq.beam_search(image = image[i].unsqueeze(0),
                        mask = mask[i].unsqueeze(0),
                        pictures_captions = padded_sequence[i].unsqueeze(0))
            for data in (sbeams):
                y_t, *_ = data
                TTarget = torch.tensor(y_t).clone().detach()[torch.tensor(y_t)!=0 ]
                if TTarget[2]==TTarget[1]:
                    TTarget[1]=1
                TTarget = TTarget[TTarget!=1]
                TTarget = TTarget[TTarget!=2]
                sent.append(ind2word(TTarget))
                break
            
        for index in range(padded_sequence.shape[0]):
            Ground  = padded_sequence[index,:][padded_sequence[index,:] != 0][1:-1]
            Ground  = [f"{dataloaders.source.index2word[int(word.to('cpu'))]} " for word in Ground]
                
            try:
                scores[0] += sentence_bleu(["".join(Ground),],sent[index],weights = (1,0,0,0),smoothing_function=chencherry.method4)
                count_sussec[0] +=1
            except:
                count_failed[0] += 1
                
            try:
                scores[1] += sentence_bleu(["".join(Ground),],sent[index],weights = (0.5,0.5,0,0),smoothing_function=chencherry.method4)
                count_sussec[1] +=1
            except:
                count_failed[1] += 1
            
            try:
                scores[2] += sentence_bleu(["".join(Ground),],sent[index],weights = (0.333,0.333,0.334,0),smoothing_function=chencherry.method4)
                count_sussec[2] +=1
            except:
                count_failed[2] += 1
                
            try:
                scores[3] += sentence_bleu(["".join(Ground),],sent[index],weights = (0.25,0.25,0.25,0.25),smoothing_function=chencherry.method4)
                count_sussec[3] +=1
            except:
                count_failed[3] += 1


            loop.set_description(f"iteration : {batch_idx}")
            loop.set_postfix(
                BLEU_1 = f"{(np.array(scores)/np.array(count_sussec))[0]:.4f}",
                BLEU_2 = f"{(np.array(scores)/np.array(count_sussec))[1]:.4f}",
                BLEU_3 = f"{(np.array(scores)/np.array(count_sussec))[2]:.4f}",
                BLEU_4 = f"{(np.array(scores)/np.array(count_sussec))[3]:.4f}",
                refresh=True,
            )

    print(np.array(scores)/np.array(count_sussec))

import  os
import  torch
import  utils
import  time
import  torch.nn    as      nn
import  pandas      as      pd
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm

    
class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def normal_accuracy(pred,labels):    
    return ((pred.argmax(dim=1)==labels).sum()/len(labels))*100

def teacher_forcing_decay(epoch, num_epochs):
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.01
    decay_rate = (final_tf_ratio / initial_tf_ratio) ** (1 / (num_epochs - 1))

    tf_ratio = max(0.01,initial_tf_ratio * (decay_rate ** epoch))
    return tf_ratio



def train(
    train_loader:torch.utils.data.DataLoader,
    val_loader  :torch.utils.data.DataLoader,
    model       :torch.nn.Module,
    model_name  :str,
    epochs      :int,
    load_saved_model    :bool,
    ckpt_save_freq      :int ,
    ckpt_save_path      :str ,
    ckpt_path           :str ,
    report_path         :str ,
    
    criterion ,
    optimizer,
    lr_schedulerr,
    sleep_time,
    Validation_save_threshold : float ,
    
    tets_loader     :torch.utils.data.DataLoader,
    test_evaluate   :bool                           = False     ,
    device          :str                            = 'cuda'    ,
    
    Teacher_forcing_train       = True  ,
    Teacher_forcing_num_epochs  = 10    ,
    ):

    model       = model.to(device)

    if load_saved_model:
        model, optimizer = load_model(
                                      ckpt_path=ckpt_path, model=model, optimizer=optimizer
                                        )

    

    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_acc_till_current_batch"])
    
    max_Accu_validation_previous = 0
    
    for epoch in tqdm(range(1, epochs + 1)):
        acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"
        
        
        loop_train = tqdm(
                            enumerate(train_loader, 1),
                            total=len(train_loader),
                            desc="train",
                            position=0,
                            leave=True
                        )
        accuracy_dum=[]
        for batch_idx, (image,mask,padded_sequence) in loop_train:
            image = image.to('cuda')
            mask = mask.to('cuda')
            padded_sequence = padded_sequence.to('cuda')
           
            length = padded_sequence.size(1)
            
            if Teacher_forcing_train :
                _teacher_forcing = teacher_forcing_decay(epoch,Teacher_forcing_num_epochs)
            else :
                _teacher_forcing = utils.teacher_forcing
        
            decoder_outputs,predicted,attention_outputs = model(   pictures_captions = padded_sequence,
                                                            image = image,
                                                            mask = mask,
                                                            teacher_forcing_ratio = _teacher_forcing)
            ############
            # permute
            predicted = predicted.permute(dims = [1,0])
            padded_sequence = padded_sequence.permute(dims = [1,0])     # trg = [trg len, batch size]
            decoder_outputs = decoder_outputs.permute(dims = [1,0,2])   # output = [trg len, batch size, output dim]
            # return decoder_outputs,padded_sequence,predicted
        
            output_dim = decoder_outputs.shape[-1]

            decoder_outputs = decoder_outputs[1:].contiguous().view(-1, output_dim)
            padded_sequence = padded_sequence[1:].contiguous().view(-1)
            predicted = predicted[1:].contiguous().view(-1)

            loss = criterion.forward(   predicted = decoder_outputs,
                                        ground_truth = padded_sequence.to(torch.long),
                                        attention = attention_outputs,
                                        Region_count = utils.N_ROI)
                
            
            optimizer.zero_grad()   
            loss.backward()
                     
            # gradient clipping
            # max_grad_norm = 1.0
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
            optimizer.step()

            correct = 0
            num_word = torch.count_nonzero(padded_sequence)
            for i in range(len(padded_sequence)):
                if padded_sequence[i] != 0 and padded_sequence[i] == predicted[i]:
                    correct += 1
            acc1 = correct / num_word

            accuracy_dum.append(acc1)
            acc1 = sum(accuracy_dum)/len(accuracy_dum)
            
            loss_avg_train.update(loss.item(), length)
            
            # if batch_idx%sleep_time==0:
            #     time.sleep(1)
            
            
            
            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "image_type":"original",
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size":length,
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_train_acc_till_current_batch":acc1,
                 "avg_val_loss_till_current_batch":None,
                 "avg_val_acc_till_current_batch":None},index=[0])

            
            report.loc[len(report)] = new_row.values[0]
            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                accuracy_train="{:.4f}".format(acc1),
                refresh=True,
            )
        # time.sleep(3)
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )
        if utils.scheduler_activate:    
            lr_schedulerr.step()
            
        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            acc1 = 0
            total = 0
            accuracy_dum=[]
            for batch_idx, (image,mask,padded_sequence) in loop_val:
                image = image.to('cuda')
                mask = mask.to('cuda')
                padded_sequence = padded_sequence.to('cuda')
                length = padded_sequence.size(1)

                decoder_outputs,predicted,attention_outputs = model(   pictures_captions = padded_sequence,
                                            image = image,
                                            mask = mask)
                ############
                # permute
                predicted = predicted.permute(dims = [1,0])
                padded_sequence = padded_sequence.permute(dims = [1,0])     # trg = [trg len, batch size]
                decoder_outputs = decoder_outputs.permute(dims = [1,0,2])   # output = [trg len, batch size, output dim]
                # return decoder_outputs,padded_sequence,predicted
            
                output_dim = decoder_outputs.shape[-1]

                decoder_outputs = decoder_outputs[1:].contiguous().view(-1, output_dim)
                padded_sequence = padded_sequence[1:].contiguous().view(-1)
                predicted = predicted[1:].contiguous().view(-1)

                loss = criterion.forward(   predicted = decoder_outputs,
                                        ground_truth = padded_sequence.to(torch.long),
                                        attention = attention_outputs,
                                        Region_count = utils.N_ROI)
                
                num_word = torch.count_nonzero(padded_sequence)

                correct = 0
                for i in range(len(padded_sequence)):
                    if padded_sequence[i] != 0 and padded_sequence[i] == predicted[i]:
                        correct += 1
                acc1 = correct / num_word

                accuracy_dum.append(acc1)
                acc1 = sum(accuracy_dum)/len(accuracy_dum)

                loss_avg_val.update(loss.item(),length)
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "image_type":"original",
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": length,
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_acc_till_current_batch":acc1},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    accuracy_val="{:.4f}".format(acc1),
                    refresh=True,
                )
                
            max_Accu_validation = acc1
            if max_Accu_validation>Validation_save_threshold and max_Accu_validation>max_Accu_validation_previous:
                torch.save(model.state_dict(), f"{report_path}/{model_name}_valid_acc {acc1}.pt")

        if test_evaluate==True:
            mode = "test"
            with torch.no_grad():
                loop_val = tqdm(
                                enumerate(tets_loader, 1),
                                total=len(tets_loader),
                                desc="test",
                                position=0,
                                leave=True,
                                )
                accuracy_dum=[]

                for batch_idx, (image,mask,padded_sequence) in loop_val:
                    image = image.to('cuda')
                    mask = mask.to('cuda')
                    padded_sequence = padded_sequence.to('cuda')
                    length = padded_sequence.size(1)
                    decoder_outputs,predicted,attention_outputs = model(   pictures_captions = padded_sequence,
                                                image = image,
                                                mask = mask)
                    ############
                    # permute
                    predicted = predicted.permute(dims = [1,0])
                    padded_sequence = padded_sequence.permute(dims = [1,0])     # trg = [trg len, batch size]
                    decoder_outputs = decoder_outputs.permute(dims = [1,0,2])   # output = [trg len, batch size, output dim]
                    # return decoder_outputs,padded_sequence,predicted
                    output_dim = decoder_outputs.shape[-1]
                    decoder_outputs = decoder_outputs[1:].contiguous().view(-1, output_dim)
                    padded_sequence = padded_sequence[1:].contiguous().view(-1)
                    predicted = predicted[1:].contiguous().view(-1)

                    loss = criterion.forward(   predicted = decoder_outputs,
                                                ground_truth = padded_sequence.to(torch.long),
                                                attention = attention_outputs,
                                                Region_count = utils.N_ROI)
                    
                    num_word = torch.count_nonzero(padded_sequence)

                    correct = 0
                    for i in range(len(padded_sequence)):
                        if padded_sequence[i] != 0 and padded_sequence[i] == predicted[i]:
                            correct += 1
                    acc1 = correct / num_word
            
                    loss_avg_val.update(loss.item(),length)
                    new_row = pd.DataFrame(
                        {"model_name": model_name,
                        "mode": mode,
                        "image_type":"original",
                        "epoch": epoch,
                        "learning_rate":optimizer.param_groups[0]["lr"],
                        "batch_size": length,
                        "batch_index": batch_idx,
                        "loss_batch": loss.detach().item(),
                        "avg_train_loss_till_current_batch":None,
                        "avg_train_acc_till_current_batch":None,
                        "avg_val_loss_till_current_batch":loss_avg_val.avg,
                        "avg_val_acc_till_current_batch":acc1},index=[0],)
                    
                    report.loc[len(report)] = new_row.values[0]
                    loop_val.set_description(f"test - iteration : {epoch}")
                    loop_val.set_postfix(
                        loss_batch="{:.4f}".format(loss.detach().item()),
                        avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                        accuracy_val="{:.4f}".format(acc1),
                        refresh=True,
                    )    
            
        report.to_csv(os.path.join(report_path,f"{model_name}_report.csv"))
    torch.save(model.state_dict(), os.path.join(report_path,f"{model_name}.pt"))
    return model, optimizer, report
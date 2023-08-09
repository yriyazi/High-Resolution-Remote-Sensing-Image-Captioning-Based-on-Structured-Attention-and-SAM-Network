import  torch
import  utils
import losses
utils.set_seed(42)
import  os
import torch.nn as nn
import  nets, dataloaders , deeplearning
import  torchvision.models          as      models
import  torchvision.transforms      as      transforms
from    torch.utils.data            import  DataLoader

import cProfile
import pstats


if __name__ == "__main__":
    with cProfile.Profile() as profile:




        #%%

        train_dataloader = DataLoader(dataloaders.tarin_dataset, batch_size=32, shuffle=True,collate_fn=dataloaders.collate_fn,num_workers=5)
        valid_dataloader = DataLoader(dataloaders.valid_dataset, batch_size=32, shuffle=True,collate_fn=dataloaders.collate_fn,num_workers=5)
    
        #%%
        loss = losses.custum_loss()
        Pic2Seq = nets.pic2Seq()

        # Specify different learning rates for different layers
        learning_rates = {
            'resnet': 1e-5,
            'Fc'    : 1e-3,
        }

        # Group the parameters based on their names and learning rates
        if utils.fine_tune_resnet:
            param_groups = [
                {'params': Pic2Seq.decoder.parameters()                 , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.Structured_Attention.parameters()    , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.res50.fc1.parameters()               , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.res50.fc2.parameters()               , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.res50.resnet.parameters()            , 'lr': learning_rates['resnet']},
            ]
        else:
            param_groups = [
                {'params': Pic2Seq.decoder.parameters()                 , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.Structured_Attention.parameters()    , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.res50.fc1.parameters()               , 'lr': learning_rates['Fc']    },
                {'params': Pic2Seq.res50.fc2.parameters()               , 'lr': learning_rates['Fc']    },
            ]

        optimizer = torch.optim.Adam(param_groups)

        lr_schedulerr = torch.optim.lr_scheduler.StepLR(optimizer   = optimizer,
                                                        step_size = utils.total_iters,
                                                        gamma     = utils.gamma,
                                                        )
        #%%
        ######### change variale here to train
        epoch_start = 1
        epoch_end   = 2
        #%%
        ################################################
        # Specify the folder path
        model_name = f'DSR={utils.loss_DSR} AVR={utils.loss_AVR} FT_Resnet={utils.fine_tune_resnet} N_ROI={utils.N_ROI} {utils.loss_Beta=} {utils.loss_Gamma=}'
        folder_path = os.path.join('models',
                                f'{utils.DatasetInUse}',
                                model_name)

        folder_ckptpath = os.path.join(folder_path,'checkpoints')
        # Check if the folder exists
        if not os.path.exists(folder_ckptpath):
            # Create the folder if it doesn't exist
            os.makedirs(folder_ckptpath)
            print(f"Folder '{folder_ckptpath}' created successfully.")
        else:
            print(f"Folder '{folder_ckptpath}' already exists.")
        ################################################
        model, optimizer, report = deeplearning.train(
                                                        train_loader = train_dataloader,
                                                        val_loader   = valid_dataloader,
                                                        model        = Pic2Seq,
                                                        model_name   = f'{model_name}_{epoch_start:03}_{epoch_end:03}',
                                                        epochs       = epoch_end - epoch_start,
                                                        load_saved_model   =False,
                                                        ckpt_save_freq     =1 ,
                                                        ckpt_save_path     = os.path.join(folder_ckptpath)  ,
                                                        ckpt_path          = os.path.join(folder_ckptpath,r'ckpt_DSR=True AVR=True FT_Resnet=False N_ROI=8 utils.loss_Beta=1 utils.loss_Gamma=1_005_010_epoch5.ckpt') ,
                                                        report_path        = os.path.join(folder_path)      ,
                                                        
                                                        criterion         = loss,
                                                        optimizer         = optimizer,
                                                        lr_schedulerr     = lr_schedulerr,
                                                        sleep_time        = 5,
                                                        Validation_save_threshold = 0.50 ,
                                                        
                                                        tets_loader         = None,
                                                        test_evaluate       = False     ,
                                                        device              = 'cuda'    ,
                                                        Teacher_forcing_train = False)

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.dump_stats("profiling.prof")
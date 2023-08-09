import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
inference_mode      = config['inference_mode']
learning_rate       = config['learning_rate']
num_epochs          = config['num_epochs']
seed                = config['seed']
ckpt_save_freq      = config['ckpt_save_freq']


batch_train_size        = config['batch']['batch_train_size']
Validation_Set          = config['batch']['Validation_Set']
batch_validation_size   = config['batch']['batch_validation_size']
batch_test_size         = config['batch']['batch_test_size']

# Access dataset parameters
sorted_mask         = config['dataset']['sorted_mask']
DatasetInUse        = config['dataset']['DatasetInUse']
rand_sent           = config['dataset']['rand_sent']
dataframe           = config['dataset']['dataframe']

#loss
loss_DSR          = config['loss']['DSR']
loss_AVR          = config['loss']['AVR']
loss_Beta         = config['loss']['Beta']
loss_Gamma        = config['loss']['Gamma']

# Access model architecture parameters
model_name          = config['model']['name']
    # structure pooling
resize_ROI          = config['model']['resize_ROI']
N_ROI               = config['model']['N_ROI']
    # resnet
fine_tune_resnet    = config['model']['fine_tune_resnet']
resnet_Cut          = config['model']['resnet_Cut']
resnet_OutPut       = config['model']['resnet_output']
decoder_embbeding_size  = config['model']['decoder_embbeding_size']
    # decoder
decoder_hiddenState     = config['model']['decoder_hiddenState']
word_score              = config['model']['word_score']
decoder_num_layer       = config['model']['decoder_num_layer']
    # attention later
intermediate_weight     = config['model']['intermediate_weight']
teacher_forcing         = config['model']['teacher_forcing']
    #train parameters
Beam_width              = config['model']['Beam_width']

# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_betas           = config['optimizer']['betas']
# Access scheduler parameters
scheduler_name          = config['scheduler']['name']
scheduler_activate      = config['scheduler']['scheduler_activate']
gamma                   = config['scheduler']['gamma']
total_iters             = config['scheduler']['total_iters']


# print("configuration hass been loaded!!! \n successfully")
# print(learning_rate)
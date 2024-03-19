import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# output settings
channel_input   = config['channel_input']
dim_input       = config['dim_input']
# model settings
model_type      = config['model_type']
device          = config['device']
# Access hyperparameters
points_per_side                 = config['points_per_side']
pred_iou_thresh                 = config['pred_iou_thresh']
stability_score_thresh          = config['stability_score_thresh']
crop_n_layers                   = config['crop_n_layers']
crop_n_points_downscale_factor  = config['crop_n_points_downscale_factor']
min_mask_region_area            = config['min_mask_region_area']
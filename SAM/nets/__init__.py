import  os
import  utils
import  urllib.request
from    tqdm                import tqdm
from    segment_anything    import sam_model_registry, SamAutomaticMaskGenerator


model_path = os.path.join('SAM_PreTrained', 'sam_vit_b_01ec64.pth')
download_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

if not os.path.exists(model_path):
    print("Model file doesn't exist. Downloading...")
    try:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=download_url.split('/')[-1]) as t:
            urllib.request.urlretrieve(download_url, model_path, reporthook=lambda blocknum, blocksize, totalsize: t.update(blocksize))
        print("Download completed successfully.")
    except Exception as e:
        print("Error occurred while downloading the model:", str(e))
else:
    print("Model file already exists.")


sam_checkpoint  = os.path.join('SAM_PreTrained', 'sam_vit_b_01ec64.pth')
model_type      = utils.model_type
device          = utils.device
sam             = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

mask_generator = SamAutomaticMaskGenerator(
                                            model=sam,
                                            points_per_side                     = utils.points_per_side                 ,
                                            pred_iou_thresh                     = utils.pred_iou_thresh                 ,
                                            stability_score_thresh              = utils.stability_score_thresh          ,
                                            crop_n_layers                       = utils.crop_n_layers                   ,
                                            crop_n_points_downscale_factor      = utils.crop_n_points_downscale_factor  ,
                                            min_mask_region_area                = utils.min_mask_region_area            ,  # Requires open-cv to run post-processing
                                            )
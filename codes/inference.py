from PIL import Image
from keras.models import model_from_json
import os
import utils
import numpy as np

# set filepath
datasets=["DRIVE","STARE"]
fundus_dir="../data/{}/test/images/"
mask_dir="../data/{}/test/mask/"
out_dir="../inference_outputs/{}"
f_model="../pretrained/{}_best.json"
f_weights="../pretrained/{}_best.h5"

for dataset in datasets:     
    # make directory
    if not os.path.isdir(out_dir.format(dataset)):
        os.makedirs(out_dir.format(dataset))
    
    # load the model and weights
    with open(f_model.format(dataset), 'r') as f:
        model=model_from_json(f.read())
    model.load_weights(f_weights.format(dataset))
    
    # iterate all images
    img_size=(640,640) if dataset=="DRIVE" else (720,720)
    ori_shape=(1,584,565) if dataset=="DRIVE" else (1,605,700)  # batchsize=1
    fundus_files=utils.all_files_under(fundus_dir.format(dataset))
    mask_files=utils.all_files_under(mask_dir.format(dataset))
    for index,fundus_file in enumerate(fundus_files):
        print "processing {}...".format(fundus_file)
        # load imgs
        img=utils.imagefiles2arrs([fundus_file])
        mask=utils.imagefiles2arrs([mask_files[index]])
        
        # z score with mean, std (batchsize=1)
        mean=np.mean(img[0,...][mask[0,...] == 255.0],axis=0)
        std=np.std(img[0,...][mask[0,...] == 255.0],axis=0)
        img[0,...]=(img[0,...]-mean)/std
        
        # run inference
        padded_img=utils.pad_imgs(img, img_size)
        vessel_img=model.predict(padded_img,batch_size=1)*255
        cropped_vessel=utils.crop_to_original(vessel_img[...,0], ori_shape)
        final_result=utils.remain_in_mask(cropped_vessel[0,...], mask[0,...])
        Image.fromarray(final_result.astype(np.uint8)).save(os.path.join(out_dir.format(dataset),os.path.basename(fundus_file)))

#V-GAN #
### Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks ###

![bitbucket_header.jpg](https://bitbucket.org/repo/ekyjKAX/images/3167681377-bitbucket_header.jpg)

## Package Dependency ##
scikit_image==0.12.3  
numpy==1.12.0  
matplotlib==2.0.0  
scipy==0.18.1  
Keras==2.0.4  
Pillow==4.1.1  
skimage==0.0  
scikit_learn==0.18.1  

## Directory Hierarchy ##
```
.
├── codes
│   ├── evaluation.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
├── data
│   ├── DRIVE
│   └── STARE
├── evaluation
│   ├── DRIVE
│   └── STARE
├── inference_outputs
│   ├── DRIVE
│   └── STARE
├── pretrained
│   ├── DRIVE_best.h5
│   ├── DRIVE_best.json
│   ├── STARE_best.h5
│   ├── STARE_best.json
│   ├── auc_pr_STARE.npy
│   ├── auc_roc_DRIVE.npy
│   ├── auc_roc_STARE.npy
│   └── auc_roc_pr_DRIVE.npy
└── results
    ├── DRIVE
    └── STARE
```
**codes** : source codes   
**data** : original data. File hierarchy is modified for convenience.  
**evaluation** : quantitative and qualitative evaluation.  
**inferenced_outputs** : outputs of inference with our model  
**pretrained** : pretrained model and weights  
**results** : results of other methods. These image files are retrieved from [here](http://www.vision.ee.ethz.ch/~cvlsegmentation/driu/downloads.html)  

## Training ##
Move to **codes** folder and run train.py 

``` python train.py --ratio_gan2seg=<int> --gpu_index=<int> --batch_size=<int> --dataset=[DRIVE|STARE] --discriminator=[pixel|patch1|patch2|image]```
### arguments ###
ratio_gan2seg : trade-coefficient between GAN loss and segmentation loss  
gpu_index : starting index for gpus to be used  
batch_size : number of images per a batch  
dataset : type of a dataset (DRIVE or STARE)  
discriminator : type of a discriminator (pixel or patch1 or patch2 or image)  

**CAVEAT**   
Training with the current codes requires main memory more than 50 GB and GPUs dedicated to Deep Learning. If no such system is available, it is recommended to use pre-trained model only for inference.

## Inference ##
Move to **codes** folder and run inferency.py

``` python inference.py```

Outputs of inference are generated in **inference_outputs** folder.


## Evaluation ##
Move to **codes** folder and run evaluation.py

``` python evaluation.py```

Results are generated in **evaluation** folder. Hierarchy of the folder is

```
.
├── DRIVE
│   ├── comparison
│   ├── measures
│   └── vessels
└── STARE
    ├── comparison
    ├── measures
    └── vessels
```
**comparison** : difference maps of our method  
**measures** : ROC and PR curves  
**vessels** : vessels superimposed on segmented masks

## LICENSE ##
This is under the MIT License  
Copyright (c) 2017 Vuno Inc. (www.vuno.co)
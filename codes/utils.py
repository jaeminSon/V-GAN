import os
import sys

from PIL import Image
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = sorted(filenames)
    
    return filenames

def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape

def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    if len(img_shape)==3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape)==2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)
    
    for file_index in xrange(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)
    
    return images_arr

def STARE_files(data_path):
    img_dir=os.path.join(data_path, "images")
    vessel_dir=os.path.join(data_path,"1st_manual")
    mask_dir=os.path.join(data_path,"mask")
    
    img_files=all_files_under(img_dir, extension=".ppm")
    vessel_files=all_files_under(vessel_dir, extension=".ppm")
    mask_files=all_files_under(mask_dir, extension=".ppm")
    
    return img_files, vessel_files, mask_files
def DRIVE_files(data_path):
    img_dir=os.path.join(data_path, "images")
    vessel_dir=os.path.join(data_path,"1st_manual")
    mask_dir=os.path.join(data_path,"mask")
    
    img_files=all_files_under(img_dir, extension=".tif")
    vessel_files=all_files_under(vessel_dir, extension=".gif")
    mask_files=all_files_under(mask_dir, extension=".gif")
    
    return img_files, vessel_files, mask_files

def crop_imgs(imgs,pad):
    """
    crop images (4D tensor) by [:,pad:-pad,pad:-pad,:] 
    """
    return imgs[:,pad:-pad,pad:-pad,:]

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
def discriminator_shape(n, d_out_shape):
    if len(d_out_shape)==1: # image gan
        return (n, d_out_shape[0])
    elif len(d_out_shape)==3:   # pixel, patch gan
        return (n, d_out_shape[0], d_out_shape[1], d_out_shape[2])
    return None

def input2discriminator(real_img_patches, real_vessel_patches, fake_vessel_patches, d_out_shape):
    real=np.concatenate((real_img_patches,real_vessel_patches), axis=3)
    fake=np.concatenate((real_img_patches,fake_vessel_patches), axis=3)
    
    d_x_batch=np.concatenate((real,fake), axis=0)
    
    # real : 1, fake : 0
    d_y_batch=np.ones(discriminator_shape(d_x_batch.shape[0], d_out_shape))
    d_y_batch[real.shape[0]:,...] = 0
    
    return d_x_batch, d_y_batch
 
def input2gan(real_img_patches, real_vessel_patches, d_out_shape):    
    g_x_batch=[real_img_patches,real_vessel_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch=np.ones(discriminator_shape(real_vessel_patches.shape[0], d_out_shape))
    return g_x_batch, g_y_batch
    
def print_metrics(itr, **kargs):
    print "*** Round {}  ====> ".format(itr),
    for name, value in kargs.items():
        print ( "{} : {}, ".format(name, value)),
    print ""
    sys.stdout.flush()

class TrainBatchFetcher(Iterator):
    """
    fetch batch of original images and vessel images
    """
    def __init__(self, train_imgs, train_vessels, batch_size):
        self.train_imgs=train_imgs
        self.train_vessels=train_vessels
        self.n_train_imgs=self.train_imgs.shape[0]
        self.batch_size=batch_size
        
    def next(self):
        indices=list(np.random.choice(self.n_train_imgs, self.batch_size))
        return self.train_imgs[indices,:,:,:], self.train_vessels[indices,:,:,:] 

def AUC_ROC(true_vessel_arr, pred_vessel_arr, save_fname):
    """
    Area under the ROC curve with x axis flipped
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)
    save_obj({"fpr":fpr, "tpr":tpr}, save_fname)
    AUC_ROC=roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    return AUC_ROC

def plot_AUC_ROC(fprs,tprs,method_names,fig_dir):
    # set font style
    font={'family':'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors=['r','b','y','g','#7e7e7e','m','k','c'] if len(fprs)==8 else ['r','y','m','k','g']
    indices=[6,2,5,3,4,7,1,0] if len(fprs)==8 else [3,1,2,4,0] 
    
    # print auc  
    print "****** ROC AUC ******"
    print "CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_roc*.npy)"
    for index in indices:
        if method_names[index]!='CRFs' and method_names[index]!='2nd_manual':
            print "{} : {:04}".format(method_names[index],auc(fprs[index],tprs[index]))
    
    # plot results
    for index in indices:
        if method_names[index]=='CRFs':
            plt.plot(fprs[index],tprs[index],colors[index]+'*',label=method_names[index].replace("_"," "))
        elif method_names[index]=='2nd_manual':
            plt.plot(fprs[index],tprs[index],colors[index]+'*',label='Human')
        else:
            plt.plot(fprs[index],tprs[index],colors[index],label=method_names[index].replace("_"," "))
    
    plt.title('ROC Curve')
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0,0.3)
    plt.ylim(0.7,1.0)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir,"ROC.png"))
    plt.close()

def plot_AUC_PR(precisions, recalls, method_names, fig_dir):
    # set font style
    font={'family':'serif'}
    matplotlib.rc('font', **font)
    
    # sort the order of plots manually for eye-pleasing plots
    colors=['r','b','y','g','#7e7e7e','m','k','c'] if len(precisions)==8 else ['r','y','m','k','g']
    indices=[6,2,5,3,4,7,1,0] if len(precisions)==8 else [3,1,2,4,0] 

    # print auc  
    print "****** Precision Recall AUC ******"
    print "CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_pr*.npy)"
    for index in indices:
        if method_names[index]!='CRFs' and method_names[index]!='2nd_manual':
            print "{} : {:04}".format(method_names[index],auc(recalls[index],precisions[index]))
    
    # plot results
    for index in indices:
        if method_names[index]=='2nd_manual':
            plt.plot(recalls[index],precisions[index],colors[index]+'*',label='Human')
        else:
            plt.plot(recalls[index],precisions[index],colors[index],label=method_names[index].replace("_"," "))
    
    plt.title('Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.5,1.0)
    plt.ylim(0.5,1.0)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir,"Precision_recall.png"))
    plt.close()
    
def AUC_PR(true_vessel_img, pred_vessel_img, save_fname):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    save_obj({"precision":precision, "recall":recall}, save_fname)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def threshold_by_otsu(pred_vessels, masks, connect=False, flatten=True):
    
    # cut by otsu threshold
    threshold=filters.threshold_otsu(pred_vessels[masks==1])
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1
    
    # connect pixels connected to strong intensity and has intensity more than otsu_th/factor
    if connect:
        factor=2
        for i in range(pred_vessels.shape[0]):
            pred_vessel=pred_vessels[i,...]
            pred_vessel_bin=pred_vessels_bin[i,...]
            pred_labels = measure.label(pred_vessel>threshold/factor)
            labels=np.unique(pred_labels[pred_vessel_bin==1])
            connected=np.in1d(pred_labels,labels).reshape(pred_vessels.shape)
            pred_vessel_bin[(pred_vessel>threshold/factor) & connected]=1
    
    if flatten:
        return pred_vessels_bin[masks==1].flatten()
    else:
        return pred_vessels_bin

def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
    sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
    specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
    return acc, sensitivity, specificity

def dice_coefficient(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)
    
    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)
    
    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)
    
    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def pad_imgs(imgs, img_size):
    img_h,img_w=imgs.shape[1], imgs.shape[2]
    target_h,target_w=img_size[0],img_size[1]
    if len(imgs.shape)==4:
        d=imgs.shape[3]
        padded=np.zeros((imgs.shape[0],target_h, target_w,d))
    elif len(imgs.shape)==3:
        padded=np.zeros((imgs.shape[0],img_size[0],img_size[1]))
    padded[:,(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
    
    return padded
    
def get_imgs(target_dir, augmentation, img_size, dataset):
    
    if dataset=='DRIVE':
        img_files, vessel_files, mask_files = DRIVE_files(target_dir)
    elif dataset=='STARE':
        img_files, vessel_files, mask_files = STARE_files(target_dir)
        
    # load images    
    fundus_imgs=imagefiles2arrs(img_files)
    vessel_imgs=imagefiles2arrs(vessel_files)/255
    mask_imgs=imagefiles2arrs(mask_files)/255
    fundus_imgs=pad_imgs(fundus_imgs, img_size)
    vessel_imgs=pad_imgs(vessel_imgs, img_size)
    mask_imgs=pad_imgs(mask_imgs, img_size)
    n_ori_imgs=fundus_imgs.shape[0]
    assert(np.min(vessel_imgs)==0 and np.max(vessel_imgs)==1 and np.min(mask_imgs)==0 and np.max(mask_imgs)==1)

    # augmentation
    if augmentation:
        # augment the original image (flip, rotate)
        all_fundus_imgs=[fundus_imgs]
        all_vessel_imgs=[vessel_imgs]
        flipped_imgs=fundus_imgs[:,:,::-1,:]    # flipped imgs
        flipped_vessels=vessel_imgs[:,:,::-1]
        all_fundus_imgs.append(flipped_imgs)
        all_vessel_imgs.append(flipped_vessels)
        for angle in range(5,360,5):  # rotated imgs 30~330
            all_fundus_imgs.append(rotate(fundus_imgs, angle, axes=(1, 2), reshape=False))
            all_fundus_imgs.append(rotate(flipped_imgs, angle, axes=(1, 2), reshape=False))
            all_vessel_imgs.append(rotate(vessel_imgs, angle, axes=(1, 2), reshape=False))
            all_vessel_imgs.append(rotate(flipped_vessels, angle, axes=(1, 2), reshape=False))
        fundus_imgs=np.concatenate(all_fundus_imgs,axis=0)
        vessel_imgs=np.round((np.concatenate(all_vessel_imgs,axis=0)))
    
    # z score with mean, std of each image
    means, stds=[],[]
    n_all_imgs=fundus_imgs.shape[0]
    for index in range(n_ori_imgs):
        means.append(np.mean(fundus_imgs[index,...][mask_imgs[index,...] == 1.0],axis=0))
        stds.append(np.std(fundus_imgs[index,...][mask_imgs[index,...] == 1.0],axis=0))
    for index in range(n_all_imgs):
        fundus_imgs[index,...]=(fundus_imgs[index,...]-means[index%n_ori_imgs])/stds[index%n_ori_imgs]

    return fundus_imgs, vessel_imgs, mask_imgs

def pixel_values_in_mask(true_vessels, pred_vessels,masks):
    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0
    assert np.max(masks)==1.0 and np.min(masks)==0.0
    assert pred_vessels.shape[0]==true_vessels.shape[0] and masks.shape[0]==true_vessels.shape[0]
    assert pred_vessels.shape[1]==true_vessels.shape[1] and masks.shape[1]==true_vessels.shape[1]
    assert pred_vessels.shape[2]==true_vessels.shape[2] and masks.shape[2]==true_vessels.shape[2]
     
    return true_vessels[masks==1].flatten(), pred_vessels[masks==1].flatten() 

def remain_in_mask(imgs,masks):
    imgs[masks==0]=0
    return imgs

def load_images_under_dir(path_dir):
    files=all_files_under(path_dir)
    return imagefiles2arrs(files)

def crop_to_original(imgs, ori_shape):
    pred_shape=imgs.shape
    assert len(pred_shape)<4

    if ori_shape == pred_shape:
        return imgs
    else: 
        if len(imgs.shape)>2:
            ori_h,ori_w =ori_shape[1],ori_shape[2]
            pred_h,pred_w=pred_shape[1],pred_shape[2]
            return imgs[:,(pred_h-ori_h)//2:(pred_h-ori_h)//2+ori_h,(pred_w-ori_w)//2:(pred_w-ori_w)//2+ori_w]
        else:
            ori_h,ori_w =ori_shape[0],ori_shape[1]
            pred_h,pred_w=pred_shape[0],pred_shape[1]
            return imgs[(pred_h-ori_h)//2:(pred_h-ori_h)//2+ori_h,(pred_w-ori_w)//2:(pred_w-ori_w)//2+ori_w]

def difference_map(ori_vessel, pred_vessel, mask):
    # thresholding
    ori_vessel=threshold_by_otsu(ori_vessel,mask, flatten=False)*255
    pred_vessel=threshold_by_otsu(pred_vessel,mask, flatten=False)*255
    
    diff_map=np.zeros((ori_vessel.shape[0],ori_vessel.shape[1],3))
    diff_map[(ori_vessel==255) & (pred_vessel==255)]=(0,255,0)   #Green (overlapping)
    diff_map[(ori_vessel==255) & (pred_vessel!=255)]=(255,0,0)    #Red (false negative, missing in pred)
    diff_map[(ori_vessel!=255) & (pred_vessel==255)]=(0,0,255)    #Blue (false positive)

    return diff_map

class Scheduler:
    def __init__(self, n_itrs_per_epoch_d, n_itrs_per_epoch_g, schedules, init_lr):
        self.schedules=schedules
        self.init_dsteps=n_itrs_per_epoch_d
        self.init_gsteps=n_itrs_per_epoch_g
        self.init_lr=init_lr
        self.dsteps=self.init_dsteps
        self.gsteps=self.init_gsteps
        self.lr=self.init_lr

    def get_dsteps(self):
        return self.dsteps
    
    def get_gsteps(self):
        return self.gsteps
    
    def get_lr(self):
        return self.lr
        
    def update_steps(self, n_round):
        key=str(n_round)
        if key in self.schedules['lr_decay']:
            self.lr=self.init_lr*self.schedules['lr_decay'][key]
        if key in self.schedules['step_decay']:
            self.dsteps=max(int(self.init_dsteps*self.schedules['step_decay'][key]),1)
            self.gsteps=max(int(self.init_gsteps*self.schedules['step_decay'][key]),1)

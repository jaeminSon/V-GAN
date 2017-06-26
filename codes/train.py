import numpy as np
from model import GAN, discriminator_pixel, discriminator_image, discriminator_patch1, discriminator_patch2, generator, discriminator_dummy
import utils
import os
from PIL import Image
import argparse
from keras import backend as K


# arrange arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    '--ratio_gan2seg',
    type=int,
    help="ratio of gan loss to seg loss",
    required=True
    )
parser.add_argument(
    '--gpu_index',
    type=str,
    help="gpu index",
    required=True
    )
parser.add_argument(
    '--discriminator',
    type=str,
    help="type of discriminator",
    required=True
    )
parser.add_argument(
    '--batch_size',
    type=int,
    help="batch size",
    required=True
    )
parser.add_argument(
    '--dataset',
    type=str,
    help="dataset name",
    required=True
    )
FLAGS,_= parser.parse_known_args()

# training settings 
os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu_index
n_rounds=31
batch_size=FLAGS.batch_size
n_filters_d=32
n_filters_g=32
val_ratio=0.05
init_lr=2e-4
schedules={'lr_decay':{},  # learning rate and step have the same decay schedule (not necessarily the values)
           'step_decay':{}}
alpha_recip=1./FLAGS.ratio_gan2seg if FLAGS.ratio_gan2seg>0 else 0
rounds_for_evaluation=range(n_rounds)

# set dataset
dataset=FLAGS.dataset
img_size= (640,640) if dataset=='DRIVE' else (720,720) # (h,w)  [original img size => DRIVE : (584, 565), STARE : (605,700) ]
img_out_dir="{}/segmentation_results_{}_{}".format(FLAGS.dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
model_out_dir="{}/model_{}_{}".format(FLAGS.dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
auc_out_dir="{}/auc_{}_{}".format(FLAGS.dataset,FLAGS.discriminator,FLAGS.ratio_gan2seg)
train_dir="../data/{}/training/".format(dataset)
test_dir="../data/{}/test/".format(dataset)
if not os.path.isdir(img_out_dir):
    os.makedirs(img_out_dir)
if not os.path.isdir(model_out_dir):
    os.makedirs(model_out_dir)
if not os.path.isdir(auc_out_dir):
    os.makedirs(auc_out_dir)
 
# set training and validation dataset
train_imgs, train_vessels, train_masks=utils.get_imgs(train_dir, augmentation=True, img_size=img_size, dataset=dataset)
train_vessels=np.expand_dims(train_vessels, axis=3)
n_all_imgs=train_imgs.shape[0]
n_train_imgs=int((1-val_ratio)*n_all_imgs)
train_indices=np.random.choice(n_all_imgs,n_train_imgs,replace=False)
train_batch_fetcher=utils.TrainBatchFetcher(train_imgs[train_indices,...], train_vessels[train_indices,...], batch_size)
val_imgs, val_vessels=train_imgs[np.delete(range(n_all_imgs),train_indices),...], train_vessels[np.delete(range(n_all_imgs),train_indices),...]
# set test dataset
test_imgs, test_vessels, test_masks=utils.get_imgs(test_dir, augmentation=False, img_size=img_size, dataset=dataset)

# create networks
g = generator(img_size, n_filters_g) 
if FLAGS.discriminator=='pixel':
    d, d_out_shape = discriminator_pixel(img_size, n_filters_d,init_lr)
elif FLAGS.discriminator=='patch1':
    d, d_out_shape = discriminator_patch1(img_size, n_filters_d,init_lr)
elif FLAGS.discriminator=='patch2':
    d, d_out_shape = discriminator_patch2(img_size, n_filters_d,init_lr)
elif FLAGS.discriminator=='image':
    d, d_out_shape = discriminator_image(img_size, n_filters_d,init_lr)
else:
    d, d_out_shape = discriminator_dummy(img_size, n_filters_d,init_lr)
    
gan=GAN(g,d,img_size, n_filters_g, n_filters_d,alpha_recip, init_lr)
g.summary()
d.summary()
gan.summary() 

# start training
scheduler=utils.Scheduler(n_train_imgs//batch_size, n_train_imgs//batch_size, schedules, init_lr) if alpha_recip>0 else utils.Scheduler(0, n_train_imgs//batch_size, schedules, init_lr)
print "training {} images :".format(n_train_imgs)
for n_round in range(n_rounds):
    # train D
    utils.make_trainable(d, True)
    for i in range(scheduler.get_dsteps()):
        real_imgs, real_vessels = next(train_batch_fetcher)
        d_x_batch, d_y_batch = utils.input2discriminator(real_imgs, real_vessels, g.predict(real_imgs,batch_size=batch_size), d_out_shape)
        d.train_on_batch(d_x_batch, d_y_batch)
  
    # train G (freeze discriminator)
    utils.make_trainable(d, False)
    for i in range(scheduler.get_gsteps()):
        real_imgs, real_vessels = next(train_batch_fetcher)
        g_x_batch, g_y_batch=utils.input2gan(real_imgs, real_vessels, d_out_shape)
        gan.train_on_batch(g_x_batch, g_y_batch)
    
    # evaluate on validation set
    if n_round in rounds_for_evaluation:
        # D
        d_x_test, d_y_test=utils.input2discriminator(val_imgs, val_vessels, g.predict(val_imgs,batch_size=batch_size), d_out_shape)
        loss, acc=d.evaluate(d_x_test,d_y_test, batch_size=batch_size, verbose=0)
        utils.print_metrics(n_round+1, loss=loss, acc=acc, type='D')
        # G
        gan_x_test, gan_y_test=utils.input2gan(val_imgs, val_vessels, d_out_shape)
        loss,acc=gan.evaluate(gan_x_test,gan_y_test, batch_size=batch_size, verbose=0)
        utils.print_metrics(n_round+1, acc=acc, loss=loss, type='GAN')
        # save the model and weights with the best validation loss
        
        with open(os.path.join(model_out_dir,"g_{}_{}_{}.json".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)),'w') as f:
            f.write(g.to_json())
        g.save_weights(os.path.join(model_out_dir,"g_{}_{}_{}.h5".format(n_round,FLAGS.discriminator,FLAGS.ratio_gan2seg)))
       
    # update step sizes, learning rates
    scheduler.update_steps(n_round)
    K.set_value(d.optimizer.lr, scheduler.get_lr())    
    K.set_value(gan.optimizer.lr, scheduler.get_lr())    
    
    # evaluate on test images
    if n_round in rounds_for_evaluation:    
        generated=g.predict(test_imgs,batch_size=batch_size)
        generated=np.squeeze(generated, axis=3)
        vessels_in_mask, generated_in_mask = utils.pixel_values_in_mask(test_vessels, generated , test_masks)
        auc_roc=utils.AUC_ROC(vessels_in_mask,generated_in_mask,os.path.join(auc_out_dir,"auc_roc_{}.npy".format(n_round)))
        auc_pr=utils.AUC_PR(vessels_in_mask, generated_in_mask,os.path.join(auc_out_dir,"auc_pr_{}.npy".format(n_round)))
        binarys_in_mask=utils.threshold_by_otsu(generated,test_masks)
        dice_coeff=utils.dice_coefficient(vessels_in_mask, binarys_in_mask)
        acc, sensitivity, specificity=utils.misc_measures(vessels_in_mask, binarys_in_mask)
        utils.print_metrics(n_round+1, auc_pr=auc_pr, auc_roc=auc_roc, dice_coeff=dice_coeff, 
                            acc=acc, senstivity=sensitivity, specificity=specificity, type='TESTING')
         
        # print test images
        segmented_vessel=utils.remain_in_mask(generated, test_masks)
        for index in range(segmented_vessel.shape[0]):
            Image.fromarray((segmented_vessel[index,:,:]*255).astype(np.uint8)).save(os.path.join(img_out_dir,str(n_round)+"_{:02}_segmented.png".format(index+1)))

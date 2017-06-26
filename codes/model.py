import os

from keras import backend as K
from keras import objectives
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')

def generator(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """    
        
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'
    
    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    
    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
     
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)    
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)
    
    g = Model(inputs, outputs, name=name)

    return g

def discriminator_pixel(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (pixel GAN)
    """
    
    # set image specifics
    k=3 # kernel size
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    
    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding="same")(inputs) 
    conv1 = LeakyReLU(0.2)(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding="same")(conv1) 
    conv2 = LeakyReLU(0.2)(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding="same")(conv2) 
    conv3 = LeakyReLU(0.2)(conv3)

    conv4 =  Conv2D(out_ch, kernel_size=(1, 1), padding="same")(conv3)
    outputs = Activation('sigmoid')(conv4)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_patch1(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
    stride 2 conv X 1
    max pooling X 2
    """
    
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'#'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1) 
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(pool1) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(conv2) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(pool2) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(conv3) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    
    outputs=Conv2D(out_ch, kernel_size=(1, 1), padding=padding, activation='sigmoid')(conv3)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = objectives.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_patch2(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
     stride 2 conv X 2
       max pooling X 4
    """
    
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'#'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1) 
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(pool1) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(conv2) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(pool2) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(conv3) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(pool3) 
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(conv4) 
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(pool4) 
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(conv5) 
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    
    outputs=Conv2D(out_ch, kernel_size=(1, 1), padding=padding, activation='sigmoid')(conv5)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = objectives.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_image(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """
    
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding='same'#'valid'

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1) 
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(pool1) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(conv2) 
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(pool2) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(conv3) 
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(pool3) 
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(conv4) 
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(pool4) 
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(conv5) 
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    
    gap=GlobalAveragePooling2D()(conv5)
    outputs=Dense(1, activation='sigmoid')(gap)
    
    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = objectives.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def discriminator_dummy(img_size, n_filters, init_lr, name='d'):    # naive unet without GAN
    # set image specifics
    img_ch=3 # image channels
    out_ch=1 # output channel
    img_height, img_width = img_size[0], img_size[1]

    inputs = Input((img_height, img_width, img_ch + out_ch))

    d = Model(inputs, inputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = objectives.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L
    
    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    
    return d, d.layers[-1].output_shape[1:]

def GAN(g,d,img_size,n_filters_g, n_filters_d, alpha_recip, init_lr, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_h, img_w=img_size[0], img_size[1]

    img_ch=3
    seg_ch=1
    
    fundus = Input((img_h, img_w, img_ch))
    vessel = Input((img_h, img_w, seg_ch))
    
    fake_vessel=g(fundus)
    fake_pair=Concatenate(axis=3)([fundus, fake_vessel])
    
    gan=Model([fundus, vessel], d(fake_pair), name=name)

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)
#         L_adv = objectives.mean_squared_error(y_true_flat, y_pred_flat)

        vessel_flat = K.batch_flatten(vessel)
        fake_vessel_flat = K.batch_flatten(fake_vessel)
        L_seg = objectives.binary_crossentropy(vessel_flat, fake_vessel_flat)
#         L_seg = objectives.mean_absolute_error(vessel_flat, fake_vessel_flat)

        return alpha_recip*L_adv + L_seg
    
    
    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=gan_loss, metrics=['accuracy'])
        
    return gan
    

import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

################################IMAGE_PREPROCESSING_AND_LOADING#####################################

# Basic Parameters
size = (128, 128)

# data mask
CT_img_list = []
FT_img_list = []
MN_img_list = []

# data image
T1_img_list = []
T2_img_list = []

base_path = 'dataset'

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.jpg'):
            filename = os.path.join(root, file)
            file_size = os.path.getsize(filename)
            category_name = os.path.basename(root)
            im = Image.open(filename)

            if im.mode == 'L':
                im = im.resize(size, Image.BILINEAR)
                imarray = np.array(im)

                # normalization
                imarray = (imarray - np.min(imarray))/(np.max(imarray) - np.min(imarray))

                # data class preprocessing
                if category_name == 'CT':
                    CT_img_list.append(imarray)
                elif category_name == 'FT':
                    FT_img_list.append(imarray)
                elif category_name == 'MN':
                    MN_img_list.append(imarray)
                elif category_name == 'T1':
                    T1_img_list.append(imarray)
                elif category_name == 'T2':
                    T2_img_list.append(imarray)



# mask preprocessing
CT_mask_array = np.asarray(np.float16(CT_img_list))
FT_mask_array = np.asarray(np.float16(FT_img_list))
MN_mask_array = np.asarray(np.float16(MN_img_list))

# image input preprocessing
T1_image_array = np.asarray(np.float16(T1_img_list))
T2_image_array = np.asarray(np.float16(T2_img_list))

image_array = np.concatenate((T1_image_array, T2_image_array), axis=0)
CT_mask = np.concatenate((CT_mask_array, CT_mask_array), axis=0)
FT_mask = np.concatenate((FT_mask_array, FT_mask_array), axis=0)
MN_mask = np.concatenate((MN_mask_array, MN_mask_array), axis=0)

print(CT_mask.shape)

temp_CT = list(zip(image_array, CT_mask))
temp_FT = list(zip(image_array, FT_mask))
temp_MN = list(zip(image_array, MN_mask))

random.shuffle(temp_CT)
random.shuffle(temp_FT)
random.shuffle(temp_MN)

image_array, CT_mask = zip(*temp_CT)
image_array = np.asarray(image_array)
CT_mask = np.asarray(CT_mask)

image_array, FT_mask = zip(*temp_FT)
image_array = np.asarray(image_array)
FT_mask = np.asarray(FT_mask)

image_array, MN_mask = zip(*temp_MN)
image_array = np.asarray(image_array)
MN_mask = np.asarray(MN_mask)


################################################LOSS_FUNCTION_CORRECTION##########################################

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=0)
    union = K.sum(y_true, axis=0) + K.sum(y_pred, axis=0)
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

######################################################UNET_MODEL#########################################################
def unet(input_size):

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=['categorical_crossentropy'], metrics=[dice_coef])

    model.summary()

    return model

################################################GPU_ACTIVATION############################################

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#############################################CT_TRAINING##############################################
# hyperparameter
EPOCHS = 60
input_size = (128, 128, 1)
batch_size = 8

# k fold parameters
NUM_FOLDS = 5
acc_per_fold = []
loss_per_fold = []

# k fold cross validation
kfold = KFold(n_splits=NUM_FOLDS, shuffle= True)
fold_no = 1
for train, test in kfold.split(image_array, CT_mask):

    # unet
    model = unet(input_size)

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    print(image_array[train].shape)
    print(CT_mask[train].shape)

    # training
    model_history = model.fit(image_array[train], CT_mask[train], epochs=EPOCHS,batch_size=batch_size)

    scores = model.evaluate(image_array[test], CT_mask[test], verbose=1)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; '
          f'{model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    model.save('fold_model/CT-unet-fold-' + str(fold_no) + '.h5')

    fold_no += 1

# CT_model
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


###########################################FT_TRAINING##################################################

# k fold cross validation
kfold = KFold(n_splits=NUM_FOLDS, shuffle= True)
fold_no = 1

for train, test in kfold.split(image_array, FT_mask):

    # unet
    model = unet(input_size)

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    print(image_array[train].shape)
    print(FT_mask[train].shape)

    # training
    model_history = model.fit(image_array[train], FT_mask[train], epochs=EPOCHS, batch_size=batch_size)

    scores = model.evaluate(image_array[test], FT_mask[test], verbose=1)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    model.save('fold_model/FT-unet-fold-' + str(fold_no) + '.h5')

    fold_no += 1

# FT_model
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

##########################################MN_TRAINING###################################################

# k fold cross validation
kfold = KFold(n_splits=NUM_FOLDS, shuffle= True)
fold_no = 1
batch_size = 8

for train, test in kfold.split(image_array, MN_mask):

    # unet
    model = unet(input_size)

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    print(image_array[train].shape)
    print(MN_mask[train].shape)

    # training
    model_history = model.fit(image_array[train], MN_mask[train], epochs=EPOCHS, batch_size=batch_size)

    scores = model.evaluate(image_array[test], MN_mask[test], verbose=1)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    model.save('fold_model/MN-unet-fold-' + str(fold_no) + '.h5')

    fold_no += 1

# MN_model
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
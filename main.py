import tkinter as tk
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk, ImageFilter
from tkinter.filedialog import askopenfilename, askdirectory
from tensorflow.keras import backend as K

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
###########################################LOSS_FUNCTION_CORRECTION##########################################

def binary_focal_loss(gamma=2, alpha=0.25):

    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=0)
    union = K.sum(y_true, axis=0) + K.sum(y_pred, axis=0)
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


################################################### GUI ###############################################
window = tk.Tk()
window.title('FinalProject')
window.geometry('1200x700')

# initial matrix and parameters
size = (128, 128)

# prediction DC
CT_DC = []
FT_DC = []
MN_DC = []

# sequence mask
CT_img_list = []
FT_img_list = []
MN_img_list = []

# sequence image
T1_img_list = []
T2_img_list = []

def Select_Squence():
    filepath = askdirectory(title='choose t1 file', initialdir='dataset/')

    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith('.jpg'):
                filename = os.path.join(root, file)
                category_name = os.path.basename(root)
                im = Image.open(filename)

                if im.mode == 'L':
                    im = im.resize(size, Image.BILINEAR)
                    imarray = np.array(im)

                    # normalization
                    imarray = (imarray - np.min(imarray)) / (np.max(imarray) - np.min(imarray))

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

# image initialization matrix
# do the normalization first
t1_image = []
t2_image = []
ct_image = []
ft_image = []
mn_image = []

# select image and normalization
def Select_t1_Image():

    filename = askopenfilename(initialdir='dataset/')

    # test and prediction
    im = Image.open(filename)
    im = im.convert('L')
    im = im.resize(size, Image.BILINEAR)
    image_array = np.array(im)
    image_array = (image_array - np.min(image_array))/(np.max(image_array) - np.min(image_array))
    t1_image.append(image_array)
    print(filename)
    print(t1_image)

    # canvas show
    image = cv2.cvtColor(cv2.resize(cv2.imread(filename), (128, 128)), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    show = ImageTk.PhotoImage(image)
    T1_label.create_image(0, 0, anchor=NW, image=show)
    T1_label.img = show

def Select_t2_Image():
    filename = askopenfilename(initialdir='dataset/')

    # test and prediction
    im = Image.open(filename)
    im = im.convert('L')
    im = im.resize(size, Image.BILINEAR)
    image_array = np.array(im)
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    t2_image.append(image_array)
    print(filename)
    print(t2_image)

    # canvas show
    image = cv2.cvtColor(cv2.resize(cv2.imread(filename), (128, 128)), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    show = ImageTk.PhotoImage(image)
    T2_label.create_image(0, 0, anchor=NW, image=show)
    T2_label.img = show

def Select_ct_Image(*args):
    filename = askopenfilename(initialdir='dataset/')

    # test and prediction
    im = Image.open(filename)
    im = im.convert('L')
    im = im.resize(size, Image.BILINEAR)
    image_array = np.array(im)
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    ct_image.append(image_array)
    print(filename)
    print(ct_image)

    # canvas show
    image = cv2.cvtColor(cv2.resize(cv2.imread(filename), (128, 128)), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    show = ImageTk.PhotoImage(image)
    CT_label.create_image(0, 0, anchor=NW, image=show)
    CT_label.img = show

def Select_ft_Image():
    filename = askopenfilename(initialdir='dataset/')

    # test and prediction
    im = Image.open(filename)
    im = im.convert('L')
    im = im.resize(size, Image.BILINEAR)
    image_array = np.array(im)
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    ft_image.append(image_array)
    print(filename)
    print(ft_image)

    # canvas show
    image = cv2.cvtColor(cv2.resize(cv2.imread(filename), (128, 128)), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    show = ImageTk.PhotoImage(image)
    FT_label.create_image(0, 0, anchor=NW, image=show)
    FT_label.img = show

def Select_mn_Image():
    filename = askopenfilename(initialdir='dataset/')

    # test and prediction
    im = Image.open(filename)
    im = im.convert('L')
    im = im.resize(size, Image.BILINEAR)
    image_array = np.array(im)
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    mn_image.append(image_array)
    print(filename)
    print(mn_image)

    # canvas show
    image = cv2.cvtColor(cv2.resize(cv2.imread(filename), (128, 128)), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    show = ImageTk.PhotoImage(image)
    MN_label.create_image(0, 0, anchor=NW, image=show)
    MN_label.img = show

def predict_image():

    # pick image
    t1 = np.asarray(np.float16(t1_image))
    t2 = np.asarray(np.float16(t2_image))
    ct = np.asarray(np.float16(ct_image))
    ft = np.asarray(np.float16(ft_image))
    mn = np.asarray(np.float16(mn_image))

    # image preprocessing
    T1 = t1.reshape(-1, 128, 128, 1)
    T1 = np.uint8(T1 * 255)

    T2 = t2.reshape(-1, 128, 128, 1)
    T2 = np.uint8(T2 * 255)

    CT = ct.reshape(-1, 128, 128, 1)

    FT = ft.reshape(-1, 128, 128, 1)

    MN = mn.reshape(-1, 128, 128, 1)

    # load model
    CT_model = load_model('validation_model/CT_validation_model_dice128.h5',
                          custom_objects={'binary_focal_loss_fixed': binary_focal_loss(), 'dice_coef': dice_coef})
    FT_model = load_model('FT_validation_unet_model_diceloss128.h5', custom_objects={'dice_coef': dice_coef})
    MN_model = load_model('MN_validation_unet_model_diceloss128.h5', custom_objects={'dice_coef': dice_coef})

    index = 0
    print('=================================================')

    print('index = ' + str(index))
    print('-------------------------------------------------')
    # load image and mask
    T1_image = T1[index]
    T1_image = T1_image.reshape(1, 128, 128, 1)
    T1_image_show = np.reshape(np.uint8(T1_image), (128, 128))

    T2_image = T2[index]
    T2_image = T2_image.reshape(1, 128, 128, 1)
    T2_image_show = np.reshape(np.uint8(T2_image), (128, 128))

    # tki show
    show = ImageTk.PhotoImage(Image.fromarray(T1_image_show))
    T1_label.create_image(0, 0, anchor=NW, image=show)
    T1_label.img = show

    show = ImageTk.PhotoImage(Image.fromarray(T2_image_show))
    T2_label.create_image(0, 0, anchor=NW, image=show)
    T2_label.img = show

    T1_BG = cv2.cvtColor(T1_image_show, cv2.COLOR_GRAY2RGB)

    CT_mask = CT[index]
    FT_mask = FT[index]
    MN_mask = MN[index]

    ##
    # prediction
    # CT predict
    predict_CT = CT_model.predict(T1_image, verbose=1)
    predict_CT = (predict_CT > 0.9).astype(np.uint8)
    predict_CT_mask = predict_CT.astype(np.uint8).reshape(128, 128)
    CT_mask_DC = CT_mask.reshape((128, 128))

    # become image
    CT_mask_show = np.reshape(np.uint8(CT_mask * 255), (128, 128))
    predict_CT_show = np.squeeze(predict_CT_mask*255)

    CT_dice_coef = dice_coef(np.float16(predict_CT_mask), np.float16(CT_mask_DC))
    CT_dice_coef = CT_dice_coef.numpy()

    CT_DC_IM.set('Carpal Tunnels : ' + str(CT_dice_coef))

    CT_DC.append(CT_dice_coef)
    print('CT_DC_coefficient:' + str(CT_dice_coef))
    print('-------------------------------------------------')

    # GUI show
    CT_mask_show = Image.fromarray(CT_mask_show)
    CT_mask_show = ImageTk.PhotoImage(CT_mask_show)
    CT_label.create_image(0, 0, anchor=NW, image=CT_mask_show)
    CT_label.img = CT_mask_show

    predict_CT_show = Image.fromarray(predict_CT_show)
    show = ImageTk.PhotoImage(predict_CT_show)
    predict_CT_label.create_image(0, 0, anchor=NW, image=show)
    predict_CT_label.img = show

    mark_CT = predict_CT_show.filter(ImageFilter.FIND_EDGES)
    mark_CT = mark_CT.convert('RGB')
    mark_CT = np.array(mark_CT)
    R = mark_CT[:, :, 0]
    G = mark_CT[:, :, 1]
    B = mark_CT[:, :, 2]
    zeros = np.zeros((128, 128), dtype="uint8")
    mark_CT[:, :, 0] = R
    mark_CT[:, :, 1] = zeros
    mark_CT[:, :, 2] = zeros
    mark_CT = Image.fromarray(mark_CT)

    ##
    # FT predict
    predict_FT = FT_model.predict(T1_image, verbose=1)
    predict_FT = (predict_FT > 0.1).astype(np.uint8)
    predict_FT_mask = predict_FT.astype(np.uint8).reshape(128, 128)
    FT_mask_DC = FT_mask.reshape((128, 128))

    FT_mask_show = np.reshape(np.uint8(FT_mask * 255), (128, 128))
    predict_FT_show = np.squeeze(predict_FT_mask*255)

    FT_dice_coef = dice_coef(np.float16(predict_FT_mask), np.float16(FT_mask_DC))
    FT_dice_coef = FT_dice_coef.numpy()

    FT_DC_IM.set('Flexor Tendons : ' + str(FT_dice_coef))

    FT_DC.append(FT_dice_coef)
    print('FT_DC_coefficient:' + str(FT_dice_coef))
    print('-------------------------------------------------')

    # GUI show
    FT_mask_show = Image.fromarray(FT_mask_show)
    FT_mask_show = ImageTk.PhotoImage(FT_mask_show)
    FT_label.create_image(0, 0, anchor=NW, image=FT_mask_show)
    FT_label.img = FT_mask_show

    predict_FT_show = Image.fromarray(predict_FT_show)
    show = ImageTk.PhotoImage(predict_FT_show)
    predict_FT_label.create_image(0, 0, anchor=NW, image=show)
    predict_FT_label.img = show

    mark_FT = predict_FT_show.filter(ImageFilter.FIND_EDGES)
    mark_FT = mark_FT.convert('RGB')
    mark_FT = np.array(mark_FT)
    R = mark_FT[:, :, 0]
    G = mark_FT[:, :, 1]
    B = mark_FT[:, :, 2]
    zeros = np.zeros((128, 128), dtype="uint8")
    mark_FT[:, :, 0] = zeros
    mark_FT[:, :, 1] = G
    mark_FT[:, :, 2] = zeros
    mark_FT = Image.fromarray(mark_FT)

    ##
    # MN predict
    predict_MN = MN_model.predict(T2_image, verbose=1)
    predict_MN = (predict_MN > 0.1).astype(np.uint8)
    predict_MN_mask = predict_MN.astype(np.uint8).reshape(128, 128)
    MN_mask_DC = MN_mask.reshape(128, 128)

    MN_mask_show = np.reshape(np.uint8(MN_mask * 255), (128, 128))
    predict_MN_show = np.squeeze(predict_MN_mask*255)

    MN_dice_coef = dice_coef(np.float16(predict_MN_mask), np.float16(MN_mask_DC))
    MN_dice_coef = MN_dice_coef.numpy()

    MN_DC_IM.set('Median Nerve : ' + str(MN_dice_coef))

    MN_DC.append(MN_dice_coef)
    print('MN_DC_coefficient:' + str(MN_dice_coef))
    print('-------------------------------------------------')

    MN_mask_show = Image.fromarray(MN_mask_show)
    MN_mask_show = ImageTk.PhotoImage(MN_mask_show)
    MN_label.create_image(0, 0, anchor=NW, image=MN_mask_show)
    MN_label.img = MN_mask_show

    predict_MN_show = Image.fromarray(predict_MN_show)
    show = ImageTk.PhotoImage(predict_MN_show)
    predict_MN_label.create_image(0, 0, anchor=NW, image=show)
    predict_MN_label.img = show

    mark_MN = predict_MN_show.filter(ImageFilter.FIND_EDGES)
    mark_MN = mark_MN.convert('RGB')
    mark_MN = np.array(mark_MN)
    R = mark_MN[:, :, 0]
    G = mark_MN[:, :, 1]
    B = mark_MN[:, :, 2]
    zeros = np.zeros((128, 128), dtype="uint8")
    mark_MN[:, :, 0] = zeros
    mark_MN[:, :, 1] = zeros
    mark_MN[:, :, 2] = B
    mark_MN = Image.fromarray(mark_MN)
    #
    mark_MN = np.array(mark_MN)
    mark_CT = np.array(mark_CT)
    mark_FT = np.array(mark_FT)
    T1_BG = np.array(T1_BG)
    result = cv2.add(mark_FT, mark_CT)
    result = cv2.add(result, T1_BG)
    result = cv2.add(result, mark_MN)
    result = Image.fromarray(result)
    show = ImageTk.PhotoImage(result)
    predict_mark_label.create_image(0, 0, anchor=NW, image=show)
    predict_mark_label.img = show


    print('CT_DC = ' + str(CT_DC))
    print('FT_DC = ' + str(FT_DC))
    print('MN_DC = ' + str(MN_DC))

    CT_DC_mean = np.mean(CT_DC)
    FT_DC_mean = np.mean(FT_DC)
    MN_DC_mean = np.mean(MN_DC)

    print('CT_DC_mean = ' + str(CT_DC_mean))
    print('FT_DC_mean = ' + str(FT_DC_mean))
    print('MN_DC_mean = ' + str(MN_DC_mean))

    CT_DC_SE.set('Carpal Tunnels : ' + str(CT_DC_mean))
    FT_DC_SE.set('Flexor Tendons : ' + str(FT_DC_mean))
    MN_DC_SE.set('Median Nerve : ' + str(MN_DC_mean))

# string variation initialization
CT_DC_SE = StringVar()
FT_DC_SE = StringVar()
MN_DC_SE = StringVar()

CT_DC_IM = StringVar()
FT_DC_IM = StringVar()
MN_DC_IM = StringVar()

# predict array initialization
predict_CT_mask_array = []
predict_FT_mask_array = []
predict_MN_mask_array = []
predict_result = []

CT_DC_mean = 0
FT_DC_mean = 0
MN_DC_mean = 0

def predict_sequence():

    # image sequence
    # mask preprocessing
    CT_mask_array = np.asarray(np.float16(CT_img_list))
    FT_mask_array = np.asarray(np.float16(FT_img_list))
    MN_mask_array = np.asarray(np.float16(MN_img_list))

    # image input preprocessing
    T1_image_array = np.asarray(np.float16(T1_img_list))
    T2_image_array = np.asarray(np.float16(T2_img_list))

    # T1 image
    T1 = T1_image_array.reshape(-1, 128, 128, 1)
    T1 = np.uint8(T1 * 255)
    print('T1.shape = '+str(T1.shape))

    # T2 image
    T2 = T2_image_array.reshape(-1, 128, 128, 1)
    T2 = np.uint8(T2 * 255)
    print('T2.shape = '+str(T2.shape))

    # CT mask
    CT = CT_mask_array.reshape(-1, 128, 128, 1)
    print('CT.shape = '+str(CT.shape))

    # FT mask
    FT = FT_mask_array.reshape(-1, 128, 128, 1)
    print('FT.shape = '+str(FT.shape))

    # MN mask
    MN = MN_mask_array.reshape(-1, 128, 128, 1)
    print('MN.shape = '+str(MN.shape))

##
    # load model
    CT_model = load_model('validation_model/CT_validation_model_dice128.h5',
                          custom_objects={'binary_focal_loss_fixed': binary_focal_loss(), 'dice_coef': dice_coef})
    FT_model = load_model('FT_validation_unet_model_diceloss128.h5', custom_objects={'dice_coef': dice_coef})
    MN_model = load_model('MN_validation_unet_model_diceloss128.h5', custom_objects={'dice_coef': dice_coef})

##
    for index in range(0, 20, 1):
        print('=================================================')

        print('index = ' + str(index))
        print('-------------------------------------------------')
        # load image and mask
        T1_image = T1[index]
        T1_image = T1_image.reshape(1, 128, 128, 1)
        T1_image_show = np.reshape(np.uint8(T1_image), (128, 128))

        T2_image = T2[index]
        T2_image = T2_image.reshape(1, 128, 128, 1)
        T2_image_show = np.reshape(np.uint8(T2_image), (128, 128))

        # tki show
        show = ImageTk.PhotoImage(Image.fromarray(T1_image_show))
        T1_label.create_image(0, 0, anchor=NW, image=show)
        T1_label.img = show

        show = ImageTk.PhotoImage(Image.fromarray(T2_image_show))
        T2_label.create_image(0, 0, anchor=NW, image=show)
        T2_label.img = show

        T1_BG = cv2.cvtColor(T1_image_show, cv2.COLOR_GRAY2RGB)

        CT_mask = CT[index]
        FT_mask = FT[index]
        MN_mask = MN[index]

        ##
        # prediction
        # CT predict
        predict_CT = CT_model.predict(T1_image, verbose=1)
        predict_CT = (predict_CT > 0.9).astype(np.uint8)
        predict_CT_mask = (predict_CT).astype(np.uint8).reshape(128, 128)
        CT_mask_DC = CT_mask.reshape((128, 128))

        # become image
        CT_mask_show = np.reshape(np.uint8(CT_mask * 255), (128, 128))
        predict_CT_show = np.squeeze(predict_CT_mask*255)
        predict_CT_mask_array.append(predict_CT_show)

        CT_dice_coef = dice_coef(np.float16(predict_CT_mask), np.float16(CT_mask_DC))
        CT_dice_coef = CT_dice_coef.numpy()

        CT_DC_IM.set('Carpal Tunnels : ' + str(CT_dice_coef))

        CT_DC.append(CT_dice_coef)
        print('CT_DC_coefficient:' + str(CT_dice_coef))
        print('-------------------------------------------------')

        show = ImageTk.PhotoImage(Image.fromarray(CT_mask_show))
        CT_label.create_image(0, 0, anchor=NW, image=show)
        CT_label.img = show

        predict_CT_show = Image.fromarray(predict_CT_show)
        show = ImageTk.PhotoImage(predict_CT_show)
        predict_CT_label.create_image(0, 0, anchor=NW, image=show)
        predict_CT_label.img = show

        mark_CT = predict_CT_show.filter(ImageFilter.FIND_EDGES)
        mark_CT = mark_CT.convert('RGB')
        mark_CT = np.array(mark_CT)
        R = mark_CT[:, :, 0]
        G = mark_CT[:, :, 1]
        B = mark_CT[:, :, 2]
        zeros = np.zeros((128, 128), dtype="uint8")
        mark_CT[:, :, 0] = R
        mark_CT[:, :, 1] = zeros
        mark_CT[:, :, 2] = zeros
        mark_CT = Image.fromarray(mark_CT)

        ##
        # FT predict
        predict_FT = FT_model.predict(T1_image, verbose=1)
        predict_FT = (predict_FT > 0.1).astype(np.uint8)
        predict_FT_mask = (predict_FT).astype(np.uint8).reshape(128, 128)
        FT_mask_DC = FT_mask.reshape((128, 128))

        FT_mask_show = np.reshape(np.uint8(FT_mask * 255), (128, 128))
        predict_FT_show = np.squeeze(predict_FT_mask*255)
        predict_FT_mask_array.append(predict_FT_show)

        FT_dice_coef = dice_coef(np.float16(predict_FT_mask), np.float16(FT_mask_DC))
        FT_dice_coef = FT_dice_coef.numpy()

        FT_DC_IM.set('Flexor Tendons : ' + str(FT_dice_coef))

        FT_DC.append(FT_dice_coef)
        print('FT_DC_coefficient:' + str(FT_dice_coef))
        print('-------------------------------------------------')



        show = ImageTk.PhotoImage(Image.fromarray(FT_mask_show))
        FT_label.create_image(0, 0, anchor=NW, image=show)
        FT_label.img = show

        predict_FT_show = Image.fromarray(predict_FT_show)
        show = ImageTk.PhotoImage(predict_FT_show)
        predict_FT_label.create_image(0, 0, anchor=NW, image=show)
        predict_FT_label.img = show

        mark_FT = predict_FT_show.filter(ImageFilter.FIND_EDGES)
        mark_FT = mark_FT.convert('RGB')
        mark_FT = np.array(mark_FT)
        R = mark_FT[:, :, 0]
        G = mark_FT[:, :, 1]
        B = mark_FT[:, :, 2]
        zeros = np.zeros((128, 128), dtype="uint8")
        mark_FT[:, :, 0] = zeros
        mark_FT[:, :, 1] = G
        mark_FT[:, :, 2] = zeros
        mark_FT = Image.fromarray(mark_FT)

        ##
        # MN predict
        predict_MN = MN_model.predict(T2_image, verbose=1)
        predict_MN = (predict_MN > 0.1).astype(np.uint8)
        predict_MN_mask = (predict_MN).astype(np.uint8).reshape(128, 128)
        MN_mask_DC = MN_mask.reshape(128, 128)

        MN_mask_show = np.reshape(np.uint8(MN_mask * 255), (128, 128))
        predict_MN_show = np.squeeze(predict_MN_mask*255)
        predict_MN_mask_array.append(predict_MN_show)

        MN_dice_coef = dice_coef(np.float16(predict_MN_mask), np.float16(MN_mask_DC))
        MN_dice_coef = MN_dice_coef.numpy()

        MN_DC_IM.set('Median Nerve : ' + str(MN_dice_coef))

        MN_DC.append(MN_dice_coef)
        print('MN_DC_coefficient:' + str(MN_dice_coef))
        print('-------------------------------------------------')

        MN_mask_show = Image.fromarray(MN_mask_show)
        MN_mask_show = ImageTk.PhotoImage(MN_mask_show)
        MN_label.create_image(0, 0, anchor=NW, image=MN_mask_show)
        MN_label.img = MN_mask_show

        predict_MN_show = Image.fromarray(predict_MN_show)
        show = ImageTk.PhotoImage(predict_MN_show)
        predict_MN_label.create_image(0, 0, anchor=NW, image=show)
        predict_MN_label.img = show

        mark_MN = predict_MN_show.filter(ImageFilter.FIND_EDGES)
        mark_MN = mark_MN.convert('RGB')
        mark_MN = np.array(mark_MN)
        R = mark_MN[:, :, 0]
        G = mark_MN[:, :, 1]
        B = mark_MN[:, :, 2]
        zeros = np.zeros((128, 128), dtype="uint8")
        mark_MN[:, :, 0] = zeros
        mark_MN[:, :, 1] = zeros
        mark_MN[:, :, 2] = B
        mark_MN = Image.fromarray(mark_MN)
        #
        mark_MN = np.array(mark_MN)
        mark_CT = np.array(mark_CT)
        mark_FT = np.array(mark_FT)
        T1_BG = np.array(T1_BG)
        result = cv2.add(mark_FT, mark_CT)
        result = cv2.add(result, T1_BG)
        result = cv2.add(result, mark_MN)
        predict_result.append(result)
        result = Image.fromarray(result)
        show = ImageTk.PhotoImage(result)
        predict_mark_label.create_image(0, 0, anchor=NW, image=show)
        predict_mark_label.img = show

    print('CT_DC = ' + str(CT_DC))
    print('FT_DC = ' + str(FT_DC))
    print('MN_DC = ' + str(MN_DC))

    CT_DC_mean = np.mean(CT_DC)
    FT_DC_mean = np.mean(FT_DC)
    MN_DC_mean = np.mean(MN_DC)

    print('CT_DC_mean = ' + str(CT_DC_mean))
    print('FT_DC_mean = ' + str(FT_DC_mean))
    print('MN_DC_mean = ' + str(MN_DC_mean))

    CT_DC_SE.set('Carpal Tunnels : ' + str(CT_DC_mean))
    FT_DC_SE.set('Flexor Tendons : ' + str(FT_DC_mean))
    MN_DC_SE.set('Median Nerve : ' + str(MN_DC_mean))

##
def image_show(*args):
    # get horizontal_scaler position
    index = horizontal_scaler.get()

    CT_mask_array = np.asarray(np.float16(CT_img_list))
    FT_mask_array = np.asarray(np.float16(FT_img_list))
    MN_mask_array = np.asarray(np.float16(MN_img_list))
    T1_image_array = np.asarray(np.float16(T1_img_list))
    T2_image_array = np.asarray(np.float16(T2_img_list))

    T1 = T1_image_array.reshape(-1, 128, 128, 1)
    T1 = np.uint8(T1 * 255)
    T2 = T2_image_array.reshape(-1, 128, 128, 1)
    T2 = np.uint8(T2 * 255)
    CT = CT_mask_array.reshape(-1, 128, 128, 1)
    FT = FT_mask_array.reshape(-1, 128, 128, 1)
    MN = MN_mask_array.reshape(-1, 128, 128, 1)

    T1_image = T1[index]
    T1_image = T1_image.reshape(1, 128, 128, 1)
    T1_image_show = np.reshape(np.uint8(T1_image), (128, 128))

    T2_image = T2[index]
    T2_image = T2_image.reshape(1, 128, 128, 1)
    T2_image_show = np.reshape(np.uint8(T2_image), (128, 128))

    # tki show
    show = ImageTk.PhotoImage(Image.fromarray(T1_image_show))
    T1_label.create_image(0, 0, anchor=NW, image=show)
    T1_label.img = show

    show = ImageTk.PhotoImage(Image.fromarray(T2_image_show))
    T2_label.create_image(0, 0, anchor=NW, image=show)
    T2_label.img = show

    CT_mask = CT[index]
    FT_mask = FT[index]
    MN_mask = MN[index]

    CT_mask_show = np.reshape(np.uint8(CT_mask * 255), (128, 128))
    show = ImageTk.PhotoImage(Image.fromarray(CT_mask_show))
    CT_label.create_image(0, 0, anchor=NW, image=show)
    CT_label.img = show

    FT_mask_show = np.reshape(np.uint8(FT_mask * 255), (128, 128))
    show = ImageTk.PhotoImage(Image.fromarray(FT_mask_show))
    FT_label.create_image(0, 0, anchor=NW, image=show)
    FT_label.img = show

    MN_mask_show = np.reshape(np.uint8(MN_mask * 255), (128, 128))
    show = ImageTk.PhotoImage(Image.fromarray(MN_mask_show))
    MN_label.create_image(0, 0, anchor=NW, image=show)
    MN_label.img = show

    CT_IP = predict_CT_mask_array[index]
    predict_CT_show = Image.fromarray(CT_IP)
    show = ImageTk.PhotoImage(predict_CT_show)
    predict_CT_label.create_image(0, 0, anchor=NW, image=show)
    predict_CT_label.img = show

    FT_IP = predict_FT_mask_array[index]
    predict_FT_show = Image.fromarray(FT_IP)
    show = ImageTk.PhotoImage(predict_FT_show)
    predict_FT_label.create_image(0, 0, anchor=NW, image=show)
    predict_FT_label.img = show

    MN_IP = predict_MN_mask_array[index]
    predict_MN_show = Image.fromarray(MN_IP)
    show = ImageTk.PhotoImage(predict_MN_show)
    predict_MN_label.create_image(0, 0, anchor=NW, image=show)
    predict_MN_label.img = show

    predict_P = predict_result[index]
    result = Image.fromarray(predict_P)
    show = ImageTk.PhotoImage(result)
    predict_mark_label.create_image(0, 0, anchor=NW, image=show)
    predict_mark_label.img = show

    CT_DC_V = CT_DC[index]
    CT_DC_IM.set('Carpal Tunnels : ' + str(CT_DC_V))

    FT_DC_V = FT_DC[index]
    FT_DC_IM.set('Carpal Tunnels : ' + str(FT_DC_V))

    MN_DC_V = MN_DC[index]
    MN_DC_IM.set('Carpal Tunnels : ' + str(MN_DC_V))

## reset
def reset():

    # all data reset
    t1_image.clear()
    t2_image.clear()
    ct_image.clear()
    ft_image.clear()
    mn_image.clear()
    CT_DC.clear()
    FT_DC.clear()
    MN_DC.clear()
    CT_img_list.clear()
    FT_img_list.clear()
    MN_img_list.clear()
    T1_img_list.clear()
    T2_img_list.clear()
    predict_CT_mask_array.clear()
    predict_FT_mask_array.clear()
    predict_MN_mask_array.clear()
    predict_result.clear()

##
Select_t1_Squence_B = tk.Button(window, text = 'Select_Squence', command=Select_Squence, width = 25)
Select_t1_Squence_B.place(x = 50, y = 10)

Select_t1_Image_B = tk.Button(window, text = 'Select_T1_Image', command=Select_t1_Image, width = 25)
Select_t1_Image_B.place(x = 50, y = 50)

Select_t2_Image_B = tk.Button(window, text = 'Select_T2_Image', command=Select_t2_Image, width = 25)
Select_t2_Image_B.place(x = 50, y = 90)

##
Select_ct_Image_B = tk.Button(window, text = 'Select_CT_Image', command=Select_ct_Image, width = 25)
Select_ct_Image_B.place(x = 350, y = 10)

Select_ft_Image_B = tk.Button(window, text = 'Select_FT_Image', command=Select_ft_Image, width = 25)
Select_ft_Image_B.place(x = 350, y = 50)

Select_mn_Image_B = tk.Button(window, text = 'Select_MN_Image', command=Select_mn_Image, width = 25)
Select_mn_Image_B.place(x = 350, y = 90)

##
run_image_B = tk.Button(window, text = 'predict_image', command=predict_image, width = 25)
run_image_B.place(x = 650, y = 10)

run_sequence_B = tk.Button(window, text = 'predict_sequence', command=predict_sequence, width = 25)
run_sequence_B.place(x = 650, y = 50)

reset_B = tk.Button(window, text = 'reset', command=reset, width = 25)
reset_B.place(x = 650, y = 90)

##roller
horizontal_scaler = tk.Scale(window, from_=0, to=18, tickinterval=1, length=500, orient="horizontal", command=image_show)
horizontal_scaler.place(x = 200, y = 630)

## Label
# DC Loss
# Sequence
sequence_dc = Label(window, text='Sequence DC (mean) :')
sequence_dc.place(x = 950, y = 130)

median_nerve_sq = Label(window, text='median_nerve :', textvariable=MN_DC_SE)
median_nerve_sq.place(x = 950, y = 170)

flexor_tendons_sq = Label(window, text='flexor_tendons :', textvariable=FT_DC_SE)
flexor_tendons_sq.place(x = 950, y = 210)

carpal_tunnels_sq = Label(window, text='carpal_tunnels :', textvariable=CT_DC_SE)
carpal_tunnels_sq.place(x = 950, y = 250)

line = Label(window, text='----------------------------------------')
line.place(x = 950, y = 290)

# Image
image_dc = Label(window, text='Image DC :')
image_dc.place(x = 950, y = 330)

median_nerve_ig = Label(window, text='median_nerve :', textvariable=MN_DC_IM)
median_nerve_ig.place(x = 950, y = 370)

flexor_tendons_ig = Label(window, text='flexor_tendons :', textvariable=FT_DC_IM)
flexor_tendons_ig.place(x = 950, y = 410)

carpal_tunnels_ig = Label(window, text='carpal_tunnels :', textvariable=CT_DC_IM)
carpal_tunnels_ig.place(x = 950, y = 450)

## picturebox

T1_label = Canvas(window, width=128, height=128, bg='white')
T1_label.place(x = 100, y = 200)

T1_block = Label(text='T1').place(x=160, y=175)

T2_label = Canvas(window, width=128, height=128, bg='white')
T2_label.place(x = 100, y = 400)

T2_block = Label(text='T2').place(x=160, y=375)

CT_label = Canvas(window, width=128, height=128, bg='white')
CT_label.place(x = 300, y = 150)

CT_block = Label(text='CT GT').place(x=350, y=127)

FT_label = Canvas(window, width=128, height=128, bg='white')
FT_label.place(x = 300, y = 320)

FT_block = Label(text='FT GT').place(x=350, y=297)

MN_label = Canvas(window, width=128, height=128, bg='white')
MN_label.place(x = 300, y = 490)

MN_block = Label(text='MN GT').place(x=350, y=467)

predict_CT_label = Canvas(window, width=128, height=128, bg='white')
predict_CT_label.place(x = 500, y = 150)

CT_P_block = Label(text='CT Predict').place(x=540, y=127)

predict_FT_label = Canvas(window, width=128, height=128, bg='white')
predict_FT_label.place(x = 500, y = 320)

FT_P_block = Label(text='FT Predict').place(x=540, y=297)

predict_MN_label = Canvas(window, width=128, height=128, bg='white')
predict_MN_label.place(x = 500, y = 490)

MN_P_block = Label(text='MN Predict').place(x=540, y=467)

predict_mark_label = Canvas(window, width=128, height=128, bg='white')
predict_mark_label.place(x = 700, y = 320)

result_block = Label(text='Result').place(x=750, y=297)

window.mainloop()
#start

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image



# Functions:
# Save Results

net = None


def save_output(image_name, pred, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')

    imo.save(image_name)
# Remove Background From Image (Generate Mask, and Final Results)


def removeBg(imagePath , save_path ):
    
    global net
    # convert string of image data to uint8
    with open(imagePath, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return '---Empty image---'

    # decode image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        # build a response dict to send back to client
        return "---Empty image---"

    # save image to inputs
    
    cv2.imwrite(save_path, img)

    # processing
    image = transform.resize(img, (192, 256), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    save_output(save_path, pred, 'image')
    #save_output(save_path,pred, 'mask')
    
def maskBg(imagePath , save_path ):
    
    global net
    # convert string of image data to uint8
    with open(imagePath, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return '---Empty image---'

    # decode image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        # build a response dict to send back to client
        return "---Empty image---"

    # save image to inputs
    
    cv2.imwrite(save_path, img)

    # processing
    image = transform.resize(img, (192, 256), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    #save_output(save_path, pred, 'image')
    save_output(save_path,pred, 'mask')
    

def removeBgAPI(imagePath , save_path ):
    global net
    model_dir = os.path.join('saved_models',
                             'u2net', 'u2net' + '.pth')
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    removeBg(imagePath , save_path )

def getMaskAPI(imagePath , save_path ):
    global net
    model_dir = os.path.join('saved_models',
                             'u2net', 'u2net' + '.pth')
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    maskBg(imagePath , save_path )

if __name__=='__main__':

    model_dir = os.path.join('saved_models',
                             'u2net', 'u2net' + '.pth')
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))



    imgPath = "lady.jpg"  # Change this to your image path
    print(removeBg(imgPath , '/'))


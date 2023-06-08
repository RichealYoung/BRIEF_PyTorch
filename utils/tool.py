import numpy as np
import os
import tifffile
import cv2
import logging
from numpy import prod, zeros  

def get_type_max(data):
    dtype = data.dtype.name
    if dtype == 'uint8':
        max = 255
    elif dtype == 'uint12':
        max = 4098
    elif dtype == 'uint16':
        max = 65535
    elif dtype == 'float32':
        max = 65535
    elif dtype == 'float64':
        max = 65535
    elif dtype == 'int16':
        max = 65535   
    else:
        raise NotImplementedError
    return max

def range_limit(data,range):
    l, h = range
    max = get_type_max(data)
    assert l>=0 and l<=h and h<=max, 'Improper range setting!'
    return [l,h]

def get_dimension(path):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        dimension = 3
    elif postfix in ['.mp4']:
        dimension = 3
    elif postfix in ['.png','.jpg']:
        dimension = 2
    else:
        raise NotImplemented
    return dimension   

def read_video(videoPath:str):
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    suc = cap.isOpened()  # 
    img = None
    while suc:
        suc, frame = cap.read()
        if np.shape(img) == () and np.shape(frame) != ():
            shape = [count]+list(frame.shape)
            img= np.zeros(shape).astype(frame.dtype)
            n = 0
        if np.shape(frame) != ():
            img[n] = frame
            n += 1
    return img

def save_video(fsp:int,video_path:str,imgs:np.array):
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    heighth = imgs.shape[1]
    width = imgs.shape[2]
    video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width,heighth))
    for img in imgs:
        video_out.write(img)
    video_out.release()

# 3d->dhw thwc 2d->hwc
def read_img(path):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        img = tifffile.imread(path)
        if len(img.shape) == 3:
            img = img[...,None]
        assert len(img.shape)==4
    elif postfix in ['.mp4']:
        img = read_video(path)
        if len(img.shape) == 3:
            img = img[...,None]
        assert len(img.shape)==4
    elif postfix in ['.png','.jpg']:
        img = cv2.imread(path,-1)
        if len(img.shape) == 2:
            img = img[...,None]
        assert len(img.shape)==3
    else:
        raise NotImplemented
    return img  

def save_img(path,img):
    postfix = os.path.splitext(path)[-1]
    if postfix in ['.tif','.tiff']:
        tifffile.imsave(path,img)
    elif postfix in ['.mp4']:
        save_video(25,path,img) # fps=25
    elif postfix in ['.png','.jpg']:
        cv2.imwrite(path,img)  
    else:
        raise NotImplemented    
 
def yuv_import(filename,dims,numfrm,startfrm,type):  
    fp=open(filename,'rb')  
    if type == '8bit':
        blk_size = prod(dims)*3//2 # Y U V
    elif type == '10bit':
        blk_size = prod(dims)*3//2*2 # Y U V
    fp.seek(blk_size*startfrm,0)  
    Y=[]  
    U=[]  
    V=[]  
    d00=dims[0]//2  
    d01=dims[1]//2  
    Yt=zeros((dims[0],dims[1]),np.uint8,'C')  
    Ut=zeros((d00,d01),np.uint8,'C')  
    Vt=zeros((d00,d01),np.uint8,'C')  
    for i in range(numfrm):  
        for m in range(dims[0]):  
            for n in range(dims[1]):  
                if type == '8bit':
                    Yt[m,n] = ord(fp.read(1))
                elif type == '10bit':
                    Yt[m,n] = (ord(fp.read(1)) + ord(fp.read(1))*255)//4 
        for m in range(d00):  
            for n in range(d01):  
                if type == '8bit':
                    Ut[m,n] = ord(fp.read(1))
                elif type == '10bit':
                    Ut[m,n] = (ord(fp.read(1)) + ord(fp.read(1))*255)//4 
        for m in range(d00):  
            for n in range(d01):  
                if type == '8bit':
                    Vt[m,n] = ord(fp.read(1))
                elif type == '10bit':
                    Vt[m,n] = (ord(fp.read(1)) + ord(fp.read(1))*255)//4 
        Y=Y+[Yt]  
        U=U+[Ut]  
        V=V+[Vt]  
    fp.close()  
    return (Y,U,V)  

def yuv2bgr(filename,height,width,numfrm,startfrm,type):  
    fp=open(filename,'rb')  
    if type == '8bit':
        blk_size = height*width*3//2
    elif type == '10bit':
        blk_size = height*width*3//2*2
    fp.seek(blk_size*startfrm,0)  
    yuv_video = []
    bgr_video = []
    Yt=zeros((height,width),np.uint8,'C')  
    Ut=zeros((height//2,width//2),np.uint8,'C')  
    Vt=zeros((height//2,width//2),np.uint8,'C')  
    for i in range(numfrm):
        for m in range(height):  
            for n in range(width):  
                if type == '8bit':
                    Yt[m,n] = ord(fp.read(1))
                elif type == '10bit':
                    Yt[m,n] = (ord(fp.read(1)) + ord(fp.read(1))*255)//4 
        for m in range(height//2):  
            for n in range(width//2):  
                if type == '8bit':
                    Ut[m,n] = ord(fp.read(1))
                elif type == '10bit':
                    Ut[m,n] = (ord(fp.read(1)) + ord(fp.read(1))*255)//4 
        for m in range(height//2):  
            for n in range(width//2):   
                if type == '8bit':
                    Vt[m,n] = ord(fp.read(1))
                elif type == '10bit':
                    Vt[m,n] = (ord(fp.read(1)) + ord(fp.read(1))*255)//4 
        yuv_img = np.concatenate((Yt.reshape(-1), Ut.reshape(-1), Vt.reshape(-1)))
        yuv_img = yuv_img.reshape((height*3//2, width)).astype('uint8') # YUV ：NV12（YYYY UV）
        bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_I420) #  YUV 
        yuv_video.append(yuv_img)
        bgr_video.append(bgr_img[600:600+1024,1340:1340+1024])
        print("Extract frame %d " % (i + 1))
    yuv_video = np.array(yuv_video)
    yuv_video = yuv_video
    bgr_video = np.array(bgr_video)
    fp.close()  
    return yuv_video, bgr_video


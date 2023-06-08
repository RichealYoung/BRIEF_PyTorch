from utils.tool import read_img, save_img
import time
import os
import argparse

def alpha(x):
    return 0.8*(2**(x/6)-1)

def beta(x):
    return 0.5*x-7

def clip(x,min,max):
    if x < min:
        x = min
    if x > max:
        x = max
    return x

def judge_filter(p1,p0,q0,q1,index_a,index_b,thres):
    if (p1+p0+q0+q1)/4 > thres: # The block artifacts of high brightness area is not obvious
        return False
    if abs(p0-q0) < alpha(index_a):
        if abs(p1-p0) < beta(index_b) and abs(q1-q0) < beta(index_b):
            return True
    return False

def filter(p2,p1,p0,q0,q1,q2,index_b):
    # basic filter operation
    delta0 = (4*(q0-p0)+(p1-q1)+4)/8
    deltap1 = (p2+(p0+q0+1)/2-2*p1)/2
    deltaq1 = (q2+(q0+p0+1)/2-2*q1)/2
    # clipping
    table = [[20]] # TODO
    c1 = table[0][0]
    c0 = c1
    if abs(p2-p0) < beta(index_b): # luminance
        c0 += 1
    if abs(q2-q0) < beta(index_b): # luminance
        c0 += 1
    delta0 = clip(delta0,-c0,c0)
    deltap1 = clip(deltap1,-c1,c1)
    deltaq1 = clip(deltaq1,-c1,c1)
    # result
    p1 += deltap1
    p0 += delta0
    q0 -= delta0
    q1 += deltaq1
    return p1,p0,q0,q1

def filter2d(p,img,index_a,index_b,thres):
    x1,y1,x2,y2 = p
    x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
    if x1 == x2:
        dir = 0
        if x1 - 3 < 0 or x1 + 3 > img.shape[1] -1:
            return img
    elif y1 == y2:
        dir = 1
        if y1 - 3 < 0 or y1 + 3 > img.shape[0] -1:
            return img
    for y in range(y1,y2+1):
        for x in range(x1,x2+1):
            if dir == 0: 
                p2,p1,p0,q0,q1,q2 = img[y,int(x-3):int(x+3)]
                p2,p1,p0,q0,q1,q2 = float(p2),float(p1),float(p0),float(q0),float(q1),float(q2)
                # print(f'1:{p1},{p0},{q0},{q1}')
                if judge_filter(p1,p0,q0,q1,index_a,index_b,thres):
                    p1,p0,q0,q1 = filter(p2,p1,p0,q0,q1,q2,index_b)
                    img[y,int(x-2):int(x+2)] = int(p1),int(p0),int(q0),int(q1)
            elif dir == 1: 
                p2,p1,p0,q0,q1,q2 = img[int(y-3):int(y+3),x]
                p2,p1,p0,q0,q1,q2 = float(p2),float(p1),float(p0),float(q0),float(q1),float(q2)
                # print(f'1:{p1},{p0},{q0},{q1}')
                if judge_filter(p1,p0,q0,q1,index_a,index_b,thres):
                    p1,p0,q0,q1 = filter(p2,p1,p0,q0,q1,q2,index_b)
                    img[int(y-2):int(y+2),x] = int(p1),int(p0),int(q0),int(q1)
    return img

def main(step_dir:str, index_a:float, index_b:float, thres:float):
    decompressed_dir = os.path.join(step_dir,'decompressed')
    save_dir = os.path.join(step_dir,'deblock')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    origin_name = os.listdir(decompressed_dir)[0]
    save_name = origin_name[:-4] + '_deblocked_python.tif'
    save_path = os.path.join(save_dir, save_name)
    img_path = os.path.join(step_dir,'decompressed',origin_name)
    module_dir = os.path.join(step_dir,'compressed/module')
    # deblocking
    img = read_img(img_path)
    info = f'index_a:{index_a},index_b:{index_b},thres:{thres}'
    print(info)
    block_infos = os.listdir(module_dir)
    lines = []
    if len(img.shape) == 3: # 2d image, hwc
        for block_info in block_infos:
            h, w = block_info.split('-')
            x1, x2 = w.split('_')[1:]
            y1, y2 = h.split('_')[1:]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            if not [x1, y1, x1, y2] in lines:
                lines.append([x1, y1, x1, y2])
            if not [x2, y1, x2, y2] in lines:
                lines.append([x2, y1, x2, y2])
            if not [x1, y1, x2, y1] in lines:
                lines.append([x1, y1, x2, y1])
            if not [x1, y2, x2, y2] in lines:
                lines.append([x1, y2, x2, y2])
        for k in range(img.shape[-1]):
            for p in lines:
                img[:,:,k] = filter2d(p,img[:,:,k],index_a,index_b,thres)
        save_img(save_path,img)
    elif len(img.shape) == 4: # 3d image, dhwc
        for block_info in block_infos:
            d, h, w = block_info.split('-')
            z1, z2 = d.split('_')[1:]
            x1, x2 = w.split('_')[1:]
            y1, y2 = h.split('_')[1:]
            x1, x2, y1, y2, z1, z2 = int(x1), int(x2), int(y1), int(y2), int(z1), int(z2)
            l_flag = 1 if [z1, x1, y1, x1, y2] in lines else 0
            r_flag = 1 if [z1, x2, y1, x2, y2] in lines else 0
            d_flag = 1 if [z1, x1, y1, x2, y1] in lines else 0
            u_flag = 1 if [z1, x1, y2, x2, y2] in lines else 0
            for i in range(z1,z2+1):
                if l_flag == 0:
                    lines.append([i, x1, y1, x1, y2])
                if r_flag == 0:
                    lines.append([i, x2, y1, x2, y2])
                if d_flag == 0:
                    lines.append([i, x1, y1, x2, y1])
                if u_flag == 0:
                    lines.append([i, x1, y2, x2, y2])
        for k in range(img.shape[-1]):
            for p in lines:
                img[p[0],:,:,k] = filter2d(p[1:],img[p[0],:,:,k],index_a,index_b,thres)
        save_img(save_path,img)

if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(description='Deblock')
    parser.add_argument('-stp', type=str, default="", help='step path')
    args = parser.parse_args()
    # Although the high brightness area has block artifacts, it is not obvious visually. 
    # Using deblocking filtering may aggravate the block artifacts in high brightness area.
    # The user needs to select the threshold according to the decompressed data.
    index_a, index_b, thres = 51, 2000, 65535
    step_dir = args.stp

    main(step_dir, index_a, index_b, thres)
    print('The run time is:'+str(time.time()-time_start))
    
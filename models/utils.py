# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import sys
import numpy as np
import cv2
import torch
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def print_indented(n_sp, *args):
    if n_sp >= 0:
        print('  ' * n_sp, *args)


def letterboxing_opencv(image, wh_tgt, letterbox_type, n_sp, only_return_image = True, means_4_pad = None, interpolation = None):
    '''resize image with unchanged aspect ratio using padding'''
    print_indented(n_sp, 'letterboxing_opencv START');
    #iw, ih = image.shape[0:2][::-1]
    is_color = len(image.shape) > 2
    h_src, w_src = image.shape[:2]
    w_tgt, h_tgt = wh_tgt
    scale = min(w_tgt / w_src, h_tgt / h_src)
    if abs(scale - 1.0) > 1e-5:  
        w_new = int(w_src * scale); h_new = int(h_src * scale)
        #print('w_new :', w_new);    exit()
        #print('image.shape :', image.shape);    exit()
        if interpolation:
            image = cv2.resize(image, (w_new, h_new), interpolation = interpolation)
        else:     
            image = cv2.resize(image, (w_new, h_new), interpolation = cv2.INTER_CUBIC)
    else:
        w_new = w_src;  h_new = h_src;       
    if 'top_left' == letterbox_type:
        x_offset = 0;   y_offset = 0
        x_padding = w_tgt - w_new;  y_padding = h_tgt - h_new;
    elif 'center' == letterbox_type:
        x_offset = (w_tgt - w_new) // 2;    y_offset = (h_tgt - h_new) // 2
        x_padding = x_offset;               y_padding = y_offset;
    else:
        raise NameError('Invalid letterbox_type')        
    if is_color:
        chn = image.shape[2]
        new_image = np.zeros((h_tgt, w_tgt, chn), np.float32)
    else:
        new_image = np.zeros((h_tgt, w_tgt), np.float32)
        
    #print('new_image.dtype : ', new_image.dtype); exit(); 
    if means_4_pad:
        #new_image.fill(128)
        new_image[...] = means_4_pad
    #new_image[dy:dy+nh, dx:dx+nw,:] = image
    #new_image[y_offset : y_offset + h_new, x_offset : x_offset + w_new, :] = image
    '''
    print('new_image.shape :', new_image.shape);  #exit(); 
    print('w_new :', w_new);   print('h_new :', h_new);  #exit(); 
    print('x_offset :', x_offset);   print('y_offset :', y_offset);  exit(); 
    '''
    new_image[y_offset : y_offset + h_new, x_offset : x_offset + w_new] = image
    print_indented(n_sp, 'letterboxing_opencv END');
    if only_return_image:
        return new_image
    else:
        return new_image, (w_new, h_new), (w_src, h_src), (x_offset, y_offset) 



class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dtdccecfdewcdwsxd = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath, shall_letterbox, is_color, resize=None):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        print('impath :', impath)
        #grayim = cv2.imread(impath, 0)
        im = cv2.imread(impath, 1 if is_color else 0)
        #if grayim is None:
        if im is None:
            raise Exception('Error reading image %s' % impath)
        #w, h = grayim.shape[1], grayim.shape[0]
        w, h = im.shape[1], im.shape[0]
        wh_new = process_resize(w, h, resize if resize else self.resize)
        x_offset = 0; y_offset = 0;
        '''
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        '''
        if shall_letterbox:
            n_sp = 0;
            '''
            im_bgr = cv2.imread(impath)
            print('im_bgr.shape b4 :', im_bgr.shape);  #exit()
            im_bgr = letterboxing_opencv(im_bgr, (w_new, h_new), 'center', n_sp + 1)
            print('im_bgr.shape after :', im_bgr.shape);  #exit()
            print('grayim.shape b4 :', grayim.shape);  #exit()
            '''
            #grayim, wh_new, wh_src, xy_offset = letterboxing_opencv(grayim, wh_new, 'center', n_sp + 1, False, 128)
            im, wh_new, wh_src, xy_offset = letterboxing_opencv(im, wh_new, 'center', n_sp + 1, False, 128)
            #print('grayim.shape after :', grayim.shape);  exit()
        else:
            #grayim = cv2.resize(grayim, (w_new, h_new), interpolation=self.interp)
            im = cv2.resize(im, (w_new, h_new), interpolation=self.interp)
        #print('x_offset :', x_offset);   print('y_offset :', y_offset);    exit();
        im = im.astype('uint8')
        #print('im.dtype :', im.dtype);    exit();
        #return grayim, xy_offset, wh_new
        return im, xy_offset, wh_new

    def next_frame(self, shall_letterbox, is_color):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """
        x_offset = 0;   y_offset = 0;

        if self.i == self.max_length:
            return False, None, x_offset, y_offset, -1, -1
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return False, None, x_offset, y_offset
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            wh_new = process_resize(w, h, self.resize)
            if not is_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if shall_letterbox:
                image, w_new, h_new, w_src, h_src, x_offset, y_offset = letterboxing_opencv(image, wh_new, 'center', n_sp + 1, True, 128)
            else:     
                image = cv2.resize(image, wh_new, interpolation=self.interp)
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            #print('image_file :', image_file);    #exit()
            #print('is_color :', is_color);    exit()
            image, xy_offset, wh_new = self.load_image(image_file, shall_letterbox, is_color)
            #print('image[0, 0] :', image[0, 0]);  print('image[-1, -1] :', image[-1, -1]);    exit()
        self.i = self.i + 1
        return True, image, xy_offset, wh_new

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    #print('w :', w, ', h :', h, ', resize :', resize);  exit()
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return (w_new, h_new)



def get_rotated_position_when_image_is_rotated_around_center(rot_deg, hw, li_xy):
    n_dim = li_xy.ndim
    rad = np.deg2rad(rot_deg)
   
    #xy_center_old = (hw[1] * 0.5, hw[0] * 0.5)
    #hw = (400, 600)
    xy_center_old = ((hw[1] - 1) * 0.5, -(hw[0] - 1) * 0.5)
    #print('xy_center_old :', xy_center_old);    #exit()
    p_rotated_corners = rotate_1st_around_2nd_by_3rd_radian_and_shift_by_4th(((0, 0), (hw[1] - 1, 0), (hw[1] - 1, -(hw[0] - 1)), (0, -(hw[0] - 1))), xy_center_old, rad)
    x_min = min(p_rotated_corners[:, 0]); x_max = max(p_rotated_corners[:, 0]) 
    y_min = min(p_rotated_corners[:, 1]); y_max = max(p_rotated_corners[:, 1]) 
    #print('x_min :', x_min);    print('x_max :', x_max);    #exit();
    #print('y_min :', y_min);    print('y_max :', y_max);    #exit();
    #print('p_rotated_corners b4 :');   print(p_rotated_corners);    #exit()
    xy_center_new = ((x_max - x_min) * 0.5, -(y_max - y_min) * 0.5)
    #print('xy_center_new :', xy_center_new);    #exit()
    p_rotated = rotate_1st_around_2nd_by_3rd_radian_and_shift_by_4th(li_xy, xy_center_old, rad, xy_center_new)
    if 1 == n_dim: 
        p_rotated[1] *= -1
    else:     
        p_rotated[:, 1] *= -1
    #print('p_rotated :');   print(p_rotated);    exit()
    #exit()
    return p_rotated
    
def rotate_1st_around_2nd_by_3rd_radian_and_shift_by_4th(p, origin=(0, 0), radians=0, shift=(0, 0)):
    #angle = np.deg2rad(degrees)
    #print('radians :', radians);    exit()
    R = np.array([[np.cos(radians), -np.sin(radians)],
                  [np.sin(radians),  np.cos(radians)]])
    o = np.atleast_2d(origin)
    s = np.atleast_2d(shift)
    if 0 == np.sum(s):
        s = o
    p = np.atleast_2d(p)
    #print('p.shape :', p.shape);    exit()
    t0 = np.squeeze((R @ (p.T - o.T) + s.T).T)
    #print('p :', p);	#exit()
    #print('t0 :', t0);  #exit()
    return np.squeeze((R @ (p.T - o.T) + s.T).T)


def unrotate_keypoints_2(kpts, hw, n_rots, deg_per_rot):
    #print('hw :', hw);  exit()
    for iP, n_rot in enumerate(n_rots):
        if 0 == n_rot:
            continue
        rot_deg = deg_per_rot * n_rot
        kpt = kpts[iP]
        #print('kpt b4 :', kpt);
        kpt[1] *= -1
        #print('kpt after :', kpt);  #exit()
        kpt_rotated = get_rotated_position_when_image_is_rotated_around_center(rot_deg, hw, kpt)
        #print('kpt_rotated :', kpt_rotated);  exit()
        kpts[iP] = kpt_rotated
    return kpts       

def unrotate_keypoints(kp_data, hw, n_rot, device, key = 'keypoints'):
    #xy_center = (im_bgr_or_gray_init.shape[1] * 0.5, -im_bgr_or_gray_init.shape[0] * 0.5)
    #hw = (sheip[2], sheip[3]) 
    batch_size = len(kp_data[key])
    #print('type(kp_data[key]) :', type(kp_data[key]));
    #print('len(kp_data[key] :', len(kp_data[key])); exit()
    if batch_size <= 1:
        return kp_data
    #li_kpts_rotated = [last_data[key][0]]
    #for iR in range(n_rot_src + 1):
    #print('kp_data[key][-1][0] b4:', kp_data[key][-1][0]) 
    for iR in range(n_rot):
        #rot_deg = -90.0 * iR
        rot_deg = -90.0 * (iR + 1)
        #ar_xy = last_data['keypoints'][iR].cpu().numpy()
        kpts = kp_data[key][iR + 1].cpu().numpy()
        kpts[:, 1] *= -1
        kpts_rotated = get_rotated_position_when_image_is_rotated_around_center(rot_deg, hw, kpts)
        kp_data[key][iR + 1] = torch.from_numpy(kpts_rotated).float().to(device)
        #li_kpts0_rotated.append(kpts0_rotated)
    #kpts0 = np.vstack(li_kpts0_rotated)
    #print('kp_data[key][-1][0] after :', kp_data[key][-1][0]);  exit() 
    return kp_data

def aggregate_kp_data(kp_data):
    batch_size = len(kp_data['keypoints'])
    if batch_size <= 1:
        return kp_data
    '''
    len_sum = 0     
    for iI in range(batch_size):
        len_cur = kp_data['scores'][iI].shape[0]
        len_sum += len_cur
    print('len_sum :', len_sum);    #exit(()
    '''
    for key in kp_data:
        '''
        li_shape = list(kp_data[key][0].shape)
        print('li_shape :', li_shape);
        li_shape[0] = len_sum
        tensor_agg = torch.Tensor(torch.Size(li_shape))
        print('tensor_agg.shape b4 :', tensor_agg.shape);
        torch.cat(kp_data[key], out = tensor_agg)
        '''
        #print('\nkey :', key);
        #print('kp_data[key][0].shape :', kp_data[key][0].shape);    #exit()
        #print('kp_data[key][-1].shape :', kp_data[key][-1].shape);    #exit()
        tensor_agg = torch.cat(kp_data[key], 1 if 'descriptors' == key else 0)
        #print('key :', key, ', tensor_agg.shape :', tensor_agg.shape);    #exit()
        #print('tensor_agg_tmp.shape :', tensor_agg_tmp.shape);    exit()
        #print('kp_data[key][0].is_cuda b4 :', kp_data[key][0].is_cuda) 
        #print('kp_data[key][0].dtype b4 :', kp_data[key][0].dtype) 
        kp_data[key] = [tensor_agg]
        #print('kp_data[key][0].is_cuda after :', kp_data[key][0].is_cuda) 
        #print('kp_data[key][0].dtype after :', kp_data[key][0].dtype);   exit()
    #exit()
    return kp_data        
        
def sort_according_2_confidence_and_trim(data, key_conf, li_key, len_trim):
    '''
    t0 = len(data[key_conf])
    if 4 != t0:
        for key in data:
            print('\nkey :', key);  
            if torch.is_tensor(data[key]):
                print('data[key].shape :', data[key].shape)
            else:
                print('data[key] :', data[key])
                
        exit()
    '''     
    _, t_idx_sorted = torch.sort(data[key_conf], descending = True)
    #print('t_idx_sorted :', t_idx_sorted);
    for key in li_key:
        #print('key "{}" will be sorted'.format(key))
        #print('data[key] b4 :', data[key])
        data[key] = reindex_and_trim_tensor(data[key], 0, t_idx_sorted, len_trim)
        #print('data[key] after :', data[key])
        #exit()
    #exit()     
    return data      

def sort_and_trim_kp_data(kp_data):
    #print('type(kp_data[keypoints]) :', type(kp_data['keypoints']));    exit()
    batch_size = len(kp_data['keypoints'])
    if batch_size <= 1:
        return kp_data
    len_min = sys.maxsize
    for iI in range(batch_size):
        len_cur = kp_data['scores'][iI].shape[0]
        if len_cur < len_min:
            len_min = len_cur
    for iI in range(batch_size):
        _, t_idx_sorted = torch.sort(kp_data['scores'][iI], descending = True)
        for key in kp_data:
            '''
            print('\nkey :', key);    #exit()
            print('type(kp_data[key]) :', type(kp_data[key]));    #exit()
            print('len(kp_data[key]) :', len(kp_data[key]));    #exit()
            print('kp_data[key][iI].shape b4) :', kp_data[key][iI].shape);    #exit()
            print('t_idx_sorted.shape) :', t_idx_sorted.shape);    #exit()
            '''
            dim = 1 if 'descriptors' == key else 0
            kp_data[key][iI] = reindex_and_trim_tensor(kp_data[key][iI], dim, t_idx_sorted, len_min)          
            #print('kp_data[key][iI].shape after) :', kp_data[key][iI].shape);    #exit()
    #exit()   
    return kp_data             

def reindex_and_trim_tensor(t_ori, dim, t_idx, len_new):
    '''
    x = torch.randn(3, 2, 4)
    print(x)
    for i1 in range(3):
        print('i1 :', i1)
        for i2 in range(3):
            print('\ti2 :', i2)
            t0 = torch.transpose(x, i1, i2)
            print('\t\tt0 :', t0)
            print('\t\tt0.shape :', t0.shape)
    exit()                
    if 0 != dim:
        tsr = torch.transpose(t_ori, 0, dim)
    '''
    #print('t_ori.shape :', t_ori.shape);    exit()
    t_trans = torch.transpose(t_ori, 0, dim)
    t_sorted = t_trans[t_idx]
    t_trim = t_sorted.narrow(0, 0, len_new) 
    return torch.transpose(t_trim, 0, dim)
    
    
    

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[], show_homography=False, x_offset=0, y_offset=0):
    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text, show_homography)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()

def is_left(xy_a, xy_b, xy_c):
    return ((xy_b[0] - xy_a[0]) * (xy_c[1] - xy_a[1]) - (xy_b[1] - xy_a[1]) * (xy_c[0] - xy_a[0])) > 0


def degree_between_3_points(xy_a, xy_b, xy_c):
    a = xy_a;   b = xy_b;   c = xy_c;
    ba = a - b; bc = c - b
    cosine_rad = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    rad = np.arccos(cosine_rad)
    deg = np.degrees(rad)
    return deg

def rectness_check(ar_xy, th_deg):
    is_rectful = True
    n_corner = ar_xy.shape[0]
    if n_corner > 2:
        ar_xy_aug = np.vstack((ar_xy, ar_xy[0 : 2]))
        #is_left_012 = is_left(ar_xy_aug[0], ar_xy_aug[1], ar_xy_aug[2])
        for idx in range(n_corner):
            deg = degree_between_3_points(ar_xy_aug[idx], ar_xy_aug[idx + 1], ar_xy_aug[idx + 2])
            if deg < th_deg or deg > 180 - th_deg:
                is_rectful = False
                break
    else:
        is_rectful = False
    return is_rectful



def is_polygon_convex(ar_xy):
    is_convex = True
    n_corner = ar_xy.shape[0]
    if n_corner > 2:
        ar_xy_aug = np.vstack((ar_xy, ar_xy[0 : 2]))
        is_left_012 = is_left(ar_xy_aug[0], ar_xy_aug[1], ar_xy_aug[2])
        for idx in range(3, n_corner):
            is_left_cur = is_left(ar_xy_aug[idx - 2], ar_xy_aug[idx - 1], ar_xy_aug[idx])
            if is_left_cur != is_left_012:
                is_convex = False
                break
    else:
        is_convex = False
    return is_convex


def is_inside_enough(ar_xy, ar_wh, r_margin_inside):
    dim = ar_wh.shape[0]
    wh_margin = ar_wh * r_margin_inside
    is_inside = True
    for xy in ar_xy:
        for idx in range(dim):
            if xy[idx] < -wh_margin[idx] or xy[idx] > ar_wh[idx] + wh_margin[idx]:
                is_inside = False
                break
        if not is_inside:
            break
    return is_inside 

    


def is_polygon_degenerated(ar_xy, th_dist):
    
    is_degenerated = False
    n_corner = ar_xy.shape[0] 
    for idx in reversed(range(n_corner)):
        dist = np.linalg.norm(ar_xy[idx, :] - ar_xy[idx - 1, :])
        if dist < th_dist:
            is_degenerated = True
            break
    return is_degenerated


def is_valid_homography(mat_homo, arr_xy_lt_src, arr_wh_src, arr_wh_tgt, r_margin_inside, th_dist_degen, th_deg_rectness):
    
    is_valid = False
    arr_xy_lt_src = arr_xy_lt_src.astype(np.float32)
    arr_lt_rt_rb_lb_src = np.tile(arr_xy_lt_src, (4, 1, 1))
    arr_lt_rt_rb_lb_src[1, :, 0] += arr_wh_src[0] - 1;
    arr_lt_rt_rb_lb_src[2, :] += arr_wh_src - 1;
    arr_lt_rt_rb_lb_src[3, :, 1] += arr_wh_src[1] - 1;
    '''
    print('mat_homo.shape :', mat_homo.shape)
    print('mat_homo.dtype :', mat_homo.dtype)
    print('arr_xy_lt_src.dtype :', arr_xy_lt_src.dtype)
    print('arr_lt_rt_rb_lb_src.shape :', arr_lt_rt_rb_lb_src.shape);  #exit();
    '''
    arr_lt_rt_rb_lb_tgt = cv2.perspectiveTransform(arr_lt_rt_rb_lb_src, mat_homo)
    '''
    print('arr_lt_rt_rb_lb_src :', arr_lt_rt_rb_lb_src);
    print('arr_lt_rt_rb_lb_tgt :', arr_lt_rt_rb_lb_tgt);    #exit()
    '''
    arr_lt_rt_rb_lb_tgt = np.squeeze(arr_lt_rt_rb_lb_tgt)
    '''
    print('type(mat_homo) :', type(mat_homo));
    print('mat_homo.shape :', mat_homo.shape);
    '''
    h_00 = mat_homo[0, 0];
    h_11 = mat_homo[1, 1]; 
    h_01 = mat_homo[0, 1]; 
    h_10 = mat_homo[1, 0];  
    h_20 = mat_homo[2, 0]; 
    h_21 = mat_homo[2, 1];
    det = h_00 * h_11 - h_01 * h_10
    is_positive_det = det > 0
    if is_positive_det:
        norm_1 = math.sqrt(h_00 * h_00 + h_10 * h_10)
        #print('norm_1 :', norm_1)
        if 0.1 < norm_1 and norm_1 < 4:
            norm_2 = math.sqrt(h_01 * h_01 + h_11 * h_11)
            #print('norm_2 :', norm_2)
            if 0.1 < norm_2 and norm_2 < 4:
                norm_3 = math.sqrt(h_20 * h_20 + h_21 * h_21)
                #print('norm_3 :', norm_3)
                if norm_3 < 0.002:
                    is_inside = is_inside_enough(arr_lt_rt_rb_lb_tgt, arr_wh_tgt, r_margin_inside)
                    #print('is_inside :', is_inside)
                    if is_inside:
                        is_degenerated = is_polygon_degenerated(arr_lt_rt_rb_lb_tgt, th_dist_degen)
                        #print('is_degenerated :', is_degenerated)
                        if not is_degenerated:
                            is_convex = is_polygon_convex(arr_lt_rt_rb_lb_tgt)
                            #print('is_convex :', is_convex)
                            if is_convex:
                                #print('th_deg_rectness'   exit();
                                is_still_rect = rectness_check(arr_lt_rt_rb_lb_tgt, th_deg_rectness)
                                #print('is_still_rect :', is_still_rect)
                                #exit()
                                if is_still_rect:
                                    is_valid = True
                                    #exit()
                                    
    return is_valid                
            
        
def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[], show_homography=False, xy_offset_init=(0, 0), xy_offset=(0, 0), wh_new_init=(-1, -1), wh_new=(-1, -1)):
    if len(image0.shape) > 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
         
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    '''
    print('image0.shape :', image0.shape);  print('image1.shape :', image1.shape);  #exit(0);
    print('xy_offset_init :', xy_offset_init);  print('xy_offset :', xy_offset);  #exit(); 
    print('wh_new_init :', wh_new_init);  print('wh_new :', wh_new);  exit(); 
    '''
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)
    w_new_init, h_new_init = wh_new_init
    if show_homography and w_new_init > 0 and h_new_init > 0:
        is_there_matching_enough_4_homography = mkpts0.shape[0] >= 7
        if is_there_matching_enough_4_homography:
            r_margin_inside = 0.6
            th_dist_degen = 10
            #th_deg_rectness = 30
            #th_deg_rectness = 70
            th_deg_rectness = 75
            '''
            pts_src = np.float32([52, 376, 240, 528, 413, 291, 217, 266]).reshape(4, 1, -1)
            pts_dst = np.float32([56, 478, 387, 497, 376, 124, 148, 218]).reshape(4, 1, -1)
            h = cv2.findHomography(pts_src, pts_dst)
            h0 = h[0]
            print('pts_src.shape :', pts_src.shape);    #exit()
            print('pts_dst.shape :', pts_dst.shape);    #exit()
            print('h :', h);    #exit()
            print('h0.shape :', h0.shape);   # exit()
            '''
            #print('type(mkpts0) :', type(mkpts0));  exit(); 
            #M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 20.0)
            #M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 10.0)
            #M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            #M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 2.0)
            M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 1.0)
            #M, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 0.5)
            #print('mkpts0.shape :', mkpts0.shape);  print('mkpts1.shape :', mkpts1.shape);  #exit(); 
            x_offset_init, y_offset_init = xy_offset_init
            line_thick = 1; 
            #y_margin = int(0.5 * line_thick + 0.5)
            y_top = y_offset_init
            y_bot = y_top + h_new_init - 1
            x_left = x_offset_init
            x_right = x_left + w_new_init - 1
            #'''
            cv2.line(out, (x_left, y_top), (x_right, y_top), color=(0, 0, 255), thickness=line_thick, lineType=cv2.LINE_AA)
            cv2.line(out, (x_right, y_top), (x_right, y_bot), color=(0, 0, 255), thickness=line_thick, lineType=cv2.LINE_AA)
            cv2.line(out, (x_right, y_bot), (x_left, y_bot), color=(0, 0, 255), thickness=line_thick, lineType=cv2.LINE_AA)
            cv2.line(out, (x_left, y_bot), (x_left, y_top), color=(0, 0, 255), thickness=line_thick, lineType=cv2.LINE_AA)
            #'''

            if M is not None:
                '''
                print('M', M);  print('mask', inliers);  exit();
                '''
                line_thick_homo = 2
                p_lt_rt_rb_lb_init = np.float32([x_left, y_top, x_right, y_top, x_right, y_bot, x_left, y_bot]).reshape(4, 1, -1)
                '''
                print('p_lt_rt_rb_lb_init :', p_lt_rt_rb_lb_init);  #exit()
                print('p_lt_rt_rb_lb_init.shape :', p_lt_rt_rb_lb_init.shape);  #exit()
                print('M :', M);  #exit()
                print('type(M) :', type(M));  #exit()
                print('M.shape :', M.shape);  #exit()
                '''
                p_lt_rt_rb_lb = cv2.perspectiveTransform(p_lt_rt_rb_lb_init, M)
                #print('p_lt_rt_rb_lb.shape b4 :', p_lt_rt_rb_lb.shape)
                p_lt_rt_rb_lb = np.squeeze(p_lt_rt_rb_lb)
                #print('p_lt_rt_rb_lb.shape after :', p_lt_rt_rb_lb.shape)
                #exit()
                is_homography_valid = is_valid_homography(M, np.array([x_left, y_top]), np.array([w_new_init, h_new_init]), np.array([W1, H1]), r_margin_inside, th_dist_degen, th_deg_rectness)
                '''
                p_lt_init = np.array([x_left, y_top, 1])
                p_rt_init = np.array([x_right, y_top, 1])
                p_lb_init = np.array([x_left, y_bot, 1])
                p_rb_init = np.array([x_right, y_bot, 1])
                p_offset = np.array([margin + W0, 0, 0])
                p_lt = np.matmul(M, p_lt_init);   p_lt /= p_lt[-1]; p_lt = np.rint(p_lt + p_offset).astype(int)
                p_rt = np.matmul(M, p_rt_init);   p_rt /= p_rt[-1]; p_rt = np.rint(p_rt + p_offset).astype(int) 
                p_lb = np.matmul(M, p_lb_init);   p_lb /= p_lb[-1]; p_lb = np.rint(p_lb + p_offset).astype(int)
                p_rb = np.matmul(M, p_rb_init);   p_rb /= p_rb[-1]; p_rb = np.rint(p_rb + p_offset).astype(int) 
                p_lt_rt_rb_lb[:, :, :2] += p_offset[:2]
                print('p_lt.shape :', p_lt.shape) 
                print('p_lt :', p_lt) 
                print('p_rt :', p_rt) 
                print('p_rb :', p_rb) 
                print('p_lb :', p_lb);  exit() 
                '''
                p_offset = np.array([margin + W0, 0])
                #p_lt_rt_rb_lb[:, :] += p_offset
                p_lt_rt_rb_lb_int = np.rint(p_lt_rt_rb_lb + p_offset).astype(int)
                '''
                print('p_lt_rt_rb_lb :', p_lt_rt_rb_lb) 
                print('p_lt_rt_rb_lb_int :', p_lt_rt_rb_lb_int) 
                '''
                li_p_lt_rt_rb_lb_tuple = list(map(tuple, p_lt_rt_rb_lb_int))
                '''
                print('li_p_lt_rt_rb_lb_tuple :', li_p_lt_rt_rb_lb_tuple);  #exit()
                '''
                color_rect = (0, 0, 255) if is_homography_valid else (255, 0, 0)
                for idx in reversed(range(len(li_p_lt_rt_rb_lb_tuple))):
                    cv2.line(out, li_p_lt_rt_rb_lb_tuple[idx], li_p_lt_rt_rb_lb_tuple[idx - 1], color_rect, thickness=line_thick_homo, lineType=cv2.LINE_AA)
                    #cv2.line(out, li_p_lt_rt_rb_lb_tuple[idx], li_p_lt_rt_rb_lb_tuple[idx - 1], color=(0, 0, 255), thickness=line_thick_homo, lineType=cv2.LINE_AA)
                '''        
                for idx, pt in reversed(list(enumerate(li_p_lt_rt_rb_lb_tuple))):
                    cv2.line(out, (p_lt[0], p_lt[1]), (p_rt[0], p_rt[1]), color=(0, 0, 255), thickness=line_thick_homo, lineType=cv2.LINE_AA)
                cv2.line(out, (p_rt[0], p_rt[1]), (p_rb[0], p_rb[1]), color=(0, 0, 255), thickness=line_thick_homo, lineType=cv2.LINE_AA)
                cv2.line(out, (p_rb[0], p_rb[1]), (p_lb[0], p_lb[1]), color=(0, 0, 255), thickness=line_thick_homo, lineType=cv2.LINE_AA)
                cv2.line(out, (p_lb[0], p_lb[1]), (p_lt[0], p_lt[1]), color=(0, 0, 255), thickness=line_thick_homo, lineType=cv2.LINE_AA)
                '''
                #cv2.imwrite("temp.bmp", out);   exit();

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

#! /usr/bin/env python3
#
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
import argparse
import cv2
import numpy as np
import matplotlib.cm as cm
import torch
import features, globals

from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults

from models.matching import Matching
from models.utils import AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor, sort_and_trim_kp_data, get_rotated_position_when_image_is_rotated_around_center, unrotate_keypoints, aggregate_kp_data, sort_according_2_confidence_and_trim, unrotate_keypoints_2

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--target', type=str, default='0',
        help='Path to the image file of object to be detected.')

    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
   

    parser.add_argument('--weight', type=str, help="Path to the checkpoint of LoFTR.")

    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')

    parser.add_argument(
        '--kp_thres_cv', type=int, default=-1,
        help='OpenCV keypoint threshold')

    parser.add_argument(
        '--neg_src_0_none_pos_des', type=int, default=0,
        help='Rotate source image 4 times if negative.  No rotation if 0. Rotate destination image 4 times if positive.')

    parser.add_argument(
        '--neg_opencv_0_sg_pos_loftr', type=int, default=0,
        help='If negative, use opencv feature matching. If zero, use super-glue. Otherwise, use LoFTR.')


    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--ratio_thresh', type=float, default=0.7,
        help='Opencv feature matching ratio threshold between first and second best candidates')

    parser.add_argument(
        '--aggregate_rotations', action='store_true',
        help='Stack all keypoints of rotations into one vector')

    parser.add_argument(
        '--top_k', type=int, default=2000, help="The max vis_range (please refer to the code).")
    parser.add_argument(
        '--bottom_k', type=int, default=0, help="The min vis_range (please refer to the code).")


    parser.add_argument(
        '--letterboxing', action='store_true',
        help='Resize the input image using letterboxing')

    parser.add_argument(
        '--show_homography', action='store_true',
        help='Show planar homography.')
    '''
    parser.add_argument(
        '--use_opencv_feature_matching', action='store_true',
        help='Use good old fashioned OpenCV feature matching such as SIFT.')
    '''
    parser.add_argument(
        '--detector', type=str, default='BRISK',
        help='Opencv feature detector')

    parser.add_argument(
        '--descriptor', type=str, default='BRISK',
        help='Opencv feature descriptor')

    #parser.add_argument('--matcher', type=str, default='0', help='ID of a USB webcam, URL of an IP camera, or path to an image directory or movie file')
    parser.add_argument(
        '--matcher', choices={'BF', 'FLANN'}, default='FLANN',
        help='Opencv feature matcher method')


    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if opt.neg_src_0_none_pos_des < 0:
        n_rot_src = 3;  n_rot_des = 0
    elif opt.neg_src_0_none_pos_des > 0:
        n_rot_src = 0;  n_rot_des = 3
    else:     
        n_rot_src = 0;  n_rot_des = 0
    n_rot = max(n_rot_src, n_rot_des)
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    #exit()
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    '''
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'
    '''
    
    #print('frame.shape :', frame.shape);   cv2.imwrite("target.bmp", frame);   #exit()
    shall_read_color = False
    #if opt.use_opencv_feature_matching:
    if opt.neg_opencv_0_sg_pos_loftr < 0:
       
        # Initiate detector selected
        if opt.detector == 'SIFT':
            globals.detector = features.SIFT()

        elif opt.detector == 'SURF':
            globals.detector = features.SURF()

        elif opt.detector == 'KAZE':
            globals.detector = features.SIFT()

        elif opt.detector == 'ORB':
            globals.detector = features.ORB()

        elif opt.detector == 'BRISK':
            globals.detector = features.BRISK()

        elif opt.detector == 'AKAZE':
            globals.detector = features.AKAZE()

        # Initiate descriptor selected
        if opt.descriptor == 'SIFT':
            globals.descriptor = features.SIFT()

        elif opt.descriptor == 'SURF':
            globals.descriptor = features.SURF()

        elif opt.descriptor == 'KAZE':
            globals.descriptor = features.SIFT()

        elif opt.descriptor == 'BRIEF':
            globals.descriptor = features.BRIEF()

        elif opt.descriptor == 'ORB':
            globals.descriptor = features.ORB()

        elif opt.descriptor == 'BRISK':
            globals.descriptor = features.BRISK()

        elif opt.descriptor == 'AKAZE':
            globals.descriptor = features.AKAZE()

        elif opt.descriptor == 'FREAK':
            globals.descriptor = features.FREAK()

        #print('globals.detector.getThreshold() :', globals.detector.getThreshold());    exit()
        shall_read_color = 'BRISK' == opt.detector
        print('shall_read_color :', shall_read_color);  #exit() 
        if opt.kp_thres_cv >= 0:
            globals.detector.setThreshold(opt.kp_thres_cv)#    exit()
        k_thresh = globals.detector.getThreshold()#    exit()
        m_thresh = opt.ratio_thresh
        im_bgr_or_gray_init, xy_offset_init, wh_new_init = vs.load_image(opt.target, opt.letterboxing,
            shall_read_color)
        #print('im_init.shape :', im_init.shape);    #exit()

        # Find the keypoints and compute
        # the descriptors for input image
        globals.keypoints1, globals.descriptors1 = features.features(im_bgr_or_gray_init)
        features.prints(keypoints = globals.keypoints1, descriptor = globals.descriptors1)
    elif opt.neg_opencv_0_sg_pos_loftr >= 0:
        device = 'cuda' if (torch.cuda.is_available() and not opt.force_cpu) else 'cpu'
        print('Running inference on device \"{}\"'.format(device))
        im_bgr_or_gray_init, xy_offset_init, wh_new_init = vs.load_image(opt.target, opt.letterboxing,
            shall_read_color, resize=[min(opt.resize), min(opt.resize)] if n_rot else None)
        #cv2.imwrite('im_bgr_or_gray_init.bmp', im_bgr_or_gray_init);    exit()     
        li_frame_tensor = [frame2tensor(im_bgr_or_gray_init, device)]
        #print('li_frame_tensor[0].shape :', li_frame_tensor[0].shape);         #   [1, 1, 480, 640]
        #print('type(li_frame_tensor[0]) :', type(li_frame_tensor[0]));         #   [1, 1, 480, 640]
        #print('type(li_frame_tensor[0].data) :', type(li_frame_tensor[0].data));         #   [1, 1, 480, 640]
        #print('li_frame_tensor[0].type() :', li_frame_tensor[0].type());         #   [1, 1, 480, 640]
        #exit()
        for iR in range(n_rot_src):
            #if opt.neg_src_0_none_pos_des > 0:
            #    li_frame_tensor.append(li_frame_tensor[iR])
            #elif opt.neg_src_0_none_pos_des < 0:       
            li_frame_tensor.append(torch.rot90(li_frame_tensor[0], iR + 1, [2, 3]).float().to(device))
            #print('\niR : ', iR); 
            #print('li_frame_tensor[-2].type() :', li_frame_tensor[-2].type());   #   [1, 1, 480, 640]    
            #print('li_frame_tensor[-1].type() :', li_frame_tensor[-1].type());   #   [1, 1, 640, 480]
        #exit()
        sheip = li_frame_tensor[0].shape 
        frame_tensor = torch.Tensor(len(li_frame_tensor), sheip[1], sheip[2], sheip[3]) 
        torch.cat(li_frame_tensor, out = frame_tensor)
        frame_tensor = frame_tensor.to(device)


        if opt.neg_opencv_0_sg_pos_loftr > 0:
            opt.show_keypoints = False
            vis_range = [opt.bottom_k, opt.top_k]
            #vis_range = [0, 5]
            n_vis = vis_range[1] - vis_range[0]
            # Initialize LoFTR
            #print('kkk');   exit()
            matcher = LoFTR(config=default_cfg)
            #print('opt.weight :', opt.weight);  exit()
            matcher.load_state_dict(torch.load(opt.weight)['state_dict'])
            matcher = matcher.eval().to(device=device)
            #print('frame_tensor.shape after :', frame_tensor.shape);  exit() # [4, 1, 480, 480]
            last_data = {'image0': frame_tensor}
        else:

            config = {
                'superpoint': {
                    'nms_radius': opt.nms_radius,
                    'keypoint_threshold': opt.keypoint_threshold,
                    'max_keypoints': opt.max_keypoints
                },
                'superglue': {
                    'weights': opt.superglue,
                    'sinkhorn_iterations': opt.sinkhorn_iterations,
                    'match_threshold': opt.match_threshold,
                }
            }
            matching = Matching(config).eval().to(device)
            keys = ['keypoints', 'scores', 'descriptors']

            last_data = matching.superpoint({'image': frame_tensor})
            #print('last_data.keys() :', last_data.keys());               # exit()
            #print('type(last_data[keypoints]):', type(last_data['keypoints']));                exit()
            #last_data = unrotate_keypoints(last_data, (sheip[2], sheip[3]), n_rot_src, device, 'keypoints') 
            if opt.aggregate_rotations:

                xy_center = (im_bgr_or_gray_init.shape[1] * 0.5, -im_bgr_or_gray_init.shape[0] * 0.5)
                hw = (sheip[2], sheip[3]) 
                li_kpts0_rotated = [last_data['keypoints'][0].cpu().numpy()]
                #for iR in range(n_rot_src + 1):
                for iR in range(n_rot_src):
                    #rot_deg = -90.0 * iR
                    rot_deg = -90.0 * (iR + 1)
                    #ar_xy = last_data['keypoints'][iR].cpu().numpy()
                    kpts0 = last_data['keypoints'][iR + 1].cpu().numpy()
                    kpts0[:, 1] *= -1
                    kpts0_rotated = get_rotated_position_when_image_is_rotated_around_center(rot_deg, hw, kpts0)
                    li_kpts0_rotated.append(kpts0_rotated)
                kpts0 = np.vstack(li_kpts0_rotated)


                #last_data = unrotate_keypoints(last_data, (sheip[2], sheip[3]), n_rot_src, device, 'keypoints') 
                last_data = aggregate_kp_data(last_data)
            else:
                last_data = sort_and_trim_kp_data(last_data)
                #'''
                xy_center = (im_bgr_or_gray_init.shape[1] * 0.5, -im_bgr_or_gray_init.shape[0] * 0.5)
                hw = (sheip[2], sheip[3]) 
                li_kpts0_rotated = [last_data['keypoints'][0].cpu().numpy()]
                #for iR in range(n_rot_src + 1):
                for iR in range(n_rot_src):
                    #rot_deg = -90.0 * iR
                    rot_deg = -90.0 * (iR + 1)
                    #ar_xy = last_data['keypoints'][iR].cpu().numpy()
                    kpts0 = last_data['keypoints'][iR + 1].cpu().numpy()
                    kpts0[:, 1] *= -1
                    kpts0_rotated = get_rotated_position_when_image_is_rotated_around_center(rot_deg, hw, kpts0)
                    li_kpts0_rotated.append(kpts0_rotated)
                kpts0 = np.vstack(li_kpts0_rotated)
                #'''
                #kpts0 = np.vstack([li_kp.cpu().numpy() for li_kp in last_data['keypoints']])
            #print('kpts0_rotated.shape :', kpts0_rotated.shape);  exit()
            last_data = {k+'0': last_data[k] for k in keys}
            last_data['image0'] = frame_tensor
    last_image_id = 0
    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)
    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    while True:
        ret, im_bgr_or_gray, xy_offset, wh_new = vs.next_frame(opt.letterboxing, is_color=shall_read_color)
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1
        if 0 > opt.neg_opencv_0_sg_pos_loftr:
            globals.keypoints2, globals.descriptors2 = features.features(im_bgr_or_gray)
            kpts0, kpts1, li_valid_match, mkpts0, mkpts1 = features.matcher_kevin(image1 = im_bgr_or_gray_init,
                image2 = im_bgr_or_gray,
                keypoints1 = globals.keypoints1,
                keypoints2 = globals.keypoints2,
                descriptors1 = globals.descriptors1,
                descriptors2 = globals.descriptors2,
                matcher = opt.matcher,
                descriptor = opt.descriptor,
                ratio_thresh = m_thresh)            
            #print('AAA');   exit()
            #k_thresh = globals.detector.getThreshold()
            color = cm.jet([0.5] * len(li_valid_match))
            '''
            str_matcher = 'OpenCV_match'
            text = [
                'OpenCV_Match',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            '''
            text = [
                'OpenCV_Match',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            small_text = [
                'Keypoint Threshold: {:.3f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]


           #print('color :', color);    exit()
        elif opt.neg_opencv_0_sg_pos_loftr >= 0:

            if opt.neg_opencv_0_sg_pos_loftr > 0:
                frame_tensor = frame2tensor(im_bgr_or_gray, device)
                if opt.neg_src_0_none_pos_des < 0:
                    bb = 0
                    #   repeat
                    #print('frame_tensor.shape b4 :', frame_tensor.shape); 
                    frame_tensor = frame_tensor.repeat(n_rot_src + 1, 1, 1, 1)
                    #print('frame_tensor.shape after :', frame_tensor.shape);  exit() # [4, 1, 480, 480]
                    frame_tensor = frame_tensor.to(device); #exit()
                elif opt.neg_src_0_none_pos_des > 0:
                    #   rotate
                    aa = 0      
                    #frame_tensor = frame_tensor.to(device)
                    
                last_data = {**last_data, 'image1': frame_tensor}
                #print('ccc');   exit()
                #last_data, ts_b_ids = matcher(last_data)
                last_data = matcher(last_data)

                total_n_matches = len(last_data['mkpts0_f'])
                '''
                print('total_n_matches :', total_n_matches);    #exit()
                print('vis_range :', vis_range)
                vis_range_2 = [2, 5]
                print('vis_range_2 :', vis_range_2)
                t0 = last_data['mkpts0_f'].cpu().numpy()
                print('t0 :', t0);  
                t1 = t0[vis_range[0]:vis_range[1]]
                print('t1 :', t1);  
                t2 = t0[vis_range_2[0]:vis_range_2[1]]
                print('t2 :', t2);  
                t3 = last_data['mconf'].cpu().numpy()
                print('t3 :', t3);  
                exit()
                '''
                if total_n_matches > n_vis:
                    last_data = sort_according_2_confidence_and_trim(
                    last_data, 'mconf', ['mconf', 'mkpts0_f', 'mkpts1_f', 'b_ids'], n_vis)
                '''
                mkpts0 = last_data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
                mkpts1 = last_data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
                mconf = last_data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]
                b_ids = last_data['b_ids'].cpu().numpy()[vis_range[0]:vis_range[1]]
                '''
                mkpts0 = last_data['mkpts0_f'].cpu().numpy()
                mkpts1 = last_data['mkpts1_f'].cpu().numpy()
                mconf = last_data['mconf'].cpu().numpy()
                b_ids = last_data['b_ids'].cpu().numpy()

                i_rot_max = int(np.max(b_ids)) if b_ids.size > 0 else 0
                if i_rot_max > 0:
                    #print('im_bgr_or_gray_init.shape :', im_bgr_or_gray_init.shape)
                    #print('im_bgr_or_gray.shape :', im_bgr_or_gray.shape);  exit()
                    if n_rot_src:
                        mkpts0 = unrotate_keypoints_2(mkpts0, im_bgr_or_gray_init.shape, b_ids, -90)
                    else:
                        mkpts1 = unrotate_keypoints_2(mkpts1, im_bgr_or_gray.shape, b_ids, -90)
                kpts0 = mkpts0; kpts1 = mkpts1; 
                
                '''
                print('vis_range : ', vis_range);
                print('last_data[mkpts0_f].cpu().numpy().shape : ', last_data['mkpts0_f'].cpu().numpy().shape)
                print('last_data[mkpts1_f].cpu().numpy().shape : ', last_data['mkpts1_f'].cpu().numpy().shape)
                print('last_data[mconf].cpu().numpy().shape : ', last_data['mconf'].cpu().numpy().shape)
                exit();
                '''


                timer.update('forward')

                # Normalize confidence.
                if len(mconf) > 0:
                    conf_vis_min = 0.
                    conf_min = mconf.min()
                    conf_max = mconf.max()
                    mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)
                    alpha = 0
                    color = cm.jet(mconf, alpha=alpha)
                else:
                    conf_min = -1
                    conf_max = -1
                    color = cm.jet([0.5] * 0)
                text = [
                    f'LoFTR',
                    '# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),
                ]
                small_text = [
                    f'Showing matches from {vis_range[0]}:{vis_range[1]}',
                    f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
                    'Image Pair: {:06}:{:06}'.format(stem0, stem1),
                ]
         

            else:
                #print('frame_tensor.shape b4 :', frame_tensor.shape);   exit()
       
                li_frame_tensor = [frame2tensor(im_bgr_or_gray, device)]
                for iR in range(n_rot_des):
                    #if opt.neg_src_0_none_pos_des > 0:
                    li_frame_tensor.append(torch.rot90(li_frame_tensor[iR], iR + 1, [2, 3]))
                
                sheip = li_frame_tensor[0].shape 
                hw = (sheip[2], sheip[3]) 
                frame_tensor = torch.Tensor(len(li_frame_tensor), sheip[1], sheip[2], sheip[3]) 
                torch.cat(li_frame_tensor, out = frame_tensor)
                #print('frame_tensor.shape :', frame_tensor.shape);  #exit()
                frame_tensor = frame_tensor.to(device)
                #print('last_data[image0].shape :', last_data['image0'].shape);     
                #print('frame_tensor.shape :', frame_tensor.shape);    exit()     

                pred = matching({**last_data, 'image1': frame_tensor})

                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                
                if opt.aggregate_rotations:
                    
                    #kpts0 = last_data['keypoints0'][0].cpu().numpy()
                    kpts1 = pred['keypoints1'][0].cpu().numpy()
                    matches = pred['matches0'][0].cpu().numpy()
                    confidence = pred['matching_scores0'][0].cpu().numpy()
                    '''
                    print('kpts0.shape :', kpts0.shape)             #   (531, 2)    (# of kp in initial image, 2) 
                    print('kpts1.shape :', kpts1.shape)             #   (721, 2)    (# of kp in current image, 2) 
                    print('matches.shape :', matches.shape)         #   (531,)      # of kp in initial image 
                    print('confidence.shape :', confidence.shape);  #   (531,)      # of kp in initial image
                    '''
                    timer.update('forward')
                    valid = matches > -1
                    #print('kpts0.shape :', kpts0.shape);    #exit()
                    #mkpts0 = kpts0[valid]
                    mkpts0 = kpts0[valid]
                    mkpts1 = kpts1[matches[valid]]
                    color = cm.jet(confidence[valid])
                    #li_mkpts0.append(mkpts0);   li_mkpts1.append(mkpts1);   li_color.append(color)
                    #print('type(color) :', type(color));    exit()
                    #print('mkpts0.shape :', mkpts0.shape);    exit()
                    #print('valid.shape :', valid.shape);                            #   (531,)
                    #print('confidence.shape :', confidence.shape);                  #   (531,)
                    #print('type(valid) :', type(valid));                            #   numpy.ndarray
                    #print('type(confidence) :', type(confidence));                  #   numpy.ndarray
                    #print('confidence[valid].shape :', confidence[valid].shape);    #   (14,) 
                    #print('confidence[valid] :', confidence[valid]);                #   [0.26 0.22 0.31 0.36 0.20 0.22 0.23 0.27 0.30 0.27 0.23 0.22]     
                    #print('color :', color);                                        #   [[0. 0.56 1. 1.] [0. 0.37 1. 1.] [0. 0.75 1. 1.] [0.06 0.97 0.90 1.] [0. 0.3 1. 1.] [0. 0.4 1. 1.] [0. 0.4 1. 1.] [0. 0.59 1. 1] [0. 0.70 1. 1.] [0. 0.61 1. 1.] [0 0.42 1. 1.] [0. 0.39 1. 1.] [0. 0.47 1. 1.] [0. 0.34 1. 1.]]     
                    #exit()



                else:
                    li_kpts1_rotated = [];  
                    li_mkpts0 = [];  
                    li_mkpts1 = [];  
                    li_color = [];  
                    #li_kpts1_rotated = [pred['keypoints1'][0].cpu().numpy()]
                    for iR in range(n_rot + 1):
                        if 0 == iR:
                            kpts0_rotated = li_kpts0_rotated[0]
                            #kpts0_rotated = last_data['keypoints0'][0].cpu().numpy()
                            kpts1_rotated = pred['keypoints1'][0].cpu().numpy()
                            #li_kpts1_rotated.append(kpts1_rotated)
                        else:
                            if 0 == n_rot_des:
                                kpts0_rotated = li_kpts0_rotated[iR]
                                #kpts0_rotated = last_data['keypoints0'][iR].cpu().numpy()
                                kpts1_rotated = pred['keypoints1'][0].cpu().numpy()
                            else:     
                                kpts0_rotated = li_kpts0_rotated[0]
                                #kpts0_rotated = last_data['keypoints0'][0].cpu().numpy()
                                kpts1 = pred['keypoints1'][iR].cpu().numpy()

                                rot_deg = -90.0 * iR
                                kpts1[:, 1] *= -1
                                kpts1_rotated = get_rotated_position_when_image_is_rotated_around_center(rot_deg, hw, kpts1)
                        li_kpts1_rotated.append(kpts1_rotated)
         
                        #kpts0 = last_data['keypoints0'][i_src].cpu().numpy()
                        #kpts1 = pred['keypoints1'][i_des].cpu().numpy()
                        matches = pred['matches0'][iR].cpu().numpy()
                        confidence = pred['matching_scores0'][iR].cpu().numpy()
                        '''
                        print('kpts0.shape :', kpts0.shape)             #   (531, 2)    (# of kp in initial image, 2) 
                        print('kpts1.shape :', kpts1.shape)             #   (721, 2)    (# of kp in current image, 2) 
                        print('matches.shape :', matches.shape)         #   (531,)      # of kp in initial image 
                        print('confidence.shape :', confidence.shape);  #   (531,)      # of kp in initial image
                        '''
                        timer.update('forward')

                        valid = matches > -1
                        #print('kpts0.shape :', kpts0.shape);    #exit()
                        #mkpts0 = kpts0[valid]
                        mkpts0 = kpts0_rotated[valid]
                        mkpts1 = kpts1_rotated[matches[valid]]
                        color = cm.jet(confidence[valid])
                        li_mkpts0.append(mkpts0);   li_mkpts1.append(mkpts1);   li_color.append(color)
                        #print('type(color) :', type(color));    exit()
                        #print('mkpts0.shape :', mkpts0.shape);    exit()
                        #print('valid.shape :', valid.shape);                            #   (531,)
                        #print('confidence.shape :', confidence.shape);                  #   (531,)
                        #print('type(valid) :', type(valid));                            #   numpy.ndarray
                        #print('type(confidence) :', type(confidence));                  #   numpy.ndarray
                        #print('confidence[valid].shape :', confidence[valid].shape);    #   (14,) 
                        #print('confidence[valid] :', confidence[valid]);                #   [0.26 0.22 0.31 0.36 0.20 0.22 0.23 0.27 0.30 0.27 0.23 0.22]     
                        #print('color :', color);                                        #   [[0. 0.56 1. 1.] [0. 0.37 1. 1.] [0. 0.75 1. 1.] [0.06 0.97 0.90 1.] [0. 0.3 1. 1.] [0. 0.4 1. 1.] [0. 0.4 1. 1.] [0. 0.59 1. 1] [0. 0.70 1. 1.] [0. 0.61 1. 1.] [0 0.42 1. 1.] [0. 0.39 1. 1.] [0. 0.47 1. 1.] [0. 0.34 1. 1.]]     
                        #exit()
                    kpts1 = np.vstack(li_kpts1_rotated)
                    mkpts0 = np.vstack(li_mkpts0)
                    mkpts1 = np.vstack(li_mkpts1)
                    color = np.vstack(li_color)
                text = [
                        'SuperGlue',
                        #str_matcher,
                        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                        'Matches: {}'.format(len(mkpts0))
                ]
                small_text = [
                    'Keypoint Threshold: {:.3f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {:06}:{:06}'.format(stem0, stem1),
                ]
         
        out = make_matching_plot_fast(
            #last_frame, frame, 
            im_bgr_or_gray_init, im_bgr_or_gray, 
            kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text, show_homography=opt.show_homography, xy_offset_init=xy_offset_init, xy_offset=xy_offset, wh_new_init=wh_new_init, wh_new=wh_new)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        #print('opt.output_dir :', opt.output_dir);  print("");  exit()
        timer.update('viz')
        timer.print()
        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            #exit()
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()

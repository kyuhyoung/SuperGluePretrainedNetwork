import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        #print('data[image0].size() :', data['image0'].size());    #exit()
        #print('data[image1].size() :', data['image1'].size());    exit()
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            #print('data[image1][0, 0, 0] :', data['image1'][0, 0, 0]);  print('data[image1][-1, -1, -1] :', data['image1'][-1, -1, -1]);    exit()
            #print('data[image0].size() :', data['image0'].size());    print('data[image1].size() :', data['image1'].size());    
            #print('feats_c.size() :', feats_c.size());    print('feats_f.size() :', feats_f.size());    #exit()
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
            #print('feat_c1[0, 0, 0] b4 :', feat_c1[0, 0, 0]);  print('feat_c1[-1, -1, -1] b4 :', feat_c1[-1, -1, -1]);    exit()
            #print('feat_c0.size() :', feat_c0.size());    print('feat_c1.size() :', feat_c1.size());#   exit()
            #print('feat_f0.size() :', feat_f0.size());    print('feat_f1.size() :', feat_f1.size());    exit()
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        #print('feat_c0[0, 0, 0] :', feat_c0[0, 0, 0]);  print('feat_c0[-1, -1, -1] :', feat_c0[-1, -1, -1]);    exit()

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        #print('feat_c0.shape 1 :', feat_c0.shape);    print('feat_c1.shape 1 :', feat_c1.shape); 
        #print('feat_c0[0, 0, 0] b4 :', feat_c0[0, 0, 0]);  print('feat_c0[-1, -1, -1] b4 :', feat_c0[-1, -1, -1]);    #exit()
        #print('feat_c1[0, 0, 0] b4 :', feat_c1[0, 0, 0]);  print('feat_c1[-1, -1, -1] b4 :', feat_c1[-1, -1, -1]);    #exit()
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        #print('feat_c0[0, 0, 0] after :', feat_c0[0, 0, 0]);  print('feat_c0[-1, -1, -1] after :', feat_c0[-1, -1, -1]);    exit()
        #print('feat_c0.shape 2 :', feat_c0.shape);    print('feat_c1.shape 2 :', feat_c1.shape);    exit()
        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        #print('feat_c0.shape 3 :', feat_c0.shape);    print('feat_c1.shape 3 :', feat_c1.shape);    #exit()
        #print('feat_c0[0, 0, 0] :', feat_c0[0, 0, 0]);  print('feat_c0[-1, -1, -1] :', feat_c0[-1, -1, -1]);    #exit()
        # 4. fine-level refinement
        #feat_f0_unfold, feat_f1_unfold, b_ids = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        #print('feat_f0_unfold.shape 1 :', feat_f0_unfold.shape);    print('feat_f1_unfold.shape 1 :', feat_f1_unfold.shape);    #exit()
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
            #print('feat_f0_unfold.shape 2 :', feat_f0_unfold.shape);    print('feat_f1_unfold.shape 2 :', feat_f1_unfold.shape);    #exit()

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        #print('data[mkpts0_f] :' , data['mkpts0_f'])
        #print('data[mkpts1_f] :' , data['mkpts1_f']);    #exit()
        #'''
        #if not(torch.all(data['b_ids'].eq(data['m_bids']))):   
            #print('b_ids is NOT equal to m_bids')
            #print('data[b_ids] :', data['b_ids']);   
            #print('data[m_bids] :', data['m_bids']);   
            #print('data[mconf] :', data['mconf'])
            #exit()
        #'''     
        return data#, b_ids

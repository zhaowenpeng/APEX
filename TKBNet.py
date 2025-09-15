########################################################################################################################
####################The full version of the code will be presented upon the article's acceptance.#####################
############################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from ikan import ChebyKANLinear, GroupKANLinear
from timm.layers import DropPath, trunc_normal_


class TKBNet(nn.Module):

    def __init__(self, num_classes=2):
        super(TKBNet, self).__init__()
        self.num_classes = num_classes
                

        self.backbone = ResNet50()

        self.msfb_decoder = GlobalContextPerceptionBranch(
            num_class=2,
            feature_list=[32, 64, 128, 256, 512, 1024, 2048],
            drop_out=0.0
        )
        

        self.hegb_decoder = HybridEdgeGuidanceDecoder(
            ppm_in_feat=2048,
            high_level_ch=32,
            low_level_ch=256
        )
        

        self.kesb_decoder = KANEnhancedSemanticDecoder()
        

        self.atf_fusion = TriBranchAdaptiveFusion(in_channels=64)
        
    
        self.final_classifier = EnhancedClassificationHead(in_feature=64, out_feature=num_classes, drop_out=0.3)

    def forward(self, x):

        x1, x2, x3, x4 = self.backbone(x)
        

        msfb_features, _ = self.msfb_decoder(x1, x2, x3, x4)
        
        print(x1.shape,x2.shape,x3.shape,x4.shape)

        hegb_features, _ = self.hegb_decoder(x4, x1)

        kesb_features = self.kesb_decoder(x2, x3, x4)
        
 
        fused_features = self.atf_fusion(msfb_features, kesb_features, hegb_features)
        

        output = self.final_classifier(fused_features)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        
        return output

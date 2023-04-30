# A Dual Attention Network for Automatic Metallic Corrosion Detection in Natural Environment

## Abstract

Industrial and residential facilities worldwide are exposed to significant costs and hazards due to metallic corrosion in urban systems. It is necessary to develop a method to automatically detect metallic corrosion precisely. However, existing metallic corrosion detection methods can not detect either from the realistic natural images or all kinds of metallic erosion with a high accuracy rate. In this paper, a Dual Attention Metallic Corrosion Network (DAMCN) is proposed to automatically detect the metallic corrosion regions with various sizes and shapes from digital images, which are collected in natural environment by unmanned aerial vehicles. By embedding the dual attention modules (i.e. the CBAM blocks) into different locations of the backbone network (i.e. Mask-RCNN), important metallic corrosion features are enhanced either in the feature encoding (i.e. DAMCN-V1) or in the multi-scale feature extraction (i.e. DAMCN-V2). Extensive experiments on three benchmark datasets validate the effectiveness of the proposed method on metallic corrosion detection with varying region size and shape. Comparison results among our DAMCN-V1 and DAMCN-V2 to other existing related methods also demonstrate the superiority of our DAMCN-V1, which achieves the best result while its counter-part (DAMCN-V2) comes to the second.

## Structure:

CBAM-keras: Sample code for CBAM module.

DAMCN-V1: Architecture of DAMCN-V1 model.

DAMCN-V2: Architecture of DAMCN-V2 model.

NEA-Dataset: NEA dataset and its annotations.

samples: Running scripts.

# SEFCA-Net: a semantic–edge–frequency cross-attention network for diabetic retinopathy lesion segmentation

- **Authors:** Jin Xie, Jingtao Wang, Duo Xu, Yuzhong Peng, Kunlong Wu, Fangning Gu, Xue Zhang, Zhizhong Zhang, Xinjian Chen, Jianping Liao
- **Journal:** Journal of Biomedical Optics
- **Year:** 2026
- **PMID:** 42852419
- **PMCID:** PMC13649286
- **DOI:** 10.1117/1.JBO.31.10.106002
- **Keywords:** diabetic retinopathy; lesion segmentation; complementary feature fusion; semantic-guided window-based cross-attention

## Abstract

Significance: Diabetic retinopathy (DR) lesion segmentation provides pixel-level lesion localization in fundus images. However, such lesions are often tiny, sparsely distributed, low-contrast, morphologically diverse, and visually similar to vessels or surrounding retinal structures, making accurate localization difficult. Many existing methods emphasize multi-scale spatial representation learning and handle boundary ambiguity and subtle texture differences mainly through encoder–decoder fusion. We therefore examine boundary and frequency representations as explicit, complementary inputs to semantic decoding.

Aim: We aim to develop SEFCA-Net, a semantic–edge–frequency cross-attention network for diabetic retinopathy lesion segmentation. The main objective is to examine whether semantic, edge, and frequency-domain representations can be fused under semantic guidance to improve pixel-level lesion localization without compromising the semantic decoding stream. The evaluation focuses on four lesion categories, including hard exudates, hemorrhages, soft exudates, and microaneurysms, using public benchmarks and a DDR-to-IDRiD cross-dataset setting.

Approach: SEFCA-Net uses HRNet-W48 as the semantic backbone to extract multi-scale semantic features and explicitly constructs edge and frequency-domain representations from input fundus images. The learnable multi-scale edge branch extracts boundary-sensitive features, while the all-channel multi-scale discrete cosine transform (DCT) frequency-domain branch generates texture-sensitive responses to low-contrast lesions. The semantic–edge–frequency cross-attention decoder uses the semantic representation as the primary stream and injects edge and frequency-domain features through semantic-guided window-based cross-attention. Performance is evaluated using the area under the precision–recall curve (AUPR), Dice, and Intersection-over-Union (IoU).

Results: On DDR, SEFCA-Net achieves 47.85% mean AUPR (mAUPR), 46.76% mean Dice, and 31.22% mean IoU. On IDRiD, it reaches 69.74% mAUPR; when trained on DDR and tested on IDRiD, it obtains 59.83% mAUPR. Ablation studies show that either auxiliary branch improves the semantic-only baseline and that the complete configuration gives the highest mAUPR. Multi-scale Sobel modeling and DCT outperform the tested single-scale and wavelet alternatives, respectively.

Conclusions: SEFCA-Net shows that semantic features can remain the main decoding stream while boundary-sensitive features and texture-sensitive responses are integrated in a controlled manner. Under the evaluated protocols, the model shows competitive in-domain performance and records the highest DDR-to-IDRiD mAUPR among the reproduced methods.

## Full text

## Introduction

Diabetic retinopathy (DR) remains one of the major causes of visual impairment worldwide.1–3 Color fundus photography is commonly used to assess retinal findings in DR, and deep learning systems have shown strong performance in image-level screening and referral decisions.4–9 DR lesion segmentation aims to localize lesions and quantify typical lesions, including hard exudates, hemorrhages, soft exudates, and microaneurysms. However, DR lesions are often tiny, sparsely distributed, low-contrast, and morphologically diverse, and some lesion regions are visually similar to vessels or surrounding retinal structures. Therefore, DR lesion segmentation remains a highly challenging task.

Current DR lesion segmentation methods mainly pursue stronger multi-scale spatial representation learning. L-Seg10 uses weighted multi-level feature fusion, whereas RTNet11 models intra-class lesion dependencies and inter-class lesion–vessel relationships. SAA,12 PMCNet,13 M2MRF,14 and HRDecoder15 further improve scale reweighting, cross-level consistency, feature reassembly, or high-resolution decoding. Although these methods are effective, and their central operations act on spatial feature maps; the separately parameterized edge and frequency branches examined here are not part of these designs.

High-resolution representations help preserve fine structures, while frequency-aware transforms can capture subtle texture changes and local details.16,17 However, separately derived boundary and frequency cues are not inherently reliable: gradients can respond to vessels, and frequency components may emphasize background variation. The resulting design question is how to integrate such features under semantic guidance.

This paper proposes SEFCA-Net, a semantic–edge–frequency cross-attention network designed for diabetic retinopathy lesion segmentation. The network uses HRNet as the backbone to generate multi-scale semantic features, integrates a multi-scale edge modeling branch to extract boundary-sensitive features, and introduces an all-channel multi-scale discrete cosine transform (DCT) frequency modeling branch to generate texture-sensitive responses. Semantic features are then used as the main decoding stream, while boundary-sensitive features and texture-sensitive responses are injected through semantic-guided window-based cross-attention. This asymmetric design aims to refine lesion localization while limiting interference from vessel edges and background frequency components.

The contributions of this work are mainly threefold. First, we formulate DR lesion segmentation as the fusion of semantic, edge, and frequency representations under semantic guidance and implement this framework in SEFCA-Net. Second, we design a learnable multi-scale edge branch, an all-channel multi-scale frequency branch, and a semantic-guided window-based cross-attention decoder to selectively integrate auxiliary information. Third, we evaluate the model through in-domain comparisons, a DDR-to-IDRiD transfer experiment, an efficiency analysis, and controlled ablations of the main design choices.

## Related Work

### Fundus Lesion Segmentation

Retinal lesion segmentation has evolved from methods targeting individual lesion types to networks that jointly predict several DR lesion categories. Early studies usually focused on a single lesion category or lesion group, such as red-lesion detection18 and exudate detection.19 A later dual-branch network with dual-sampling modulated Dice loss improved hard-exudate localization.20 Representative multi-lesion methods include L-Seg,10 which adopts weighted multi-level feature fusion; RTNet,11 which models intra-class lesion dependencies and inter-class lesion-vessel relationships; and SAA,12 which dynamically adjusts multi-scale feature weights through scale-aware attention. PMCNet,13 M2MRF,14 and HRDecoder15 further improve performance through progressive multi-scale consistency, many-to-many feature reassembly, and high-resolution decoding. MSAG combines multi-scale inputs with hierarchical spatial attention on DDR and IDRiD.21 Attention-DeepLab places convolutional block attention modules after the atrous spatial pyramid pooling module and in the low-level feature path of the DeepLabV3+ decoder to refine semantic responses and spatial details.22 SEFCA-Net instead explicitly constructs Sobel-derived edge and DCT-derived frequency representations and integrates them into an HRNet semantic stream through semantic-guided cross-attention. These studies demonstrate the value of improved spatial aggregation.

### Multi-Scale and High-Resolution Representation Learning

Modern semantic segmentation performance is closely related to the quality of multi-scale and high-resolution representations, a trend also emphasized in broad medical image analysis surveys.23,24 FCN25 and U-Net26 establish the encoder–decoder paradigm for dense prediction. V-Net,27 U-Net++,28 and ResUNet-a29 further explore richer skip fusion and multi-scale refinement in medical or high-resolution settings. FPN30 and DeepLabV3+31 improve multi-scale context aggregation through pyramid fusion and atrous spatial pyramid pooling, respectively. HRNet16 maintains parallel streams at several resolutions and repeatedly exchanges information among them, preserving spatial detail while accumulating semantic context. Transformer-based models, starting from ViT32 and later developing into hierarchical variants such as Swin Transformer33 and SegFormer,34 continue this direction with stronger long-range context modeling. In fundus image analysis, M2MRF14 and HRDecoder15 confirm that refined multi-scale features and high-resolution decoding are important for tiny lesion segmentation. For SEFCA-Net, the remaining design question is how separately derived boundary and frequency features should update this high-resolution semantic stream.

### Boundary and Frequency-Domain Auxiliary Modeling

Beyond semantic feature extraction, auxiliary structural and spectral information has also been explored in dense prediction. HED35 shows that explicit boundary-aware learning can improve pixel-level prediction, while CBAM36 demonstrates that lightweight spatial and channel attention can refine feature extraction by strengthening informative responses. In frequency-domain analysis, Wave-ViT,17 fast Fourier convolution,37 FsaNet,38 and LCSeg-Net39 show that spectral transforms or frequency-aware operators can preserve detailed features and improve segmentation, especially for low-contrast structures. These studies indicate that structural or spectral information can enhance the expressiveness of semantic features. In this paper, the phrase “complementary representations” refers specifically to the outputs of the HRNet semantic branch, the Sobel-derived boundary branch, and the DCT-derived frequency branch. This operational definition supports branch-level ablation and does not imply that other architectures cannot learn related cues implicitly.

## Method

### Overall Architecture

Figure 1 shows the overall architecture of SEFCA-Net. Given an input fundus image I∈R3×H×W, the model first uses an HRNet-W4816 encoder to extract semantic features at four different resolutions, denoted as F4, F8, F16, and F32, with strides of 4, 8, 16, and 32, respectively. On the decoder side, these features are transformed by a 1×1 projection followed by spatial attention refinement into C4, C8, C16, and C32. Among them, C4 preserves higher-resolution spatial details, while C8, C16, and C32 are aligned to the same spatial resolution and fused to form the low-resolution semantic body.

*[Figure] Architecture of SEFCA-Net. (a) Overall network. (b) Semantic decoder. (c) Learnable multi-scale edge branch. (d) All-channel multi-scale DCT branch. (e) Semantic-guided window-based cross-attention.*

In addition to the semantic backbone, SEFCA-Net explicitly constructs two auxiliary representations, namely, edge and frequency-domain representations. The edge branch extracts multi-scale gradient features using 3×3, 5×5, and 7×7 Sobel kernels to supplement lesion boundary information. The frequency branch performs multi-scale DCT modeling on each color channel of the input image to obtain frequency information for low-contrast lesions. The semantic–edge–frequency cross-attention decoder then uses the semantic representation as the main stream, introduces edge and frequency features into the decoding process through semantic-guided window-based cross-attention, and outputs prediction logits for the four lesion classes. The implementation details of each module of SEFCA-Net are described in Secs. 3.2–3.5.

### Learnable Multi-Scale Edge Modeling

A single fixed edge operator is difficult to adapt to the complex boundary patterns of fundus lesions, and the gradient scale suitable for describing these boundaries also varies. Therefore, we design a learnable multi-scale edge modeling branch, which introduces scale-adaptive weighting and lightweight convolutional enhancement on top of multi-scale edge information.

Specifically, as shown in Fig. 1(c), the branch applies Sobel kernels of sizes 3×3, 5×5, and 7×7 to obtain three groups of edge features, denoted as M(3), M(5), and M(7), respectively. Then, the softmax-normalized scale coefficients α3, α5, and α7 are used to adaptively weight the three scales, and the aggregated edge feature is fed into a lightweight fusion subnetwork. More specifically, a 1×1 convolution with batch normalization and ReLU adjusts the channel representation, a 3×3 depthwise separable convolution followed by a 3×3 convolution with batch normalization and ReLU enhances local boundary structures, and a final 1×1 convolution with batch normalization projects the feature to the decoder dimension. The final edge feature is written as E=Align(Proje(Enhe(Cat[α3M(3),α5M(5),α7M(7)])))∈RCe×Hd×Wd,(1)where Enhe(·) denotes the edge fusion subnetwork in Fig. 1(c), which performs channel mixing and local structural enhancement, and Proje(·) denotes the subsequent projection operation, which maps the fused feature to Ce=32 channels required by cross-attention. The resulting E preserves fine-grained edges, medium-scale contours, and coarse boundaries at the same time and can serve as an explicit boundary prior for the decoder.

### All-Channel Multi-Scale Frequency-Domain Modeling

Frequency-domain modeling is used to supplement low-contrast lesion appearance differences that are difficult to distinguish from spatial features alone. Microaneurysms often appear as small local bright-dark changes, whereas exudates and hemorrhages are accompanied by broader intensity changes and texture differences. Different color channels also highlight different retinal structures. Based on this observation, we design an all-channel multi-scale frequency-domain modeling branch and use multi-scale DCT, SE channel recalibration, and frequency-band weighting to generate frequency-domain auxiliary features.

Specifically, as shown in Fig. 1(d), for each color channel c∈{R,G,B}, we use DCT convolution blocks with non-overlapping block sizes of 8 and 16 to encode two frequency scales. Each block produces four absolute coefficient maps. The convolution kernels are initialized with low- and mid-frequency DCT basis functions and are further optimized during training. The 16×16 responses are bilinearly upsampled to the 8×8 response resolution. After concatenating the frequency-domain features from all color channels at the two scales, we obtain the all-channel frequency-domain representation Dall=Cat[{Ψ8(c)(I(c)),Up(Ψ16(c)(I(c)))}c∈{R,G,B}].(2)

Finally, we first use a squeeze-and-excitation (SE) module to recalibrate the channels of Dall then apply the frequency band weights β=softmax(wfreq) to the 24 concatenated responses. A 3×3 convolution with batch normalization and ReLU fuses the weighted responses, and a 1×1 convolution with batch normalization produces the final feature D=Align(Projd(Fused(β⊙SE(Dall))))∈RCd×Hd×Wd,(3)where Fused(·) and Projd(·) denote the 3×3 fusion block and 1×1 projection in Fig. 1(d), respectively. The projection sets the output channel dimension, while the decoder aligns the spatial resolution before cross-attention. The final D is a frequency-domain representation aligned with the decoder.

### Semantic–Edge–Frequency Cross-Attention Decoder

The decoder head is responsible for introducing edge and frequency-domain information while preserving the semantic main stream. As shown in Figs. 1(b) and 1(e), the decoding process can be divided into two steps. First, a base semantic representation S is constructed from multi-scale semantic features. Then, S is used as the shared query feature to perform semantic-guided window-based cross-attention with the edge feature E and the frequency-domain feature D. This design keeps edge and frequency-domain information in a supplementary role and prevents them from replacing the dominant semantic branch.

Specifically, we first use the lightweight compression module Refiner(·), implemented as a 1×1 projection followed by spatial attention refinement, to process the semantic features at four scales and obtain C4, C8, C16, and C32. The low-resolution features are fused into the semantic body, while the highest-resolution feature is kept as detail guidance. This process is written as Cr=Refiner(Fr),r∈{4,8,16,32},SB=Up(Conv3×3(Cat[C8,Up(C16),Up(C32)])),SD=Conv1×1(C4),S=SB+SD.(4)

Here, SB and SD are the semantic body and detail guidance in Fig. 1(b), respectively, and S is the base semantic representation fed into the SEFCA module.

Then, we project the edge and frequency-domain features to the same dimension as S, partition them into non-overlapping local windows, and update S through two semantic-guided window-based cross-attention operations. With the default setting of w=16 and 4 attention heads, the process is written as YE=WinMHA(Q=S,K=V=ProjE(E)),YD=WinMHA(Q=S,K=V=ProjD(D)),Saux=S+γESE(YE)+γDSE(YD),Plogit=Head(S+γFConvBN3×3(Saux)),(5)where WinMHA(·) denotes the local window multi-head cross-attention with relative position bias, ProjE(·) and ProjD(·) are the input projection layers that map the auxiliary features to the semantic-query channels, and Head(·) is the output head. YE and YD denote the edge-guided and frequency-guided semantic updates, respectively, and γE and γD are the LayerScale coefficients initialized to 0. After SE recalibration, the two updates are added to the base representation to form Saux. A 3×3 convolution with batch normalization transforms this feature, and a third zero-initialized LayerScale coefficient γF controls its residual addition to the original semantic representation before Head(·). The window mechanism reduces global quadratic complexity to local window complexity and ensures that auxiliary information participates in updates only within semantically related local regions. During training and inference, the logits are finally resized to the input resolution and passed through a sigmoid function to output prediction probabilities.

### Loss Function

The training objective uses a class-averaged binary Dice loss to address the severe class imbalance in fundus lesion segmentation. We apply sigmoid independently to each lesion channel and define the overall loss as L=1K∑k=1K[1−2∑ipikgik+ε∑i(pik)2+∑i(gik)2+ε],(6)where K=4 denotes the number of lesion categories; pik and gik denote the predicted probability and binary annotation of class k at pixel i, respectively; and ϵ=10−5 is the smoothing term. The background class is not included in the loss calculation.

## Experiments

### Experimental Setup

#### Datasets

We evaluate SEFCA-Net on two public DR lesion segmentation benchmarks, DDR and IDRiD. IDRiD has been released both as a public dataset and as a lesion segmentation challenge benchmark,40,41 while DDR is widely used in several recent DR lesion segmentation studies.14,15,42 According to the data split used in the current experiments, DDR contains 383 training images, 148 validation images, and 225 testing images. IDRiD follows the official split and contains 54 training images and 27 testing images. Both datasets provide pixel-level annotations for four lesion categories: hard exudates (EX), hemorrhages (HE), soft exudates (SE), and microaneurysms (MA).

#### Evaluation metrics

We use area under the precision–recall curve (AUPR), Dice, and Intersection-over-Union (IoU) as evaluation metrics and report the class-averaged results over the four lesion categories, namely, area under the precision–recall curve (mAUPR), mean Dice (mDice), and mean Intersection-over-Union (mIoU). Here, mDice is calculated from the evaluator’s mF1.11,13,15 Following the recommendation of M2MRF,14 we use AUPR as the primary metric because DR lesion segmentation has severe class imbalance and contains many tiny lesions. Dice and IoU are used as complementary measures of region overlap quality. Precision and sensitivity are summarized across thresholds by AUPR, while Dice corresponds to the binary F1 score at a fixed threshold. Specificity and area under the receiver operating characteristic curve (AUROC) are not additionally reported because they can be dominated by the large number of background pixels; lesion-level sensitivity and free-response receiver operating characteristic (FROC) require a separate connected component matching protocol beyond the pixel-level evaluation used here.

#### Network settings

SEFCA-Net is implemented based on the MMSegmentation framework,43 and the semantic backbone is HRNet-W48.16 Unless otherwise specified, the edge branch outputs 32 channels, the frequency-domain branch outputs 8 channels, the decoder hidden dimension is 128, and the SEFCA module uses a default window size of 16 and 4 attention heads.

#### Training details

All experiments follow a unified implementation and training protocol. DDR images are resized to 1024×1024 and trained for 40k iterations, while IDRiD images are resized to 1440×960 and trained for 20k iterations. The mini-batch size is 4 in all experiments. We use stochastic gradient descent (SGD) as the optimizer, with an initial learning rate of 0.01 and a weight decay of 5×10−4, together with warm-up and a poly learning rate decay schedule. During training, each image is randomly rescaled by a factor sampled from [0.5, 2.0], cropped to the target input size, horizontally flipped with a probability of 0.5, and randomly rotated within [−45 deg, 45 deg]. These geometric transformations form the base augmentation pipeline used in all main experiments. Validation and test images undergo deterministic resizing without random augmentation. Since IDRiD does not provide an official validation set, we do not perform additional hyperparameter tuning on its test set. Instead, we use the same training and tuning strategy as DDR and directly report test results under fixed settings. All locally reproduced comparison results and the ablations in Tables 5–10 are obtained from three independent runs, and the best result of each model or experimental setting is reported. Training experiments are mainly conducted on four RTX 4090D graphics processing units (GPUs). For models with higher memory consumption, training uses two A800 GPUs. Frames per second (FPS), parameter count, and giga floating-point operations (GFLOPs) in the efficiency evaluation are all measured on an RTX 5060 Ti. FPS is measured with a batch size of 1, 5 warm-up batches, and 200 timed batches per repeat, with 5 repeats.

### Comparison with State-of-the-Art Methods

This section evaluates SEFCA-Net on DDR, IDRiD, and the cross-dataset setting from DDR to IDRiD. We reproduce DeepLabV3+,31 HRNetV2-W48,16 PSPNet,44 ConvNeXt-B,45 Swin-B,33 Twins-SVT-B,46 SegFormer-B5,34 and M2MRF-A14 under unified training settings and mark them with ‡. Published results for L-Seg,10 HED,35 DNL,47 SAA,12 PMCNet,13 RTNet,11 and MSAG21 are also included where available.

#### Results on the DDR dataset

Table 1 presents the quantitative comparison on DDR. Among the locally reproduced baselines, SEFCA-Net achieves the highest 47.85% mAUPR, 46.76% mDice, and 31.22% mIoU. Compared with the second-best method, Twins-SVT-B,46 it improves the three mean metrics by 0.19, 0.35, and 0.38 percentage points. At the category level, SEFCA-Net achieves the best EX and SE AUPR and the second-best MA AUPR, Dice, and IoU among the reproduced methods.

*[Table] Quantitative comparison on the DDR dataset (%). ‡ denotes the methods evaluated under the unified settings; bold and italic data indicate the best and second-best results within this group, respectively.*

#### Results on the IDRiD dataset

Table 2 reports the results on IDRiD. SEFCA-Net obtains the highest mAUPR of 69.74% among the reproduced methods, slightly higher than SegFormer-B534 at 69.73% and Swin-B33 at 69.49%. It also achieves the best EX Dice, EX IoU, and MA IoU among the reproduced methods.

*[Table] Quantitative comparison on the IDRiD dataset (%). ‡ denotes the methods evaluated under the unified settings; bold and italic data indicate the best and second-best results within this group, respectively.*

#### Generalization study

The cross-dataset experiment adopts the setting of training on DDR and testing on IDRiD, and the results are shown in Table 3. SEFCA-Net achieves the highest mAUPR of 59.83% among the reproduced methods and obtains the best AUPR of 79.44% for the EX category. Compared with the strongest reproduced baseline, Swin-B,33 mAUPR increases by 0.16 percentage points. Because DDR and IDRiD differ in lesion prevalence and image characteristics, the absolute mAUPR values in Tables 1–3 should not be compared across protocols.

*[Table] Cross-dataset generalization results from DDR to IDRiD (AUPR, %). ‡ denotes the methods evaluated under the unified transfer setting; bold and italic data indicate the best and second-best results within this group, respectively.*

### Efficiency Analysis

Table 4 compares the accuracy and computational cost. Except for HRDecoder, which uses 2048×2048 inputs, all other methods are evaluated at 1024×1024.15 SEFCA-Net achieves 47.85% mAUPR and 12.33 FPS. Compared with HRNetV2-W48, SEFCA-Net has a similar number of parameters and slightly lower GFLOPs while improving both mAUPR and FPS.

*[Table] Efficiency comparison. HRDecoder uses 2048×2048 inputs; all other methods use 1024×1024. Bold and italic data indicate the best and second-best results, respectively.*

### Ablation Studies

The ablation experiments examine four aspects: the auxiliary branches and fusion strategy, the key settings of the edge and frequency-domain branches, the branch hyperparameters, and training time contrast augmentation. Since the task has clear class imbalance, the following analysis mainly focuses on mAUPR, with mDice and mIoU used as supporting metrics.

#### Effects of auxiliary branches and fusion strategy

The ablation results of the auxiliary branches show that both edge and frequency-domain information can complement semantic decoding. When only the edge branch is added, mIoU and mDice are the best. When the full model introduces both the edge and frequency-domain branches, it achieves the highest mAUPR, indicating that the two auxiliary representations are complementary in terms of prediction reliability. The frequency-only setting also improves all three mean metrics over the semantic-only configuration (Table 5).

*[Table] Ablation results of auxiliary branches. The full model achieves the highest mAUPR, while the edge-only branch performs best on mIoU and mDice. Bold and italic data indicate the best and second-best results, respectively.*

The comparison of fusion strategies further shows that semantic-guided window-based cross-attention is more suitable for the current task than direct concatenation and co-attention. Its advantage lies in preserving the dominant role of the semantic branch while allowing edge and frequency-domain information to participate in updates in a controlled manner (Table 6).

*[Table] Ablation results of fusion strategies. Cross-attention achieves the best result on all three mean metrics. Bold and italic data indicate the best and second-best results, respectively.*

#### Design analysis of edge and frequency-domain modeling

Edge scale and frequency-domain transform type are the two most direct module design choices. Table 7 shows that the multi-scale 3/5/7 setting outperforms all single-scale Sobel settings, suggesting that combining multiple gradient scales is beneficial for the boundary patterns represented in these experiments.

*[Table] Ablation results of Sobel scale settings in the edge branch. The multi-scale 3/5/7 setting is best on all three metrics. Bold and italic data indicate the best and second-best results, respectively.*

Table 8 shows that DCT outperforms wavelet in the tested all-channel multi-scale frequency-domain branch. After replacing wavelet with DCT, mAUPR increases by 1.59 percentage points, and mIoU and mDice also improve by 0.30 and 0.52 percentage points, respectively.

*[Table] Ablation results of frequency-domain modeling methods. DCT outperforms wavelet transform on all three mean metrics. Bold data indicate the best result.*

#### Analysis of branch hyperparameters

Tables 9 and 10 analyze the input scale and output capacity of the auxiliary branches, respectively. The 1024×1024 setting achieves the best mAUPR and is therefore selected as the default input scale. The 32/8 channel configuration is best on all 3 metrics; increasing the tested capacity to 64/16 does not provide a further gain.

*[Table] Ablation results of input resolution for the edge and frequency-domain branches. 1024×1024 achieves the highest mAUPR, while 256×256 is slightly higher on mIoU and mDice. Bold and italic data indicate the best and second-best results, respectively.*

*[Table] Ablation results of auxiliary branch output channels. The 32/8 configuration achieves the best results on all 3 mean metrics; both the smaller and larger tested configurations perform worse. Bold and italic data indicate the best and second-best results, respectively.*

#### Effect of training-time contrast augmentation

All settings in Table 11 use the base geometric augmentation pipeline described in Sec. 4.1.4. The geometric-only setting adds no photometric transformation, whereas random gamma-L and random CLAHE-L are applied on top of this pipeline. Contrast-limited adaptive histogram equalization (CLAHE) denotes contrast-limited adaptive histogram equalization. Each added transformation is applied to the CIELAB lightness channel with a probability of 0.5; gamma is sampled from U(0.85,1.15), and the CLAHE clip limit from U(1.2,1.8). Validation and test images remain unchanged, and all other settings follow Sec. 4.1.4.

*[Table] Effect of training time contrast augmentation on DDR (%). All settings include the base geometric augmentation; “+” denotes an additional photometric transformation. Bold and italic data indicate the best and second-best results, respectively.*

Neither added transformation improves the base geometric pipeline. Random gamma-L reduces mAUPR from 47.85% to 46.34%, and random CLAHE-L reduces it to 44.88%; mDice and mIoU follow the same ordering. MA AUPR also decreases from 20.63% to 18.16% and 16.65%, respectively. We therefore retain the base geometric augmentation pipeline for the main comparisons.

### Qualitative Comparison

Figure 2 compares five regions of interest selected from DDR and IDRiD. The examples include small or low-contrast lesions, diffuse boundaries, vessel-adjacent structures, and variation in lesion scale and composition. Each row shows the full-image context, the cropped RGB region, ground truth, and predictions from HRNetV2-W48, M2MRF-A, SegFormer-B5, and SEFCA-Net. In several rows, SEFCA-Net preserves small or fragmented components that at least one baseline misses or merges; in Fig. 2(d), it avoids the isolated vessel region responses shown by the three baselines. These selected examples illustrate local behavior, while the aggregate comparison is reported in Tables 1 and 2.

*[Figure] (a)–(e) Qualitative comparison on representative DDR and IDRiD cases.*

## Discussion and Conclusion

### Discussion

The experiments indicate that SEFCA-Net improves the overall performance across lesion classes rather than dominating every metric. On the DDR dataset, the model achieves the best mAUPR, mDice, and mIoU under the unified reproduced setting. On the IDRiD dataset, its most evident advantage lies in mAUPR, while Dice and IoU remain close to those of strong baselines. This pattern supports competitive overall segmentation under class imbalance, but not uniform superiority across lesions or overlap measures.

This interpretation is supported by the ablation results. Within SEFCA-Net, a semantic-only branch must simultaneously encode contextual information, boundary features, and low-contrast texture features, which is particularly difficult for DR lesion segmentation. Adding the edge branch produces larger gains than adding the frequency branch alone on mAUPR, mDice, and mIoU. When the frequency branch is further combined with the edge branch, mAUPR increases from 46.84% to 47.85%, whereas mDice and mIoU remain slightly below the edge-only configuration. Thus, the ablation supports the contribution of both branches within the proposed network, without assigning a unique function to either branch.

The fusion study further shows that complementary information benefits from selective integration. Compared with direct concatenation and co-attention, semantic-guided window-based cross-attention achieves the best results on all three mean metrics. Within SEFCA-Net, this result suggests that edge and frequency features are more effective when they refine the semantic-dominant decoding stream than when they are concatenated or interact with it symmetrically.

SEFCA-Net also shows competitive transfer performance and efficiency under the evaluated settings. In the cross-dataset scenario where the model is trained on DDR and directly tested on IDRiD, SEFCA-Net achieves the best mAUPR among the reproduced methods. Although the improvement is limited, it suggests that the learned lesion representations are not completely restricted to a single data distribution. In addition, under the standard 1024×1024 resolution setting, SEFCA-Net achieves a higher mAUPR than HRNetV2-W48 and a faster inference speed while keeping the number of parameters almost unchanged. The efficiency result is specific to the hardware and timing settings described in Sec. 4.1.4.

At the same time, this study still has clear limitations. The evaluation is based on only two public DR lesion segmentation datasets and does not yet cover broader multi-center scenarios or pronounced device variability. Although SEFCA-Net achieves the best overall mAUPR among the reproduced methods on IDRiD, it is not consistently optimal on all metrics or lesion categories. Both edge and frequency information are derived from the same fundus image, so the method remains a single-source complementary representation framework rather than a solution designed for cross-device, multi-source, or temporal fundus analysis. Moreover, reporting the best of three runs does not quantify optimization variance, and the contrast study covers only two generic lightness transformations. Random gamma-L and random CLAHE-L do not improve SEFCA-Net under the tested ranges, but this result does not rule out photometric augmentation designed specifically for fundus images.

### Conclusion

We propose SEFCA-Net for diabetic retinopathy lesion segmentation. The method combines HRNet-based semantic representation with learnable edge and frequency modeling and integrates them through semantic-guided window-based cross-attention while retaining the semantic representation as the main decoding stream. This design is intended to improve the representation of tiny lesions, blurred boundaries, and weak responses in low-contrast regions.

On the DDR and IDRiD datasets, SEFCA-Net shows competitive overall performance. On DDR, its mAUPR, mDice, and mIoU are all better than those of the reproduced baselines. On IDRiD, its mAUPR is also the best among the reproduced methods. In the DDR-to-IDRiD evaluation, the model again obtains the best mAUPR among the reproduced methods. Ablation results show that the auxiliary branches and the semantic-guided fusion strategy jointly improve model performance. These results support semantic–edge–frequency fusion as an effective design under the evaluated protocols. Future work will examine broader clinical validation, stronger cross-domain robustness, and lighter deployment-oriented variants.

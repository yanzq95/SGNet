# SGNet: Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-Resolution

### This repository is an official PyTorch implementation of the paper "SGNet: Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-Resolution".

<a href=" ">Paper</a>

## Dependencies
```
Python==3.11.5
PyTorch==2.1.0
numpy==1.23.5 
torchvision==0.16.0
scipy==1.11.3
thop==0.1.1.post2209072238
Pillow==10.0.1
tqdm==4.65.0
```

## Dataset

NYU v2 dataset, our split follow: <a href="http://gofile.me/3G5St/2lFq5R3TL">DKN</a>

RGB-D-D dataset, our split follow: <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.pdf">FDSR</a>

## Models
All pretrained models can be found in <a href="https://drive.google.com/drive/folders/1rRzYDOkDtok8rk4ad03WxRqZbwP-oayR?usp=sharing">SGNet-Model</a>.


## Train
### x4 DSR
> python train.py --scale 4 --num_feats 48
### x8 DSR
>  python train.py --scale 8 --num_feats 40
### x16 DSR
>  python train.py --scale 16 --num_feats 40
### real-world DSR
>  python train.py --scale 4 --num_feats 24

## Test
> python test.py


## Experiments

### Quantitative Results
| SGNet | x4 | x8 | x16 |
|---|---|---|---|
| NYU-v2 | 1.10 | 2.44| 4.77 |
| RGB-D-D | 1.10 | 1.64 | 2.55 |
| Middlebury | 1.15 | 1.64 | 2.95 |
| Lu | 1.03 | 1.61 | 3.55 |

<p align="center">
<img src="figs/histogram.png"/>
</p>

### Visual comparison
Our SGNet can restore more precise depth predictions with clearer and sharper structure.

<b>Real world RGB-D-D: <b/>
<p align="center">
<img src="figs/Patch_RGBDD_Real.png"/>
</p>
<b>NYU-v2 (x16): <b/>
<p align="center">
<img src="figs/Patch_NYU_X16.png"/>
</p>
<b>RGB-D-D (x16): <b/>
<p align="center">
<img src="figs/Patch_RGBDD_X16.png"/>
</p>



## Acknowledgements
We thank the following repos sharing their codes： [DKN](https://github.com/cvlab-yonsei/dkn) and [SUFT](https://github.com/ShiWuxuan/SUFT).


## Citation

If you use this code, please consider citing:

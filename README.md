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

## Models
All pretrained models can be found <a href="https://drive.google.com/drive/folders/1rRzYDOkDtok8rk4ad03WxRqZbwP-oayR?usp=sharing">here</a>.


## Train on synthetic NYU-v2
### x4 DSR
> python train.py --scale 4 --num_feats 48
### x8 DSR
>  python train.py --scale 8 --num_feats 40
### x16 DSR
>  python train.py --scale 16 --num_feats 40
## Train on real-world RGB-D-D
>  python train.py --scale 4 --num_feats 24

## Test
> python test.py


## Experiments

<p align="center">
<img src="figs/histogram.png"/>
</p>

### Visual comparison

<b>Train & test on real-world RGB-D-D: <b/>
<p align="center">
<img src="figs/Patch_RGBDD_Real.png"/>
</p>
<b>Train & test on synthetic NYU-v2 (x16): <b/>
<p align="center">
<img src="figs/Patch_NYU_X16.png"/>
</p>
<b>Train on NYU-v2, test on RGB-D-D (x16): <b/>
<p align="center">
<img src="figs/Patch_RGBDD_X16.png"/>
</p>



## Acknowledgements
We thank all reviewers for their professional and instructive suggestions.

We thank these repos sharing their codes: [DKN](https://github.com/cvlab-yonsei/dkn) and [SUFT](https://github.com/ShiWuxuan/SUFT).


## Citation

If our method proves to be of any assistance, please consider citing:

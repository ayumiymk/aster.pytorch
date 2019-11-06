# ASTER: Attentional Scene Text Recognizer with Flexible Rectification

This repository implements the ASTER in pytorch. Origin software could be found in [here](https://github.com/bgshih/aster).

ASTER is an accurate scene text recognizer with flexible rectification mechanism. The research paper can be found [here](https://ieeexplore.ieee.org/abstract/document/8395027/).

![ASTER Overview](overview.png)

## Installation

```
conda env create -f environment.yml
```

## Train

[**NOTE**] Some users say that they can't reproduce the reported performance with minor modification, like [1](https://github.com/ayumiymk/aster.pytorch/issues/17#issuecomment-527380815) and [2](https://github.com/ayumiymk/aster.pytorch/issues/17#issuecomment-528718596). I haven't try other settings, so I can't guarantee the same performance with different settings. The users should just run the following script without any modification to reproduce the results.
```
bash scripts/stn_att_rec.sh
```

## Test

You can test with .lmdb files by
```
bash scripts/main_test_all.sh
```
Or test with single image by
```
bash scripts/main_test_image.sh
```

## Pretrained model
The pretrained model is available on our [release page](https://github.com/ayumiymk/aster.pytorch/releases/download/v1.0/demo.pth.tar). Download `demo.pth.tar` and put it to somewhere. Before running, modify the `--resume` to the location of this file.

## Reproduced results

|               | IIIT5k |  SVT |  IC03 |  IC13 |  IC15 | SVTP  |  CUTE |
|:-------------:|:------:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  ASTER (L2R)  |  92.67 |   -  | 93.72 | 90.74 |    -  | 78.76 | 76.39 |
| ASTER.Pytorch |  93.2  | 89.2 | 92.2  |   91  |  78.0 |  81.2 |  81.9 |

At present, the bidirectional attention decoder proposed in ASTER is not included in my implementation. 

You can use the codes to bootstrap for your next text recognition research project.


## Data preparation

We give an example to construct your own datasets. Details please refer to `tools/create_svtp_lmdb.py`.

## Citation

If you find this project helpful for your research, please cite the following papers:

```
@article{bshi2018aster,
  author  = {Baoguang Shi and
               Mingkun Yang and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title   = {ASTER: An Attentional Scene Text Recognizer with Flexible Rectification},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  volume  = {}, 
  number  = {}, 
  pages   = {1-1},
  year    = {2018}, 
}

@inproceedings{ShiWLYB16,
  author    = {Baoguang Shi and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title     = {Robust Scene Text Recognition with Automatic Rectification},
  booktitle = {2016 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2016, Las Vegas, NV, USA, June 27-30, 2016},
  pages     = {4168--4176},
  year      = {2016}
}
```

IMPORTANT NOTICE: Although this software is licensed under MIT, our intention is to make it free for academic research purposes. If you are going to use it in a product, we suggest you [contact us](xbai@hust.edu.cn) regarding possible patent issues.

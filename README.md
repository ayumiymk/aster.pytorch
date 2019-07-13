# ASTER: Attentional Scene Text Recognizer with Flexible Rectification

This repository implements the ASTER in pytorch. Origin software could be found in [here](https://github.com/bgshih/aster).

## Train
```
bash scripts/stn_att_rec.sh
```

## Test
```
bash scripts/main_test_all.sh
```

## Reproduced results

|               | IIIT5k |  SVT |  IC03 |  IC13 |  SVTP |  CUTE |
|:-------------:|:------:|:----:|:-----:|:-----:|:-----:|:-----:|
|  ASTER (L2R)  |  92.67 |   -  | 93.72 | 90.74 | 78.76 | 76.39 |
| ASTER.Pytorch |  93.2  | 89.2 | 92.2  |   91  |  81.2 |  81.9 |
|

At present, the bidirectional attention decoder proposed in ASTER is not included in my implementation. 

You can use the codes to bootstrap for your next text recognition research project.


## Data preparation

We give an example to construct your own datasets. Details please refer to `tools/create_svtp_lmdb.py`.


IMPORTANT NOTICE: Although this software is licensed under MIT, our intention is to make it free for academic research purposes. If you are going to use it in a product, we suggest you [contact us](xbai@hust.edu.cn) regarding possible patent issues.
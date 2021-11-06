# pspnet-segmentation
realixation of paper which titled "LIP: Self-supervised Structure-sensitive Learning and A New Benchmark for Human Parsing"
# Single-Human-Parsing-LIP

PSPNet implemented in PyTorch for **single-person human parsing** task, evaluating on Look Into Person (LIP) dataset.

## Model

The implementation of PSPNet is based on [HERE](https://github.com/Lextal/pspnet-pytorch).

Trained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/13DzOvUoIx0JR-BTEilhLqdAIp3h0H5Zj) or [Baidu Drive](https://pan.baidu.com/s/1SuGbwL1CF7pLxN1olBc49Q) (提取码：43cu).

## Environment

* Python 3.8
* PyTorch == 1.9.1+cu111
* torchvision == 0.10.1+cu111
* TensorBoard ==2.7.0

## Dataset

To use our code, firstly you should download LIP dataset from [HERE](http://sysu-hcp.net/lip/index.php). But the dataset probably is not available, and you can email me for accessing dataset.

Then, reorganize the dataset folder as below:

```
LIP
│ 
└───TrainVal_images
│   │   train_id.txt
│   │   val_id.txt
│   │
│   └───train_images
│   │   │   77_471474.jpg
│   │   │   113_1207747.jpg
│   │   │   ...
│   │
│   └───val_images
│   │   │   100034_483681.jpg
│   │   │   10005_205677.jpg
│   │   │   ...
│
└───TrainVal_parsing_annotations
│   │
│   └───train_segmentations
│   │   │   77_471474.png
│   │   │   113_1207747.png
│   │   │   ...
│   │
│   └───val_segmentations
│   │   │   100034_483681.png
│   │   │   10005_205677.png
│   │   │   ...
│
└───Testing_images
│   │   test_id.txt
│   │
│   └───test_images
│   │   │   100012_501646.jpg
│   │   │   ...
```

## Visualization
﻿![20 epoch train](https://img-blog.csdnimg.cn/6a3c6b4547664adaa384f78c74678601.png#pic_center)

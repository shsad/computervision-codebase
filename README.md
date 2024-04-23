2D Computer Vision-Codebase
- 
This project aims to offer a user-friendly codebase for initiating 2D Computer Vision tasks.
It includes code for training different deep learning models (such as `VGG19`, `ResNet50`, `ResNet20`, etc) on various datasets, utilizing a range of optimizers.
### Setup
- Dependencies: `pandas`, `torch`, `torchvision`, `natsort`, `pillow`, `tqdm`.
- Following models are available: [`ResNet18`](https://arxiv.org/abs/1512.03385), [`ResNet50`](https://arxiv.org/abs/1512.03385), [`VGG19`](https://arxiv.org/abs/1409.1556), [`LeNet`](https://arxiv.org/abs/1706.06083), [`Small CNN`](https://arxiv.org/abs/1608.04644), [`ResNet20`](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), [`WideResNet`](https://arxiv.org/abs/1605.07146), `Basic MLP`, [`Basic CNN`](https://arxiv.org/abs/1608.04644).

- Following datasets are available: [`CIFAR10`](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), [`CIFAR100`](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), [`MNIST`](http://yann.lecun.com/exdb/mnist/), [`NIPS2017`](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/data), [`ImageNet`](https://www.image-net.org/). 

- Create folders `Data` and `Trained_Models` using command `mkdir Data` and `mkdir Trained_Models` respectively. 
- To train a model, execute the main file with numerous arguments. For e.g., `python main.py --model 'ResNet20' --dataSet 'CIFAR10' --optim 'Adam' --maxIterations 10 --batchSize 64 --numWorkers 2 --saveModel 1 --ver 1` will train a `ResNet20` model on the `CIFAR10` data set using the [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer with a batch size of 64 along with max epochs of 10. The `--numworkers` option allows the user to select the number of workers for loading the dataset. Set `--saveModel` to 1 to save the model otherwisee set to 0, `--ver` will allow for additional information to be displayed.
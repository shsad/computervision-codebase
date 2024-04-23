import argparse
import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models import resnet50, vgg19, resnet18
from torchvision import transforms
from torch.optim import Adam, Adagrad, SGD
import warnings

warnings.filterwarnings("ignore")

from models import *
from train_model import *
from utils import *

if __name__ == "__main__":

    # -------------------------------- Reading in the command line arguments --------------------------------#

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
                        choices=['ResNet50', 'VGG19', 'ResNet20', 'WideResNet', 'BasicCNN', 'BasicMLP', 'ResNet18',
                                 'BasicLeNet', 'SmallCNN'],
                        help=('Choices ResNet50 and VGG19 for ImageNet or NIPS2017 datasets.'
                              'Choices ResNet20, WideResNet and BasicCNN for CIFAR10 and CIFAR100 datasets.'
                              'BasicMLP for MNIST.'), type=str, required=True)
    parser.add_argument('--dataSet', choices=['CIFAR10', 'CIFAR100', 'NIPS2017', 'ImageNet', 'MNIST']
                        , type=str, required=True)
    parser.add_argument('--optim', choices=['SGD', 'Adam', 'Adagrad', 'NAG'], type=str, required=True)
    parser.add_argument('--maxIterations', type=int, required=True)
    parser.add_argument('--batchSize', type=int, required=True)
    parser.add_argument('--numWorkers', type=int, required=True)
    parser.add_argument('--saveModel', choices=[0, 1], help=('Set it to 1 to save the trained model otherwise 0.'),
                        type=int,
                        required=True)
    parser.add_argument('--ver', choices=[0, 1], help=("Set to 1 to display information otherwise 0."), type=int,
                        required=True)

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    if args.dataSet in ['ImageNet', 'NIPS2017'] and args.model in ['ResNet20', 'WideResNet', 'BasicCNN', 'BasicMLP',
                                                                   'SmallCNN', 'BasicLeNet']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}.")
    elif (args.dataSet == 'CIFAR10' or args.dataSet == 'CIFAR100') and args.model in ['ResNet50', 'VGG19', 'BasicMLP',
                                                                                      'SmallCNN', 'BasicLeNet']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}.")
    elif args.dataSet == 'MNIST' and args.model in ['ResNet50', 'VGG19', 'ResNet20', 'WideResNet', 'BasicCNN']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}")
    elif args.dataSet == 'ImageNet' and args.model not in ['ResNet50', 'VGG19']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}.")
    # -------------------------------- Model Transformations --------------------------------#

    # Here, the first value is the training transformation and the second one is the test transformation
    modelTrainingTransforms = {

        'VGG19': [
            transforms.Compose([
                transforms.Resize(224),
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(224, padding=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),

            transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ],

        'ResNet50': [
            transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),

            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ],

        'ResNet20': [
            transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),

            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        ],

        'ResNet18': [
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),

            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        ],

        'WideResNet': [
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),

            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ],

        'BasicCNN': [
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),

            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        ],

        'BasicMLP': [
            transforms.Compose([
                transforms.ToTensor(),
            ]),

            transforms.Compose([
                transforms.ToTensor(),
            ])
        ],

        'BasicLeNet': [
            transforms.Compose([
                transforms.ToTensor(),
            ]),

            transforms.Compose([
                transforms.ToTensor(),
            ])
        ],

        'SmallCNN': [
            transforms.Compose([
                transforms.ToTensor(),
            ]),

            transforms.Compose([
                transforms.ToTensor(),
            ])
        ]

    }

    # -------------------------------- Initialising the dataset and dataloader --------------------------------#

    if args.dataSet == "MNIST":

        trainingData = datasets.MNIST(
            root="Data",
            train=True,
            download=True,
            transform=modelTrainingTransforms[args.model][0]
        )

        testData = datasets.MNIST(
            root="Data",
            train=False,
            download=True,
            transform=modelTrainingTransforms[args.model][1]
        )

        if args.model in ['BasicLeNet', 'SmallCNN']:
            isANN = False
        else:
            isANN = True

    elif args.dataSet == 'CIFAR10':

        trainingData = datasets.CIFAR10(
            root="Data",
            train=True,
            download=True,
            transform=modelTrainingTransforms[args.model][0]
        )

        testData = datasets.CIFAR10(
            root="Data",
            train=False,
            download=True,
            transform=modelTrainingTransforms[args.model][1]
        )

        isANN = False

    elif args.dataSet == 'CIFAR100':

        trainingData = datasets.CIFAR100(
            root="Data",
            train=True,
            download=True,
            transform=modelTrainingTransforms[args.model][0]
        )

        testData = datasets.CIFAR100(
            root="Data",
            train=False,
            download=True,
            transform=modelTrainingTransforms[args.model][1]
        )

        isANN = False

    elif args.dataSet == 'NIPS2017':

        NIPlabels = pd.read_csv('./Data/NIPS2017/images.csv')
        trainingData = CustomDataSet(main_dir='./Data/NIPS2017/images',
                                     labels=NIPlabels,
                                     transform=modelTrainingTransforms[args.model][0]
                                     )
        testData = CustomDataSet(main_dir='./Data/NIPS2017/images',
                                 labels=NIPlabels,
                                 transform=modelTrainingTransforms[args.model][1])

        isANN = False

    elif args.dataSet == 'ImageNet':

        trainingData = datasets.ImageNet(root='/software/ais2t/pytorch_datasets/imagenet/',
                                         split='train',
                                         transform=modelTrainingTransforms[args.model][0])
        testData = datasets.ImageNet(root='/software/ais2t/pytorch_datasets/imagenet/',
                                     split='val',
                                     transform=modelTrainingTransforms[args.model][1])

        isANN = False

    trainLoader = DataLoader(dataset=trainingData,
                             batch_size=args.batchSize,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=args.numWorkers)

    testLoader = DataLoader(dataset=testData,
                            batch_size=args.batchSize,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.numWorkers)

    # -------------------------------- Initialising the model --------------------------------#

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.model == 'ResNet50':
        model = resnet50()

    elif args.model == 'VGG19':
        model = vgg19()
        model.apply(init_weights)

    elif args.model == 'BasicCNN':
        if args.dataSet == 'CIFAR10':
            model = getBasicCNN(num_classes=10)
        elif args.dataSet == 'CIFAR100':
            model = getBasicCNN(num_classes=100)

    elif args.model == 'ResNet20':
        if args.dataSet == 'CIFAR10':
            model = ResNet20(num_classes=10)
        elif args.dataSet == 'CIFAR100':
            model = ResNet20(num_classes=100)

    elif args.model == 'WideResNet':
        if args.dataSet == 'CIFAR10':
            model = WideResNet(num_classes=10)
        elif args.dataSet == 'CIFAR100':
            model = WideResNet(num_classes=100)

    elif args.model == 'BasicLeNet':
        model = getLeNet()

    elif args.model == 'SmallCNN':
        model = getSmallCNN()

    elif args.model == 'BasicMLP':
        model = getBasicMLP()

    elif args.model == 'ResNet18':
        model = resnet18()

        if args.dataSet == 'CIFAR10':
            inFeatures = model.fc.in_features
            model.fc = nn.Linear(inFeatures, 10)
        elif args.dataSet == 'CIFAR100':
            inFeatures = model.fc.in_features
            model.fc = nn.Linear(inFeatures, 100)

        model.apply(init_weights)

    model = model.to(device)

    # -------------------------------- Initialising the optimizer --------------------------------#

    lr = 1e-2
    weightDecay = 5e-4

    if args.optim == 'SGD':
        optim = SGD(params=model.parameters(),
                    lr=lr,
                    weight_decay=weightDecay
                    )

    elif args.optim == 'Adagrad':
        optim = Adagrad(params=model.parameters(),
                        lr=lr,
                        weight_decay=weightDecay
                        )

    elif args.optim == 'Adam':
        optim = Adam(params=model.parameters(),
                     lr=lr,
                     weight_decay=weightDecay
                     )

    else:
        optim = NesterovMomentumSGD(params=model.parameters())
    # -------------------------------- Initialising the loss function and parameters --------------------------------#

    lossFunction = torch.nn.CrossEntropyLoss()
    saveModel = False if args.saveModel == 0 else True
    ver = True if args.ver == 1 else 0

    trainModel = TrainModel(model=model,
                            maxIters=args.maxIterations,
                            device=device,
                            trainDataloader=trainLoader,
                            testDataloader=testLoader,
                            saveModel=saveModel,
                            optim=optim,
                            lossFunction=lossFunction,
                            isANN=isANN,
                            ver=ver
                            )
    trainModel.train(args)

    if saveModel:
        print(f"The model has been saved in the Trained_Models directory.")


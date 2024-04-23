import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class TrainModel():
    def __init__(self,
                 model,
                 maxIters,
                 device,
                 trainDataloader,
                 testDataloader,
                 lossFunction,
                 optim,
                 isANN=False,
                 saveModel=False,
                 ver=False):

        self.model = model
        self.maxIters = maxIters
        self.device = device
        self.trainDataloader = trainDataloader
        self.testDataloader = testDataloader
        self.saveModel = saveModel
        self.ver = ver
        self.lossFunction = lossFunction
        self.optim = optim
        self.isANN = isANN

    def train(self, args):
        if self.ver:
            print(
                f"DataSet: {args.dataSet} | Model: {args.model} | Max Epochs: {args.maxIterations} | Optimizer: {args.optim}")

        for epoch in range(self.maxIters):
            if args.optim != 'NAG':
                scheduler = StepLR(self.optim, step_size=int(self.maxIters / 2), gamma=0.1)

            trainingLoop = tqdm(iterable=enumerate(self.trainDataloader),
                                leave=False,
                                total=len(self.trainDataloader))

            for _, (x, y) in trainingLoop:

                x = x.to(self.device)
                y = y.to(self.device)

                if self.isANN:
                    x = x.reshape(x.shape[0], -1)

                pred = self.model(x)
                loss = self.lossFunction(pred, y)

                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                self.optim.step()

                trainingLoop.set_description(f"[Epoch {epoch + 1}/{self.maxIters}]")
                trainingLoop.set_postfix(loss=loss.item())
            if args.optim != 'NAG':
                scheduler.step()

        self.evalModel()
        if self.saveModel:
            torch.save(self.model.state_dict(),
                       f'Trained_Models/{args.model}_{args.dataSet}_{args.optim}_trained_model.pt')

    def evalModel(self):

        self.model.eval()

        testLoop = tqdm(iterable=enumerate(self.testDataloader),
                        leave=False,
                        total=len(self.testDataloader))

        num_correct = 0
        num_samples = 0

        for _, (x, y) in testLoop:

            with torch.no_grad():
                x = x.to(self.device)
                y = y.to(self.device)

                if self.isANN:
                    x = x.reshape(x.shape[0], -1)

                pred = self.model(x)

                num_correct += (torch.argmax(pred, dim=1) == y).sum()
                num_samples += x.shape[0]

        if self.ver:
            accuracy = num_correct / num_samples * 100
            formatted_accuracy = "{:.2f}".format(accuracy)

            print(f"The total accuracy is {formatted_accuracy}%")
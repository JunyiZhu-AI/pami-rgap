import torchvision
import argparse
import os
import yaml
import json
from utils import *
from models import CNN6, CNN6d, FCN3, LeNetOutput
from recursive_attack import r_gap, peeling, fcn_reconstruction, inverse_udldu
import matplotlib.pyplot as plt

with open("../config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
parser = argparse.ArgumentParser(description="Model related arguments. For other configurations please check CONFIG file.")
parser.add_argument("-d", "--dataset", help="Choose the data source.", choices=["CIFAR10", "MNIST"], default="CIFAR10")
parser.add_argument("-i", "--index", help="Choose a specific image to reconstruct.", type=int, default=-1)
parser.add_argument("-b", "--batchsize", default=1, help="Mini-batch size", type=int)
parser.add_argument("-p", "--parameters", help="Load pre-trained model.", default=None)
args = parser.parse_args()
setup = {'device': 'cpu', 'dtype': torch.float32}
print(f'Running on {setup["device"]}, PyTorch version {torch.__version__}')


def main():
    ratios_output = {1: [], 2: [], 3: [], 4: []}
    ratios_grad = {1: [], 2: [], 3: [], 4: []}
    for i in range(10):
        train_sample, test_sample = dataloader(dataset=args.dataset, mode="attack", index=args.index,
                                               batchsize=2, config=config, seed=i)
        # set up inference framework
        torch.manual_seed(i)
        np.random.seed(i)
        net = LeNetOutput().to(**setup)
        net.apply(weight_init).eval()
        pred_loss_fn = logistic_loss

        tt = torchvision.transforms.ToTensor()
        image, label = list(zip(*train_sample))
        x0 = tt(image[0]).unsqueeze(0).to(**setup)
        x1 = tt(image[1]).unsqueeze(1).to(**setup)

        # generate gradients of real data
        pred0, output0 = net(x0)
        # reversed label to make sure mu is unique, just for better demonstration
        y = torch.tensor([0 if p > 0 else 1 for p in pred0]).to(**setup)
        print(f'pred0: {pred0.detach().numpy()}, y: {y}')
        pred_loss0 = pred_loss_fn(inputs=pred0, target=y)
        dy_dx = torch.autograd.grad(pred_loss0, list(net.parameters()))
        dy_dx0 = [g.detach().clone() for g in dy_dx]

        net.zero_grad()
        pred1, output1 = net(x1)
        # reversed label to make sure mu is unique, just for better demonstration
        y = torch.tensor([0 if p > 0 else 1 for p in pred1]).to(**setup)
        print(f'pred1: {pred1.detach().numpy()}, y: {y}')
        pred_loss1 = pred_loss_fn(inputs=pred1, target=y)
        dy_dx = torch.autograd.grad(pred_loss1, list(net.parameters()))
        dy_dx1 = [g.detach().clone() for g in dy_dx]

        for j in range(len(dy_dx1)):
            ratios_grad[j+1].append(
                (torch.norm(dy_dx0[j] - dy_dx1[j]) / torch.norm(dy_dx1[j])).item()
            )
            ratios_output[j+1].append(
                (torch.norm(output0[j] - output1[j]) / torch.norm(output1[j])).item()
            )

    grad_res = {}
    output_res = {}
    for i in range(1, 5):
        grad_res[i] = np.mean(ratios_grad[i])
        output_res[i] = np.mean(ratios_output[i])

    res = {
        'grad': grad_res,
        'ouput': output_res
    }

    with open(os.path.join("/home/junyi/R-GAP/test", "LeNet_output.json"), "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()

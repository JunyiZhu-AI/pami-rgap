import numpy as np
import torch
from scipy.sparse import block_diag
from utils import *
from conv2circulant import *

setup = {'device': 'cpu', 'dtype': torch.float32}


def logistic_loss(y, pred):
    y = torch.tensor(y).to(**setup)
    pred = torch.squeeze(pred, -1)
    return torch.mean(-(y*torch.log(pred)+(1-y)*torch.log(1-pred)))


def inverse_udldu(udldu):
    '''derive u from udldu using gradient descend based method'''
    lr = 0.01
    u = torch.tensor(0).to(**setup).requires_grad_(True)
    udldu = torch.tensor(udldu).to(**setup)
    optimizer = torch.optim.Adam([u], lr=lr)

    iters = 30000
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[iters // 5, iters // 2],
        gamma=0.1)
    loss_fn = nn.MSELoss()
    for i in range(iters):
        optimizer.zero_grad()
        udldu_ = -u / (1 + torch.exp(u))
        l = loss_fn(udldu_, udldu)
        l.backward()
        optimizer.step()
        scheduler.step()
    udldu_ = -u / (1 + torch.exp(u))
    print(f"The error term of inversing udldu: {udldu.item()-udldu_.item():.1e}")
    return u.detach().numpy()


def multiclass_first_step(grad, weight, label):
    y = label_to_onehot(torch.tensor(label), weight.shape[0])
    weight = torch.tensor(weight.transpose())
    weight.requires_grad = True
    lr = 0.01
    if len(label) == 1:
        x_ = torch.tensor(grad[label]).mul(-1)
        with torch.no_grad():
            beta = torch.norm(x_)
            x_ = x_ / beta
        beta.requires_grad = True
        optimizer = torch.optim.Adam([beta], lr=lr)

        iters = 10000
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[iters // 5, iters // 2],
            gamma=0.1)
        grad = torch.tensor(grad.transpose())
        for i in range(iters):
            optimizer.zero_grad()
            pred = torch.matmul(x_.mul(beta), weight)
            l = crossentropy_for_onehot(pred, y)
            grad_ = torch.autograd.grad(l, weight, create_graph=True)
            l_x = (grad - grad_[0]).square().sum()
            l_x.backward()
            optimizer.step()
            scheduler.step()

        x_.mul_(beta)
    else:
        x_ = torch.tensor(grad[label]).mul(-1)
        x_.requires_grad = True

        optimizer = torch.optim.Adam([x_], lr=lr)
        iters = 10000
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[iters // 5, iters // 2],
            gamma=0.1)
        grad = torch.tensor(grad.transpose())
        for i in range(iters):
            optimizer.zero_grad()
            pred = torch.matmul(x_, weight)
            l = crossentropy_for_onehot(pred, y)
            grad_ = torch.autograd.grad(l, weight, create_graph=True)
            l_x = (grad - grad_[0]).square().sum()
            l_x.backward()
            optimizer.step()
            scheduler.step()

    print(f"Numerical refinement error: {l_x.item()}")
    k = torch.softmax(torch.matmul(x_, weight), dim=1) - y
    return x_.detach().numpy(), k.detach().numpy()


def peeling(in_shape, padding):
    if padding == 0:
        return np.ones(shape=in_shape[1:], dtype=bool)
    h, w = np.array(in_shape[-2:]) + 2*padding
    toremain = np.ones(h*w*in_shape[1], dtype=np.bool)
    if padding:
        for c in range(in_shape[1]):
            for row in range(h):
                for col in range(w):
                    if col < padding or w-col <= padding or row < padding or h-row <= padding:
                        i = c*h*w + row*w + col
                        assert toremain[i]
                        toremain[i] = False
    return toremain


def padding_constraints(in_shape, padding):
    toremain = peeling(in_shape, padding)
    P = []
    for i in range(toremain.size):
        if not toremain[i]:
            P_row = np.zeros(toremain.size, dtype=np.float32)
            P_row[i] = 1
            P.append(P_row)
    return np.array(P)


def cnn_reconstruction(in_shape, k, g, out, kernel, stride, padding):
    coors, x_len, y_len = generate_coordinates(x_shape=in_shape, kernel=kernel, stride=stride, padding=padding)
    K = aggregate_g(k=k, x_len=x_len, coors=coors)
    W_single = circulant_w(x_len=x_len, kernel=kernel, coors=coors, y_len=y_len)
    W = block_diag([W_single] * in_shape[0]).toarray()
    P_single = padding_constraints(in_shape=in_shape, padding=padding)
    P = block_diag([P_single] * in_shape[0]).toarray()
    p = np.zeros(shape=P.shape[0], dtype=np.float32)
    # Tricks to increase the numerical stability.
    P *= np.max(W)
    ratio = np.max(W) / np.max(K)
    K *= ratio
    g *= ratio

    if np.any(P):
        a = np.concatenate((K, W, P), axis=0)
        b = np.concatenate((g.reshape(-1) * in_shape[0], out, p), axis=0)
    else:
        a = np.concatenate((K, W), axis=0)
        b = np.concatenate((g.reshape(-1) * in_shape[0], out), axis=0)
    result = np.linalg.lstsq(a, b, rcond=None)
    print(f'lstsq residual: {result[1]}, rank: {result[2]} -> {W.shape[-1]}, '
          f'max/min singular value: {result[3].max():.2e}/{result[3].min():.2e}')
    x = result[0]
    x = np.split(x, in_shape[0])
    x = [xi[peeling(in_shape=in_shape, padding=padding)] for xi in x]
    return np.array(x), W_single


def fcn_reconstruction(k, gradient):
    x = [g / c for g, c in zip(gradient, k) if c != 0]
    x = np.mean(x, 0)
    return np.array([x])


def r_gap(out, k, g, x_shape, weight, module):
    # obtain information of the convolution kernel
    if isinstance(module.layer, nn.Conv2d):
        padding = module.layer.padding[0]
        stride = module.layer.stride[0]
    else:
        padding = 0
        stride = 1

    x, weight = cnn_reconstruction(in_shape=x_shape, k=k, g=g, kernel=weight, out=out, stride=stride, padding=padding)
    return x, weight








# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

import torch
from torch import nn
from torch.optim import Adam

from moe import MoE


def train(x, y, model, loss_fn, optim):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model


def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))


def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    # dummy target
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x, y


# arguments
input_size = 1000
num_classes = 20
num_experts = 10
hidden_size = 64
batch_size = 5
k = 4

# determine device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# instantiate the MoE layer
model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = Adam(model.parameters())

x, y = dummy_data(batch_size, input_size, num_classes)

# train
model = train(x.to(device), y.to(device), model, loss_fn, optim)
# evaluate
x, y = dummy_data(batch_size, input_size, num_classes)
eval(x.to(device), y.to(device), model, loss_fn)

import argparse
import torch.nn as nn
from torch.optim import SGD
from data import get_dataset
from models import build_model
from torch.utils.data import DataLoader
from engine.trainer import do_train

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lenet')
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--dataset_root', type=str, default='/home/files/dataset/data')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--log_interval', type=int, default=50)
opt = parser.parse_args()

model = build_model(opt)
train_set, test_set = get_dataset(opt)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)
optimizer = SGD(model.parameters(), lr=opt.lr)
loss_fn = nn.NLLLoss()

do_train(opt, model, train_loader, test_loader, optimizer, loss_fn)

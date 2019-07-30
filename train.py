import argparse
import torch.nn as nn
from torch.optim import SGD, Adam
from data import get_dataloader
from models import build_model
from engine import do_train
from loss import get_loss_fn

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lenet')
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--dataset_root', type=str, default='/home/files/dataset/data')
parser.add_argument('--loss_fn', type=str, default='softmax')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--log_interval', type=int, default=50)

parser.add_argument('--use_triplet', type=int, default=0)
parser.add_argument('--identify_instances', type=int, default=8)
parser.add_argument('--triplet_margin', type=float, default=0.5)
parser.add_argument('--k_list', type=list, default=[1, 2, 4])

parser.add_argument('--bnneck', type=int, default=0)
parser.add_argument('--last_stride', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=197)

opt = parser.parse_args()
print(opt)

model = build_model(opt)
train_loader, test_loader = get_dataloader(opt)
optimizer = SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
loss_fn = get_loss_fn(opt)

trainer = do_train(opt)
trainer(opt, model, train_loader, test_loader, optimizer, loss_fn)

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def PCA(data, k=2):
    U, S, V = torch.svd(data.t())
    return torch.mm(data, U[:, :k])


def visualise(opt, data, target):
    # print(data)
    data = torch.cat(data, dim=0)
    if data.device is not 'cpu':
        data = data.cpu()
    target_names = [i for i in range(opt.num_classes)]
    data_PCA = PCA(data)
    data_PCA = data_PCA.numpy()

    plt.figure()
    target = np.array(target)
    for i, target_name in enumerate(target_names):
        plt.scatter(data_PCA[target == i, 0], data_PCA[target == i, 1], label=target_name)

    #plt.legend()
    plt.savefig(os.path.join(opt.output_dir, 'inference.png'))


# if __name__ == '__main__':
#     data = torch.randn((50, 10))
#     target = torch.randint(1, 10, (50,))
#     target_names = torch.arange(1, 10)
#     visualise(data, target, target_names)

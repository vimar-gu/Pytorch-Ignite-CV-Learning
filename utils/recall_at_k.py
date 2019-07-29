import torch

from ignite.utils import to_onehot
from ignite.metrics.metric import Metric


class Recall_at_k(Metric):
	def __init__(self, k_list):
		self.k_list = k_list
		self._true_positives = None
		self._positives = None
		self.eps = 1e-20
		super(Recall_at_k, self).__init__()

	def reset(self):
		self._true_positives = [0 for k in range(len(self.k_list))]
		self._positives = 0
		super(Recall_at_k, self).reset()

	def update(self, output):
		y_pred, y = output
		num_classes = y_pred.shape[1]
		y = to_onehot(y.view(-1), num_classes=num_classes)
		y_pred = torch.argsort(y_pred, dim=1, descending=True)

		for k in range(len(self.k_list)):
			correct = torch.zeros_like(y_pred).type(torch.FloatTensor).to(y_pred.device)
			for i in range(self.k_list[k]):
				pred = to_onehot(y_pred[:, i], num_classes)
				y = y.type_as(pred)
				correct += y * pred
			self._true_positives[k] += correct.sum(dim=0)
		self._positives += y.sum(dim=0)

	def compute(self):
		recall = [0 for k in range(len(self.k_list))]
		for k in range(len(self.k_list)):
			recall[k] = (self._true_positives[k] / (self._positives + self.eps)).mean().item()
		return recall

"""metric 정의
"""
import torch

class ConfMatrix(object):
    def __init__(self, num_classes):    # == num_labels: 4
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:  # mat = 5 * 5 torch 행렬
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)  # target = 0 ~ 4 범위 안에서만, k = 0 or 1
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)  # 최소 길이는 25 (5**2)
            # print(f"\ncall ConfMatrix.update, self.mat is : {self.mat}")

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu[iu==iu]).item(), acc.item()


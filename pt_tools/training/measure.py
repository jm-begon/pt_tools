from abc import ABCMeta
import numpy as np
import torch
from torch.nn import functional as F

from pt_tools.training.util import Deviceable
from ..architectures.memoizer import PseudoMemoizer, Memoizer


class Measure(object):
    """
    `Measure`
    =========
    Measure some property on prediction matrix.
    """
    def __call__(self, predictions, hard_target=None, soft_target=None,
                 reduction="batchmean"):
        """
        Parameters
        ----------
        predictions: float tensor [batch_size, n_classes]
            Some predictions
        hard_target: int tensor [batch_size]  where each value is :math:`0 \leq
        \text{targets}[i] \leq C-1` or None
            The true class
        soft_target: float tensor [batch_size, n_classes] or None
            What the predictions should have been
        reduction: str  'batchmean'|'mean'|'sum' (default: 'batchmean')
            The reduction to apply

        Note
        ----
        None, one or two of `hard_target` and `soft_target` can be set to None.
        It is the responsibility of the user to use the sub-classes correctly

        Return
        ------
        measure: scalar as float tensor []
            The value of the measure
        """
        pass


def accuracy_loss(input, hard_target=None, soft_target=None,
                  reduction="batchmean"):
    """Measure"""
    pred = input.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    loss = pred.eq(hard_target.view_as(pred))
    return loss.sum() if reduction == "sum" else loss.mean()


def agreement_loss(prediction, hard_target=None, soft_target=None,
                   reduction="batchmean"):
    pred = prediction.data.max(1, keepdim=True)[1]
    hard_target = soft_target.data.max(1, keepdim=True)[1]
    loss = pred.eq(hard_target).float()
    return loss.sum() if reduction == "sum" else loss.mean()


def top5_acc_loss(input, hard_target=None, soft_target=None, reduction="batchmean"):
    top5 = input.topk(5, 1)
    same = top5.eq(hard_target.view(-1, 1).expand_as(top5)).float().sum(dim=1)
    return same.sum() if reduction == "sum" else same.mean()


class AgreementLoss(Measure):
    def __init__(self, override_hard=False):
        self.override_hard = override_hard

    def __call__(self, predictions, hard_target=None, soft_target=None,
                 reduction="batchmean"):
        pred = predictions.data.max(dim=1)[1]  # [1] is argmax
        if self.override_hard and soft_target is not None:
            hard_target = soft_target.data.max(dim=1)[1]

        loss = pred.eq(hard_target).float()
        if reduction is None or reduction == "none":
            return loss
        return loss.sum() if reduction == "sum" else loss.mean()


class CrossEntropy(Measure):
    def __init__(self, soft_is_logit=True, override_hard=False):
        self.soft_is_logit = soft_is_logit
        self.override_hard = override_hard

    def __call__(self, predictions, hard_target=None, soft_target=None,
                 reduction="batchmean"):
        if not self.override_hard:
            # target is the class
            return F.cross_entropy(predictions, hard_target,
                                   reduction=reduction)

        # target are in one-hot format or is another distribution
        x = F.softmax(soft_target, dim=1) if self.soft_is_logit else soft_target
        log_y = F.log_softmax(predictions, dim=1)
        loss = -x * log_y
        if reduction == "mean":
            return loss.mean()
        loss = torch.sum(loss, dim=1)
        if reduction is None or reduction == "none":
            return loss
        return loss.mean() if reduction == "batchmean" else loss.sum()


class JSDivergence(Measure):
    def __init__(self, soft_is_logit=True, override_hard=True, base=None):
        self.soft_is_logit = soft_is_logit
        self.override_hard = override_hard
        self.base = base

    def __call__(self, predictions, hard_target=None, soft_target=None,
                 reduction="batchmean"):
        if not self.override_hard:
            # target is the class
            q = F.softmax(predictions, dim=1) / 2.
            m = q.clone()
            m[hard_target] += .5

            m = torch.log(m)

            kl_pm = F.cross_entropy(m, hard_target, reduction=reduction)
            kl_qm = F.kl_div(m, q, reduction=reduction)

        else:
            # target are in one-hot format or is another distribution
            p = F.softmax(soft_target, dim=1) if self.soft_is_logit else \
                soft_target
            q = F.softmax(predictions, dim=1)

            m = .5 * (p + q)

            m = torch.log(m)  # Maybe not the most stable way

            kl_pm = F.kl_div(m, p, reduction=reduction)
            kl_qm = F.kl_div(m, q, reduction=reduction)

        if reduction == "none":
            kl_pm = kl_pm.sum(dim=1)
            kl_qm = kl_qm.sum(dim=1)

        js_pq = .5 * (kl_pm + kl_qm)
        if self.base is None:
            return js_pq

        return js_pq / np.log(self.base)


def std_pseudo_loss(input, hard_target=None, soft_target=None,
                    reduction="batchmean"):
    """
    Compute standard deviation of the prediction, rowwise. The goal is to get
    an estimate of the (un)certainty of the prediction
    """
    input = F.softmax(input, dim=1)
    std = input.std(dim=1)
    return std.sum() if reduction == "sum" else std.mean()


def entropy_pseudo_loss(input, hard_target=None, soft_target=None,
                        reduction="batchmean"):
    """
    Compute entropy of the prediction, rowwise. The goal is to get
    an estimate of the (un)certainty of the prediction
    """
    input = F.softmax(input, dim=1)
    mask = (input == 0)
    input[mask] = 1e-20
    res = -input * torch.log2(input)
    res[mask] = 0
    res = res.sum(dim=1)
    if reduction is None or reduction == "none":
        return res
    return res.sum() if reduction == "sum" else res.mean()


def maxoutput_pseudo_loss(input, hard_target=None, soft_target=None,
                          reduction="batchmean"):
    input = F.softmax(input, dim=1)
    max, argmax = input.max(dim=1)
    if reduction is None or reduction == "none":
        return max
    return max.sum() if reduction == "sum" else max.mean()


class Reductor(object, metaclass=ABCMeta):
    def __init__(self, n_losses, device):
        self.n_losses = n_losses
        self.device = device

    @property
    def reduction(self):
        return ""

    def add(self, index, value, size):
        pass

    def __iter__(self):
        pass


class SumReductor(Reductor):
    def __init__(self, n_losses, device):
        super().__init__(n_losses, device)
        self.losses_values = torch.zeros(self.n_losses).to(self.device)
        self.size = 0

    @property
    def reduction(self):
        return "sum"

    def add(self, index, value, size):
        self.losses_values[index] += value.item()
        if index == 0:
            self.size += size

    def __iter__(self):
        for i in range(self.n_losses):
            yield (self.losses_values[i] / self.size).item()


class NoneReductor(Reductor):
    def __init__(self, n_losses, device):
        super().__init__(n_losses, device)
        self.losses_values = [[] for _ in range(n_losses)]

    @property
    def reduction(self):
        return "none"

    def add(self, index, value, size):
        self.losses_values[index].append(value.detach().cpu())

    def __iter__(self):
        for values in self.losses_values:
            yield torch.cat(values).cpu().numpy()


class Tester(Deviceable):
    def __init__(self, model, use_cuda,
                 losses=(CrossEntropy(), accuracy_loss),
                 reductor_factory=SumReductor):
        super().__init__(use_cuda)
        self.model = model
        self._losses = losses
        self.model.to(self.device)
        self.reductor_factory = reductor_factory

    def test(self, loader):
        self.model.eval()

        losses_values = self.reductor_factory(len(self._losses), self.device)

        with torch.no_grad():
            for batch in loader:
                data, target = batch[0], batch[1]
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                for i, loss_f in enumerate(self._losses):
                    losses_values.add(i,
                                      loss_f(output,
                                             hard_target=target,
                                             reduction=losses_values.reduction),
                                      data.size(0))

        return tuple(iter(losses_values))


class ComputeAUROC(Deviceable):
    def __init__(self, model, use_cuda):
        super().__init__(use_cuda)
        self.model = model

    def test(self, loader):
        from sklearn.metrics import roc_auc_score
        true_labels = []
        pred_labels = []

        def prep(tensor):
            return tensor.detach().cpu().numpy()

        self.model.eval()

        with torch.no_grad():
            for batch in loader:
                data, target = batch[0], batch[1]
                true_labels.append(prep(target))
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred_labels.append(prep(output).argmax(axis=1))

        y_pred = np.concatenate(pred_labels).ravel()
        y_true = np.concatenate(true_labels).ravel()

        return roc_auc_score(y_true, y_pred)

    def __call__(self, loader):
        return self.test(loader)


class Forwarder(Tester):
    def test(self, loader):
        self.model.eval()

        with torch.no_grad():
            for data, true_target, _ in loader:
                pass


class Comparator(Tester):
    def __init__(self, model, reference_model, use_cuda, losses,
                 memoize=True, reductor_factory=SumReductor):
        super().__init__(model, use_cuda, losses,
                         reductor_factory=reductor_factory)

        self.reference_model = Memoizer(reference_model) if memoize else \
            PseudoMemoizer(reference_model)
        self.reference_model.to(self.device)
        self.memoized_models = {}

    @property
    def memoize(self):
        return isinstance(self.reference_model, Memoizer)

    def _do_test(self, model, reference_model, loader):
        model.eval()
        reference_model.eval()

        losses_values = self.reductor_factory(len(self._losses), self.device)

        with torch.no_grad():
            for batch in loader:
                data, true_target = batch[0:2]
                index = None
                if len(batch) == 3:
                    # Got index for memoization
                    index = batch[-1]

                data = data.to(self.device)
                true_target = true_target.to(self.device)
                target = reference_model(data, index)
                output = model(data)
                for i, loss_f in enumerate(self._losses):
                    losses_values.add(i,
                                      loss_f(output,
                                             hard_target=true_target,
                                             soft_target=target,
                                             reduction=losses_values.reduction),
                                      data.size(0))
                if self.memoize:
                    # Releive the GPU
                    data.to("cpu")
                    true_target.to("cpu")

        return tuple(iter(losses_values))


    def test(self, loader):
        # Trick to get the good memoizer for the given loader

        reference_model = self.memoized_models.get(id(loader)) \
            if self.memoize else self.reference_model
        if reference_model is None:
            reference_model = Memoizer(self.reference_model)
            self.memoized_models[id(loader)] = reference_model
            reference_model.to(self.device)

        return self._do_test(self.model, reference_model, loader)





class OutputDistribution(Deviceable):
    def __init__(self, model, n_outputs, use_cuda):
        super().__init__(use_cuda)
        self.model = model
        model.to(self.device)
        self.n_outputs = n_outputs

    def compute(self, loader):
        self.model.eval()
        h = np.zeros(self.n_outputs)
        with torch.no_grad():
            for batch in loader:
                data, target = batch[0], batch[1]
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                y_pred = output.data.cpu().numpy().argmax(axis=1)

                for i in range(self.n_outputs):
                    h[i] += np.sum(y_pred == i)
        return h
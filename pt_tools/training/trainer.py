import copy

import torch
import torch.nn.functional as F

from .util import Deviceable
from ..inspector import Chrono, ProgressTracker, StreamingStat


class Trainer(Deviceable):
    def __init__(self, model, optimizer, use_cuda=False,
                 save_path=None, loss_fn=F.cross_entropy):
        super().__init__(use_cuda)
        self.optimizer = optimizer
        self.save_path = save_path
        self.model = model
        self.model.to(self.device)
        self.loss_fn = loss_fn

    def train_one_epoch(self, loader):
        self.model.train()

        loss_stats = StreamingStat()

        for batch_idx, batch in enumerate(loader):
            # Getting data on appropriate device
            data, target = batch[0], batch[1]
            data, target = data.to(self.device), target.to(self.device)

            # Optim loop
            self.optimizer.zero_grad()
            output = self.model(data)
            loss_variable = self.loss_fn(output, target, reduction="mean")
            loss_stats.add(loss_variable.cpu().data.numpy())  # Monitoring loss
            loss_variable.backward()
            self.optimizer.step()

        return loss_stats

    def train(self, train_loader, n_epochs):
        for epoch in Chrono(range(n_epochs), "Learning", update_rate=0.05):

            loss = self.train_one_epoch(ProgressTracker(train_loader,
                                                        "Epoch {}".format(
                                                            epoch + 1),
                                                        update_rate=0.1))

            yield loss

            if epoch % 5 == 0 and self.save_path is not None:
                self.save_model()

        self.save_model()

    def save_model(self, path=None):
        if path is None:
            path = self.save_path
        if path is not None:
            torch.save(self.model.state_dict(), path)


class TrainerDecorator(object):
    def __init__(self, trainer, callable=None):
        self._trainer = trainer
        if callable is None:
            callable = self._on_finish_epoch_callback
        self.callable = callable


    def _on_finish_epoch_callback(self, epoch, loss):
        pass


    def train(self, train_loader, n_epochs):
        for epoch, loss in enumerate(self._trainer.train(train_loader, n_epochs)):
            yield loss
            self.callable(epoch, loss)

    def train_one_epoch(self, loader):
        return self._trainer.train_one_epoch(loader)

    def save_model(self, path=None):
        self._trainer.save_model(path)

    def _get_base_trainer(self):
        decorated = self._trainer
        while isinstance(decorated, TrainerDecorator):
            decorated = decorated._trainer
        return decorated


class WithSaveBest(TrainerDecorator):
    def __init__(self, trainer, auto_save=False, reloads=None):
        super().__init__(trainer, callable=None)

        # Overriding save path so that it does not erase best model
        base_trainer = self._get_base_trainer()
        self.best_score = float("-inf")  # Higher the better
        self.best_epoch = -1
        self.best_save_path = base_trainer.save_path
        self.in_memory = None
        base_trainer.save_path = None

        self.reloads = set() if reloads is None else set(reloads)
        self.auto_save = auto_save

        self.save_best()

    @property
    def model(self):
        return self._get_base_trainer().model

    def save_best(self):
        if self.best_save_path is None:
            self.in_memory = copy.deepcopy(self.model.state_dict())
        else:
            self.save_model(self.best_save_path)

    def save_if_best(self, score, epoch):
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.save_best()
        return self.best_epoch

    def load_best(self):
        if self.best_save_path is None:
            state_dict = self.in_memory
        else:
            state_dict = torch.load(self.best_save_path)
        self.model.load_state_dict(state_dict)
        return self.model

    def _on_finish_epoch_callback(self, epoch, loss):
        if self.auto_save:
            self.save_if_best(-loss, epoch)

        if epoch in self.reloads:
            self.load_best()


class WithLRScheduler(TrainerDecorator):
    def __init__(self, trainer, scheduler):
        super().__init__(trainer, callable=lambda e,l : scheduler.step())


def make_trainer(model, optimizer, use_cuda=False,
                 save_path=None, loss_fn=F.cross_entropy,
                 save_best=False, scheduler=None):
    trainer = Trainer(model, optimizer, use_cuda, save_path, loss_fn)
    if save_best:
        trainer = WithSaveBest(trainer, auto_save=True)
    if scheduler:
        trainer = WithLRScheduler(trainer, scheduler)

    return trainer
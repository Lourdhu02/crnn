import os
import torch
import wandb
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.best_loss = float("inf")
        if config["wandb"]["project"]:
            wandb.init(project=config["wandb"]["project"], name=config.get("experiment_name"))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for images, labels, lengths in tqdm(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            preds = self.model(images)
            preds = preds.permute(1,0,2)

            input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(self.device)

            loss = self.loss_fn(preds, labels, input_lengths, lengths)

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config["train"]["grad_clip"])
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels, lengths in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                preds = self.model(images)
                preds = preds.permute(1,0,2)
                input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long).to(self.device)

                loss = self.loss_fn(preds, labels, input_lengths, lengths)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save(self, epoch, loss):
        os.makedirs(self.config["train"]["save_dir"], exist_ok=True)
        path = os.path.join(self.config["train"]["save_dir"], "best.pth")
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "loss": loss}, path)

    def fit(self):
        epochs = self.config["train"]["epochs"]
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            if self.scheduler:
                self.scheduler.step()

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save(epoch, val_loss)

            if self.config["wandb"]["project"]:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

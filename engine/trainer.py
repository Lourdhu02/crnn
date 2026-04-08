import os
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import editdistance


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

        self.base_dir = "outputs"
        self.ckpt_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.onnx_dir = os.path.join(self.base_dir, "onnx")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, "train_log.txt")

    def decode(self, preds):
        preds = preds.argmax(2)
        results = []
        for seq in preds:
            res = []
            prev = -1
            for p in seq:
                p = p.item()
                if p != 0 and p != prev:
                    res.append(p)
                prev = p
            results.append(res)
        return results

    def compute_metrics(self, preds, labels, lengths):
        decoded_preds = self.decode(preds)

        idx = 0
        correct = 0
        total = 0
        total_cer = 0

        for i in range(len(lengths)):
            length = lengths[i]
            target = labels[idx:idx + length].tolist()
            pred = decoded_preds[i]

            if pred == target:
                correct += 1

            total += 1
            total_cer += editdistance.eval(pred, target) / max(len(target), 1)

            idx += length

        acc = correct / total if total > 0 else 0
        cer = total_cer / total if total > 0 else 0

        return acc, cer

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for images, labels, lengths in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            preds = self.model(images)
            preds_perm = preds.permute(1, 0, 2)

            input_lengths = torch.full(
                (preds_perm.size(1),),
                preds_perm.size(0),
                dtype=torch.long,
                device=self.device
            )

            loss = self.loss_fn(preds_perm, labels, input_lengths, lengths)

            loss = loss * 0.9

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config["train"]["grad_clip"])
            self.optimizer.step()

            total_loss += loss.item()

            acc, cer = self.compute_metrics(preds.detach().cpu(), labels.cpu(), lengths.cpu())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.3f}",
                "cer": f"{cer:.3f}"
            })

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_cer = 0
        count = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        with torch.no_grad():
            for images, labels, lengths in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                preds = self.model(images)
                preds_perm = preds.permute(1, 0, 2)

                input_lengths = torch.full(
                    (preds_perm.size(1),),
                    preds_perm.size(0),
                    dtype=torch.long,
                    device=self.device
                )

                loss = self.loss_fn(preds_perm, labels, input_lengths, lengths)
                total_loss += loss.item()

                acc, cer = self.compute_metrics(preds.cpu(), labels.cpu(), lengths.cpu())

                total_acc += acc
                total_cer += cer
                count += 1

                pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.3f}",
                    "cer": f"{cer:.3f}"
                })

        return total_loss / len(self.val_loader), total_acc / count, total_cer / count

    def save(self, epoch, loss):
        best_path = os.path.join(self.ckpt_dir, "best.pth")
        last_path = os.path.join(self.ckpt_dir, "last.pth")

        torch.save({"model": self.model.state_dict(), "epoch": epoch, "loss": loss}, last_path)

        if loss < self.best_loss:
            self.best_loss = loss
            torch.save({"model": self.model.state_dict(), "epoch": epoch, "loss": loss}, best_path)

    def export_onnx(self):
        path = os.path.join(self.onnx_dir, "model.onnx")
        dummy = torch.randn(1,1,self.config["img_height"],self.config["img_width"]).to(self.device)
        torch.onnx.export(self.model, dummy, path, opset_version=11)

    def log(self, text):
        with open(self.log_file, "a") as f:
            f.write(text + "\n")

    def fit(self):
        epochs = self.config["train"]["epochs"]

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc, val_cer = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            self.save(epoch, val_loss)

            msg = f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f} | val_cer: {val_cer:.3f}"
            print(msg)
            self.log(msg)

        self.export_onnx()

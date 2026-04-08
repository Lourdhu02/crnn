import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, decoder, loader, device):
        self.model = model.to(device)
        self.decoder = decoder
        self.loader = loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, lengths in tqdm(self.loader):
                images = images.to(self.device)
                preds = self.model(images)
                preds = preds.argmax(2).cpu().numpy()

                idx = 0
                for i in range(len(lengths)):
                    length = lengths[i]
                    target = labels[idx:idx+length].tolist()
                    pred = self.decoder.decode(preds[i])
                    target_str = self.decoder.decode(target)
                    if pred == target_str:
                        correct += 1
                    total += 1
                    idx += length
        return correct / total if total > 0 else 0

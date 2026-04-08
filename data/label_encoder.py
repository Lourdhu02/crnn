class LabelEncoder:
    def __init__(self, charset):
        self.charset = charset
        self.char2idx = {c:i+1 for i,c in enumerate(charset)}
        self.idx2char = {i+1:c for i,c in enumerate(charset)}
        self.blank = 0

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, preds):
        res = []
        prev = None
        for p in preds:
            if p != self.blank and p != prev:
                res.append(self.idx2char.get(p, ""))
            prev = p
        return "".join(res)

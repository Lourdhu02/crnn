import numpy as np


class CTCDecoder:
    def __init__(self, encoder, beam_size=0):
        self.encoder = encoder
        self.beam_size = beam_size

    def greedy(self, preds):
        seq = np.argmax(preds, axis=1)

        cleaned = []
        prev = None
        for c in seq:
            if c != 0 and c != prev:
                cleaned.append(int(c))
            prev = c

        return self.encoder.decode(cleaned)

    def beam_search(self, probs):
        T, C = probs.shape

        beams = [([], 0.0)]

        for t in range(T):
            new_beams = []

            for seq, score in beams:
                for c in range(C):
                    new_seq = seq.copy()

                    if len(new_seq) == 0 or c != new_seq[-1]:
                        new_seq.append(c)

                    new_score = score + np.log(probs[t, c] + 1e-8)
                    new_beams.append((new_seq, new_score))

            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_size]

        best_seq = beams[0][0]

        cleaned = []
        prev = None
        for c in best_seq:
            if c != 0 and c != prev:
                cleaned.append(int(c))
            prev = c

        return self.encoder.decode(cleaned)

    def decode(self, preds, return_conf=False):
        if isinstance(preds, list):
            return self.encoder.decode(preds)

        if len(preds.shape) == 1:
            return self.encoder.decode(preds.tolist())

        probs = np.exp(preds)

        if self.beam_size > 1:
            text = self.beam_search(probs)
            conf = float(np.max(probs))
        else:
            seq = np.argmax(probs, axis=1)

            cleaned = []
            conf_scores = []
            prev = None

            for t, c in enumerate(seq):
                if c != 0 and c != prev:
                    cleaned.append(int(c))
                    conf_scores.append(probs[t, c])
                prev = c

            text = self.encoder.decode(cleaned)
            conf = float(np.mean(conf_scores)) if len(conf_scores) > 0 else 0.0

        if return_conf:
            return text, conf

        return text
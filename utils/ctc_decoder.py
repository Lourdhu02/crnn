import numpy as np


class CTCDecoder:
    def __init__(self, encoder, beam_size=0):
        self.encoder = encoder
        self.beam_size = beam_size

    def greedy(self, preds):
        probs = np.exp(preds)
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
        conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        return text, conf

    def beam_search(self, probs):
        T, C = probs.shape
        beams = [([], 0.0)]

        for t in range(T):
            new_beams = {}
            for seq, score in beams:
                for c in range(C):
                    if c == 0:
                        key = tuple(seq)
                        new_score = score + np.log(probs[t, c] + 1e-8)
                        if key not in new_beams or new_beams[key] < new_score:
                            new_beams[key] = new_score
                        continue
                    new_seq = seq + [c] if (len(seq) == 0 or c != seq[-1]) else seq
                    key = tuple(new_seq)
                    new_score = score + np.log(probs[t, c] + 1e-8)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
            beams = [(list(k), v) for k, v in beams[: self.beam_size]]

        best_seq = beams[0][0] if beams else []
        cleaned = [c for c in best_seq if c != 0]
        text = self.encoder.decode(cleaned)

        # geometric mean confidence from greedy path as proxy
        seq_greedy = np.argmax(probs, axis=1)
        conf_scores = []
        prev = None
        for t, c in enumerate(seq_greedy):
            if c != 0 and c != prev:
                conf_scores.append(probs[t, c])
            prev = c
        conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        return text, conf

    def decode(self, preds, return_conf=False):
        if isinstance(preds, list):
            text = self.encoder.decode(preds)
            return (text, 1.0) if return_conf else text

        if len(preds.shape) == 1:
            text = self.encoder.decode(preds.tolist())
            return (text, 1.0) if return_conf else text

        probs = np.exp(preds)

        if self.beam_size > 1:
            text, conf = self.beam_search(probs)
        else:
            text, conf = self.greedy(preds)

        return (text, conf) if return_conf else text

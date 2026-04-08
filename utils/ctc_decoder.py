class CTCDecoder:
    def __init__(self, encoder, beam_size=0):
        self.encoder = encoder
        self.beam_size = beam_size

    def decode(self, preds):
        if isinstance(preds, list):
            return self.encoder.decode(preds)

        if len(preds.shape) == 1:
            return self.encoder.decode(preds.tolist())

        seq = preds.argmax(axis=1)
        return self.encoder.decode(seq.tolist())
from data.label_encoder import LabelEncoder
from utils.ctc_decoder import CTCDecoder

def test_decoder():
    encoder = LabelEncoder("0123456789.")
    decoder = CTCDecoder(encoder)
    seq = [1,1,2,0,2,3]
    text = decoder.decode(seq)
    assert isinstance(text, str)

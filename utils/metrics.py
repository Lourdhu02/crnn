import editdistance

def cer(pred, target):
    return editdistance.eval(pred, target) / max(len(target), 1)

def accuracy(pred, target):
    return 1.0 if pred == target else 0.0

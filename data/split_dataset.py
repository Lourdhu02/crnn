import os
import random

def split(label_file, train_out, val_out, test_out, ratios=(0.8,0.1,0.1)):
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    n = len(lines)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = lines[:n_train]
    val = lines[n_train:n_train+n_val]
    test = lines[n_train+n_val:]

    with open(train_out, "w") as f:
        f.writelines(train)
    with open(val_out, "w") as f:
        f.writelines(val)
    with open(test_out, "w") as f:
        f.writelines(test)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--train")
    p.add_argument("--val")
    p.add_argument("--test")
    args = p.parse_args()
    split(args.input, args.train, args.val, args.test)

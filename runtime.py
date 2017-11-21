from trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.00005, type=float)
parser.add_argument("--diter", default=5, type=int)
parser.add_argument("-cls", default=False, action='store_true')
args = parser.parse_args()

trainer = Trainer(type=args.type, lr= args.lr, diter=args.diter)
trainer.train(args.cls)



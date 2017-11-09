from trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-vanilla", default=False, action='store_true')
args = parser.parse_args()
trainer = Trainer()

if args.vanilla:
    trainer.train_gan()
else:
    trainer.train_gan_cls()

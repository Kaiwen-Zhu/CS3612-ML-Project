import argparse
import os


def make_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'))
    arg_parser.add_argument('--batch_size', default=64, type=int)
    arg_parser.add_argument('--epochs', default=50, type=int)
    arg_parser.add_argument('--lr', default=1e-3, type=float)
    arg_parser.add_argument('--weight_decay', default=1e-4, type=float)

    return arg_parser.parse_args()

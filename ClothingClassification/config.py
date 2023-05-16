import argparse
import os


def make_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'))
    arg_parser.add_argument('--batch_size', default=64, type=int)

    return arg_parser.parse_args()

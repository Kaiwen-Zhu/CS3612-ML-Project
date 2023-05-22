import argparse
import os


def make_train_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'), type=str)
    arg_parser.add_argument('--save_root', default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'checkpoints'), type=str)
    arg_parser.add_argument('--batch_size', default=64, type=int)
    arg_parser.add_argument('--epochs', default=100, type=int)
    arg_parser.add_argument('--lr', default=1e-3, type=float)
    arg_parser.add_argument('--weight_decay', default=1e-4, type=float)
    arg_parser.add_argument('--num_res_blocks', default=3, type=int)
    arg_parser.add_argument('--num_channel_1', default=32, type=int)
    arg_parser.add_argument('--num_channel_2', default=64, type=int)
    arg_parser.add_argument('--hidden_dim_fc', default=128, type=int)

    return arg_parser.parse_args()


def make_visualization_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'))
    # save_root/model_dir should be the path to the directory containing 
    # the results to be visualized
    arg_parser.add_argument('--save_root', default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'checkpoints'))
    arg_parser.add_argument('--model_dir', default='best', type=str)

    return arg_parser.parse_args()

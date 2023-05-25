import argparse
import os


def make_train_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'))
    arg_parser.add_argument('--save_root', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'checkpoints'))
    arg_parser.add_argument('--batch_size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--lr', type=float, default=1e-3)
    arg_parser.add_argument('--weight_decay', type=float, default=1e-4)
    arg_parser.add_argument('--code_dim', type=int, default=256)
    arg_parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16, 32, 64, 64, 128])

    return arg_parser.parse_args()


def make_visualization_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'))
    # save_root/model_dir should be the path to the directory containing 
    # the results to be visualized
    arg_parser.add_argument('--save_root', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'checkpoints'))
    arg_parser.add_argument('--model_dir', type=str, default='best')
    arg_parser.add_argument('--interpolate_component', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'inter_comp'),
        help="path to the directory containing the interpolation components")

    return arg_parser.parse_args()

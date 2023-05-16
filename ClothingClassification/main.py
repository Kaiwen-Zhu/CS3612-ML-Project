from config import make_parser
from data_loader import get_dataloader


args = make_parser()

train_loader, test_loader = get_dataloader(args.data_root, args.batch_size)

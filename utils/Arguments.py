import argparse

import torch


class Arguments:
    def __init__(self):
        args = self.parse_args()

        self.seed: int = args.seed

        self.model_path: str | None = args.model

        self.report = args.report

        self.dataset: str = args.dataset
        self.dtype = torch.float32 if args.dtype == 'float32' or args.dtype == 'float' else torch.float64
        self.device = args.device

        self.batch_size: int = args.batch_size
        self.epoch: int = args.epoch

        self.slide_window: int = args.slide_window
        self.slide_stride: int = args.slide_stride
        self.k: int = args.k

        self.d_hidden: int = args.d_hidden
        self.d_output_hidden: int = args.d_output_hidden

        self.num_output_layer: int = args.num_output_layer

        self.lr: float = args.lr

        self.early_stop: int = args.early_stop

        self.log: bool = not args.nolog

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--model', type=str)

        parser.add_argument('--report', type=str, choices=['label', 'no_label'], default='no_label')

        parser.add_argument('-ds', '--dataset', type=str, default='swat')
        parser.add_argument('--dtype', choices=['float', 'float32', 'double', 'float64'], default='float')
        parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

        parser.add_argument('-b', '--batch_size', type=int, default=32)
        parser.add_argument('-e', '--epoch', type=int, default=30)

        parser.add_argument('-sw', '--slide_window', type=int, default=5)
        parser.add_argument('-ss', '--slide_stride', type=int, default=1)
        parser.add_argument('-k', '--k', type=int, default=5)

        parser.add_argument('--d_hidden', type=int, default=64)
        parser.add_argument('--d_output_hidden', type=int, default=128)

        parser.add_argument('--num_output_layer', type=int, default=1)

        parser.add_argument('--lr', type=float, default=0.001)

        parser.add_argument('--early_stop', type=int, default=20)

        parser.add_argument('--nolog', action='store_true')

        return parser.parse_args()

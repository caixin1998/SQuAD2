import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SQuAD train & test options')
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='SQuAD', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='bert', help='model of the experiment.(bert or xlnet)')

        self.parser.add_argument('--devices', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--doload', default='', type=str, required=False, help='if load ')
        self.parser.add_argument('--checkpoint_path', type=str, default='.', help='models are saved here')
        self.parser.add_argument( '--sink',  type=bool,  default=False, help='if load model from last checkpoint')
        self.parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model')
        
        
        self.parser.add_argument("--batch_size", type=int, dest="batch_size", default=64, help="Mini-batch size")
        self.parser.add_argument("--lr", type=float, dest="lr", default=5e-5, help="Base Learning Rate")
        self.parser.add_argument("--epochs", type=int, dest="epochs", default=100, help="Number of iterations")
        self.parser.add_argument('--visualize', type=bool, default=True, help='if open tensorboardX')
        self.parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
        self.parser.add_argument('--max_grad_norm', default=1, type=float, required=False, help='max grad norm')
        self.parser.add_argument('--num_workers', default=8, type=int, required=False)
        
       
    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
import argparse
import utils.paths as paths

class Arguments:

    def __init__(self, weights=paths.weights, input='data/demo/oneword.png', output='data/res/oneword_res.png', cfg='D'):
        self.cfg = cfg
        self.weights = weights
        self.input = input
        self.output = output

    def initialzer_args(self):
        parser = argparse.ArgumentParser(description="EAST inference arguments")
        args = parser.parse_args()
        args.cfg = self.cfg
        args.input = self.input
        args.weights = self.weights
        args.output = self.output
        return args

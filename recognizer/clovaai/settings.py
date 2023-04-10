import argparse
import utils.paths as paths


class Arguments:

    def __init__(self):
        self.args = [paths.crop_folder, 25, 224, 224, False, '0123456789abcdefghijklmnopqrstuvwxyz', 1,
                     paths.model,
                     False, False, False, False]

    def initialize_args(self):
        parser = argparse.ArgumentParser(description='ViTSTR evaluation')
        args = parser.parse_args()
        args.image = self.args[0]
        args.batch_max_length = self.args[1]
        args.imgH = self.args[2]
        args.imgW = self.args[3]
        args.rgb = self.args[4]
        args.character = self.args[5]
        args.input_channel = self.args[6]
        args.model = self.args[7]
        args.gpu = self.args[8]
        args.time = self.args[9]
        args.quantized = self.args[10]
        args.rpi = self.args[11]
        return args

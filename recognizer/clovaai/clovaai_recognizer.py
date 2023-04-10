import argparse


class ClovaAIRecognizer(object):
    def __init__(self, save_crop_images_folder, model_path, batch_size=25, img_h=224, img_w=224, rgb=False, gpu=False):
        self.save_crop_folder = save_crop_images_folder
        self.batch_size = batch_size
        self.input_channel = 1
        self.model_path = model_path
        self.img_h = img_h
        self.img_w = img_w
        self.rgb = rgb
        self.gpu = gpu
        self.alpha_character = "0123456789abcdefghijklmnopqrstuvwxyz"
        # args = parser.parse_args()
        # args.image = self.save_crop_folder
        # args.batch_max_length = self.batch_size
        # args.imgH = self.img_h
        # args.imgW = self.img_w
        # args.rgb = self.rgb
        # args.character = self.alpha_character
        # args.input_channel = self.input_channel
        # args.model = self.model_path
        # args.gpu = self.gpu
        # args.time = False
        # args.quantized = False
        # args.rpi = False
        # return args

    def initialize_args(self):
        parser = argparse.ArgumentParser(description='ViTSTR evaluation')
        args = parser.parse_args()
        args.image = self.save_crop_folder
        args.batch_max_length = self.batch_size
        args.imgH = self.img_h
        args.imgW = self.img_w
        args.rgb = self.rgb
        args.character = self.alpha_character
        args.input_channel = self.input_channel
        args.model = self.model_path
        args.gpu = self.gpu
        args.time = False
        args.quantized = False
        args.rpi = False
        return args

import sys

from detector.OpenVINO.detect import TextDetector
from detector.east import detect as EastDetector
from recognizer.clovaai.clovaai_recognizer import ClovaAIRecognizer
from recognizer.trocr.main import TROCRPredictor
import recognizer.clovaai.infer as clovainfer
from utils import paths
from utils.imutil import word_sequencer, cropper, load_images, empty_folder
import string


# Optical Character Recognition
class TextRecognizerService:
    def __init__(self, save_crop_images_folder=None, model_path=None):
        self.save_crop_images_folder = save_crop_images_folder
        if self.save_crop_images_folder is None:
            self.save_crop_images_folder = paths.crop_folder
        self.openvino_detector = TextDetector()
        self.east_detector = EastDetector
        self.tr_ocr_recognizer = TROCRPredictor()
        if model_path is None:
            model_path = paths.model
        self.clova_recognizer = ClovaAIRecognizer(self.save_crop_images_folder, model_path)

    def load_detector(self, image_path, is_east=False, is_openvino=False):
        if is_east and is_openvino:
            print("Please select one model for detection")
            sys.exit(1)
        elif not is_east and not is_openvino:
            print("Please select one model for detection")
            sys.exit(1)
        elif is_east:
            bboxes = self.detect_east(image_path)
        else:
            bboxes = self.detect_txt(image_path)
        return bboxes

    def load_recognizer(self, crops_img_path, is_tr_ocr=False, is_clova_ocr=False):
        if is_tr_ocr and is_clova_ocr:
            print("Please select one model for recognition")
            sys.exit(1)
        elif not is_clova_ocr and not is_tr_ocr:
            print("Please select one model for recognition")
            sys.exit(1)
        elif is_tr_ocr:
            recognized_word = self.tr_ocr_recognize(path=crops_img_path)
        else:
            recognized_word = self.clova_recognize(path=crops_img_path)
        return recognized_word

    def detect_txt(self, image_path):
        print("*****************[Using OPENVINO]*****************")
        result = self.openvino_detector.detect_text(image_path)
        return result

    def detect_east(self, image_path):
        print("*****************[Using EAST DETECTION]*****************")
        args = self.east_detector.parse_opt()
        args.input = image_path
        model, device = self.east_detector.load_model(args)
        bboxes = self.east_detector.detector(args, model, device)
        return bboxes

    def detect_and_recognize_txt(self, image_path):
        word = None
        try:
            bboxes = self.load_detector(image_path, is_east=True, is_openvino=False)
            if not bboxes:
                print("No texts detected.")
                sys.exit(1)
            crops_img_path = self.crop_handler(bboxes, self.save_crop_images_folder, image_path)
            # If no text is detected
            word = self.load_recognizer(is_clova_ocr=True, crops_img_path=crops_img_path)
            self.remove_cache(self.save_crop_images_folder)
        except Exception as e:
            print("Warning:", e)
        return word

    def crop_handler(self, bboxes, crop_path, original_image):
        bboxes = word_sequencer(bboxes)
        status = cropper(bboxes, original_image)
        if not status:
            print("Could not write all the images into the directory.")
            sys.exit(1)
        path = load_images(crop_path)
        return path

    def tr_ocr_recognize(self, path):
        print("*****************[Using TRANSFORMER OCR]*****************")
        print("Recognizing texts...")
        word = " "
        for images in path:
            predictor, confidence = self.tr_ocr_recognizer.predict_images([images])
            word = word + " " + predictor[0][1]
        return word

    def clova_recognize(self, path):
        print("*****************[Using CLOVA AI]*****************")
        print("Recognizing texts...")
        word = " "
        args = self.clova_recognizer.initialize_args()
        args.character = string.printable[:-6]
        for img in path:
            args.image = img
            data = clovainfer.infer(args)
            for filename, text in data:
                word = word + " " + text
        return word

    @staticmethod
    def remove_cache(crop_path):
        empty_folder(crop_path)

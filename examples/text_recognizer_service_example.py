import time
import cv2
from service.text_recognizer_service import TextRecognizerService
from utils.paths import image as path_to_image
from utils.text_wrapper import wrap


def txt_recogn_service_example():
    # Initializing recording time
    image_path = path_to_image
    st = time.time()
    text_recog_service = TextRecognizerService()
    recognize_words = text_recog_service.detect_and_recognize_txt(image_path)
    if recognize_words is not None:
        recognize_words = wrap(recognize_words, 40)
    print(f"Output:\n{recognize_words}")
    et = time.time()
    elapsed_time = et - st
    print('\nExecution time: {:.2f} seconds'.format(elapsed_time))


def txt_detect_service_example():
    # Initializing recording time
    image_path = path_to_image
    st = time.time()
    image = cv2.imread(image_path)
    text_recog_service = TextRecognizerService()
    list_bboxes = text_recog_service.detect_txt(image_path)
    if list_bboxes:
        for bbox in list_bboxes:
            t, l, b, r = bbox
            cv2.rectangle(image, (t, l), (b, r), (0, 255, 0), 2)
    cv2.namedWindow("Text detected", cv2.WINDOW_NORMAL)
    cv2.imshow("Text detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    et = time.time()
    elapsed_time = et - st
    print('\nExecution time: {:.2f} seconds'.format(elapsed_time))


if __name__ == '__main__':
    txt_recogn_service_example()
# txt_detect_service_example()

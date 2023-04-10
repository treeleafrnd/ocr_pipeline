import os
import torch
import string
import validators
import time
from recognizer.clovaai.infer_utils import TokenLabelConverter, NormalizePAD, ViTSTRFeatureExtractor
# from infer_utils import get_args
from recognizer.clovaai.settings import Arguments


def img2text(model, images, converter):
    pred_strs = []
    with torch.no_grad():
        for img in images:
            model = torch.nn.DataParallel(model, device_ids=[0])
            # model = model.toDevice('cpu')
            pred = model(img, seqlen=converter.batch_max_length)
            _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
            pred_index = pred_index.view(-1, converter.batch_max_length)
            length_for_pred = torch.IntTensor([converter.batch_max_length - 1])
            pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
            pred_EOS = pred_str[0].find('[s]')
            pred_str = pred_str[0][:pred_EOS]

            pred_strs.append(pred_str)

    return pred_strs


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    converter = TokenLabelConverter(args)
    args.num_class = len(converter.character)
    extractor = ViTSTRFeatureExtractor()
    if args.time:
        files = ["demo_1.png", "demo_2.jpg", "demo_3.png", "demo_4.png", "demo_5.png", "demo_6.png", "demo_7.png",
                 "demo_8.jpg", "demo_9.jpg", "demo_10.jpg"]
        images = []
        for f in files:
            f = os.path.join("demo_image", f)
            img = extractor(f)
            if args.gpu:
                img = img.to(device)
            images.append(img)
    else:
        assert (args.image is not None)
        files = [args.image]
        img = extractor(args.image)
        if args.gpu:
            img = img.to(device)
        images = [img]

    if args.quantized:
        if args.rpi:
            backend = "qnnpack"  # arm
        else:
            backend = "fbgemm"  # x86

        torch.backends.quantized.engine = backend

    if validators.url(args.model):
        checkpoint = args.model.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.model, checkpoint)
    else:
        checkpoint = args.model

    if args.quantized:
        model = torch.jit.load(checkpoint)
    else:
        model = torch.load(checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if args.gpu:
        model.to(device)

    model.eval()

    if args.time:
        n_times = 10
        n_total = len(images) * n_times
        [img2text(model, images, converter) for _ in range(n_times)]
        start_time = time.time()
        [img2text(model, images, converter) for _ in range(n_times)]
        end_time = time.time()
        ave_time = (end_time - start_time) / n_total
        print("Average inference time per image: %0.2e sec" % ave_time)
    pred_strs = img2text(model, images, converter)

    return zip(files, pred_strs)

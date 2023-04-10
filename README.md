# Optical Character Recognition
A complete optical character recognition tool (OCR) using OpenVINO for detector, and CLOVA AI and TransformerOCR for recognition.

# OpenVINO
OpenVINO is an open-source toolkit for optimizing and deploying deep learning models. It provides boosted deep learning performance for vision, audio, and language models from popular frameworks like TensorFlow, PyTorch, and more. OpenVINO enables you to optimize a deep learning model from almost any framework and deploy it with best-in-class performance on a range of Intel processors and other hardware platforms.

![openvino](https://user-images.githubusercontent.com/99968233/230010403-05898cdc-7117-4468-a0a7-f52c3c896238.png)

# TransformerOCR
A TrOCR consists of encoder and a decoder where a encoder uses a pretrained model such as Vision image Transformer (ViT), BERT image Transformer (BEiT), and Dual Intent and Entity Transformer (DiET) and decoder uses a pretrained models such as BERT, and RoBERTa. This model can be trained in printed, handwritten or scene text dataset. An self attention model is required to overcome vanishing problem in LSTM. An attention model simply predicts the next word in a sequence, thus making OCR more accurate. The model is trained and fine tuned in IAM handwriting dataset.

![trocr_architecture](https://user-images.githubusercontent.com/99968233/230011422-bffc6d11-3e82-4877-8b76-ab82c6dbf19e.jpg)

# CLOVAAI (deep-text-recognition)
This is a scene text recognition model that implements layers of models such as Residual Network (ResNet), Bidirectional Long Short Term Memory (BiLSTM), Thin-plate spline (TPS) and Connectionist Temporal Classification (CTC) that is optimized to recognize the texts which are curved regardless of their size in the scene. It performs exceptionally better in recognizing curved texts due to the use of TPS. 

<img width="770" alt="vitstr_model" src="https://user-images.githubusercontent.com/99968233/230010449-fd7c9208-2f0a-43a0-a4d5-dca65769f34c.png">

# Installation:
  ```
  git clone https://github.com/treeleafrnd/ocr_pipeline.git
  ```
# Dependencies:
  ```
  pip install -r requirements.txt
  ```
  
 # Run:
  ```
  examples/text_recognizer_service_example.py
  ```
  # Example:
  
  ![long](https://user-images.githubusercontent.com/99968233/230013294-7cbe85d8-a7ff-469c-8ca0-907208e13881.png)
  
  # Results:
  
  ![results](https://user-images.githubusercontent.com/99968233/230010434-1030314b-1234-4488-b6eb-c6fb223d1377.png)
  
  

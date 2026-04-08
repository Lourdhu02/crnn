OCR CRNN

End-to-end deep learning pipeline for reading numeric meter values from images using TPS rectification + CNN + BiLSTM + CTC.

## Features

- TPS Spatial Transformer for distortion correction
- MobileNetV3 / ResNet backbone
- BiLSTM sequence modeling
- CTC loss with decoding
- Config-driven training (YAML)
- Albumentations preprocessing
- ONNX export ready

## Architecture

Input Image  
→ Preprocessing  
→ TPS Rectifier  
→ CNN Backbone  
→ BiLSTM  
→ CTC Head  
→ Output Sequence

## Installation

git clone https://github.com/Lourdhu02/crnn.git

cd crnn  
pip install -r requirements.txt  
pip install -e .

## Dataset Format

data/  
 train/  
 val/  
 test/  
 train_labels.txt  
 val_labels.txt  
 test_labels.txt

Label format:

image.jpg 123.4

## Training

python tools/train.py -c configs/crnn_mobilenet_ctc.yaml

## Evaluation

python tools/evaluate.py -c configs/crnn_mobilenet_ctc.yaml

## Inference

python tools/infer.py -c configs/crnn_mobilenet_ctc.yaml --image_path path_to_image_or_folder

## Colab Training

!git clone https://github.com/<your-username>/meter-ocr-crnn.git  
%cd meter-ocr-crnn  
!pip install -r requirements.txt  
!pip install -e .

from tools.train import main  
main("configs/crnn_mobilenet_ctc.yaml")

## Notes

- Use grayscale images
- Recommended dataset size: 5k+
- Train 30–100 epochs for good accuracy
- Use GPU for real training

## License

MIT

train:
python tools/train.py -c configs/crnn_mobilenet_ctc.yaml

eval:
python tools/evaluate.py -c configs/crnn_mobilenet_ctc.yaml

infer:
python tools/infer.py -c configs/crnn_mobilenet_ctc.yaml --image_path $(IMAGE)

export:
python tools/export_onnx.py -c configs/crnn_mobilenet_ctc.yaml

test:
python -m pytest tests/ -v

install:
pip install -e .

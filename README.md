This is the source code for the paper, "EMDA-YOLO: An Improved YOLOv8 Model for Brain Tumor Detection".

## Training

```
python train.py --yaml yolov8.yaml --data /dataset/data.yaml --workers 8 --batch 16 --epoch 200
```
```
python train.py --yaml EMDA-YOLO.yaml --data /dataset/data.yaml --workers 8 --batch 16 --epoch 200
```

## Testing

```
python val.py --weight runs/train/exp0/weights/best.pt --data /dataset/data.yaml --split test
```


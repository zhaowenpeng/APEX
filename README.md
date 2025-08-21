# AECE: An Annotation-Free Framework for Large-Scale Agricultural Parcel Extraction via Foundation Models and Adaptive Noise Correction from VHR Imagery

## Study Area

![1755801782916](image/README/1755801782916.jpg)

## Workflow

![1755801849297](image/README/1755801849297.jpg)

## Run

### Train_WeakSupervise

```bash
python Train_WeakSupervise.py \
    --train-image-folder "" \
    --train-label-folder "" \
    --valid-image-folder "" \
    --valid-label-folder "" \
    --save-dir "" \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-epochs 100 \
    --warmup-epochs 5 \
    --correct-epochs 10 \
    --monitor-metric "val_iou" \
    --mixed-precision false
```

### Train_FullSupervise

```bash
python Train_FullSupervise.py \
    --train-image-folder "" \
    --train-label-folder "" \
    --valid-image-folder "" \
    --valid-label-folder "" \
    --save-dir "" \
    --model "ablation" \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-epochs 100 \
    --monitor-metric "val_iou" \
    --mixed-precision true
```

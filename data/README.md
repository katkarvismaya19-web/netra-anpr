# Data folder

Nothing in this folder is committed to git except this file. The scripts fill it in:

| Folder | Created by | Contents |
|---|---|---|
| `raw/` | `scripts/download_dataset.py` | Original Kaggle downloads |
| `yolo/` | `scripts/prepare_dataset.py` | Merged detector dataset in YOLO format + `data.yaml` |
| `ocr/synthetic/` | `scripts/generate_plates.py` | Rendered plates with `labels.csv` |
| `ocr/real_unlabeled/` | `prepare_dataset.py --export-crops` | Real plate crops cut from the Kaggle images |

## Source datasets

- [Indian vehicle number plate YOLO annotation](https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation) — Indian vehicles, YOLO boxes
- [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) — 433 images, Pascal VOC boxes

Please respect each dataset's licence on Kaggle when publishing results.

## Improving OCR with real plates (optional)

The character reader is trained on synthetic plates. Adding even 200–300 real, labelled
crops usually gives a clear jump in accuracy on photographs.

1. Run `python scripts/prepare_dataset.py --export-crops`.
2. Move the clearest crops into `data/ocr/real/train/` and about 20 % of them into `data/ocr/real/val/`.
3. In each folder create `labels.csv`:

   ```
   filename,text
   indian-vehic_00012_0.png,MH12DE1433
   indian-vehic_00031_0.png,KA05MN9999
   ```

   Write the plate without spaces. Leave out crops you cannot read yourself.
4. Run `python scripts/train_ocr.py` again. Real crops are picked up automatically and oversampled ×5.

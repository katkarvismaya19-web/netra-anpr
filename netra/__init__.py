"""
Netra ANPR
==========

Automatic Number Plate Recognition for Indian vehicles.

The system is a two-stage deep learning pipeline:

1. **Detection**   - a YOLOv8 model (fine-tuned from COCO weights) finds the
   bounding box of every number plate in the image.
2. **Recognition** - each plate crop is read by a CRNN (CNN feature extractor
   + bidirectional LSTM + CTC decoding) trained on rendered Indian plates.

A rule-based post-processor then fixes common OCR confusions (O/0, I/1, B/8)
using the official Indian registration format and maps the first two letters
to the issuing state or union territory.

Typical use::

    from netra.pipeline import ANPRPipeline
    pipeline = ANPRPipeline()
    result = pipeline.run(image_bgr)
"""

__version__ = "1.0.0"

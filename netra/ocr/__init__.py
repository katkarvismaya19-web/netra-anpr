"""
netra.ocr
=========

Everything related to reading the characters on a plate:

* ``charset``    - alphabet and CTC encoding / decoding
* ``model``      - the CRNN network (CNN + BiLSTM)
* ``synth``      - synthetic Indian plate renderer for training data
* ``dataset``    - PyTorch dataset with realistic augmentations
* ``recognizer`` - inference wrapper (CRNN or EasyOCR baseline)
"""

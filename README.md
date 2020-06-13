# Temporal attention networks for no-reference  video quality assessment
Using attention to extend image quality assessment algorithms for the task of video quality assessment.

## Two-stage learning

Attention is used over features extracted by SOTA no-reference IQA algorithms in attention.py.

## End-to-end learning

Paq-2-Piq (an IQA model that uses deep-learning) is paired with an RNN and attention (CNN+RNN framework) to learn VQA scores end-to-end in vqa.py.


ann.py trains feature vectors that are averaged across frames.
attention.py trains an attention network to weigh frame features before training.
vqa.py performs end-to-end training with attention and Paq-2-Piq.


Please cite this repository if you use this in your work.

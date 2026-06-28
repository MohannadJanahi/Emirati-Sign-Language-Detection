# Emirati Sign Language Detection and Translation

Detecting and classifying Emirati Sign Language (ESL) gestures from video, using two different approaches: **Dynamic Time Warping (DTW)** on hand-joint coordinates and **Deep Learning** on motion-summarized frames, built and compared on a custom-collected dataset.

## Problem

ESL is understood by only a small fraction of the UAE population, despite an official dictionary of up to 5,000 signs having existed since 2018. This creates a real communication barrier for the deaf, hard-of-hearing, and mute community. Unlike many sign languages, ESL is highly varied, as some signs rely on hand shape, some on movement, some on both, and some use one or two hands, which makes it a harder classification problem than typical sign language datasets in the literature.

This project investigates which method (geometric time-series matching (DTW) or learned image classification (CNNs)) is better suited to this kind of variable, motion-heavy sign language, using a small, self-collected dataset as a real-world-sized test case.

## Dataset

Self-collected and labeled by me, since no public ESL video dataset exists:
- **5 classes**: hundred, mother, ninety-two, relax, salute
- **166 video samples per class**
- ~3 seconds per video at 30 fps (≤90 frames per sample)

## Approach 1 — Dynamic Time Warping (DTW) + MediaPipe

1. Run Google MediaPipe's holistic hand-tracking model on every video to extract (x, y) coordinates for 21 joints per hand, per frame.
2. Store each video as a sequence of joint-coordinate arrays in a dataframe.
3. For a new input video, extract its joint sequence the same way, then compute DTW distance against every entry in the dataframe.
4. The lowest-distance match is the predicted sign; a distance threshold is used to reject low-confidence predictions instead of forcing a guess.

**Why DTW**: it naturally handles time-series of mismatched length (people sign at different speeds) and works well even with very few examples per class.

**Limitation**: DTW is O(N²), and comparing against the full dataframe took ~3 seconds for 558 rows across 5 classes. That doesn't scale to a real ESL dictionary (5,000+ signs, potentially over a million reference videos). It's a viable offline/non-real-time method, but not a live-translation one.

## Approach 2 — Deep Learning + Difference of Frames

1. Convert each video into a single motion-summary image using the **difference of frames** algorithm: consecutive frames are subtracted from each other and accumulated, collapsing a video into one frame that encodes all the motion in it. The reference pseudocode was optimized by replacing a nested pixel-loop with NumPy's `where()`, cutting processing time per video from minutes to under 5 seconds. An example of the output of a difference of frames algorithm is shown below.

<img width="216" height="159" alt="318206868-fbe818ac-9c01-40af-8a30-49b628f28cde" src="https://github.com/user-attachments/assets/f9275c22-ae99-4418-9c66-08d04d288cef" />

2. Feed the resulting motion images into a CNN classifier.
3. Compare multiple architectures to find the best-performing approach for this dataset size.

### Results

| Model | Test Accuracy | Notes |
|---|---|---|
| ResNet-50 (transfer learning) | 85% | Overfit quickly; small dataset doesn't suit transfer learning here |
| VGG-19 (transfer learning) | 85% | Same overfitting pattern as ResNet-50 |
| Custom CNN (4× conv+maxpool) | 85% | Slight improvement in confusing classes, still overfits |
| Custom CNN + Dropout | 87% | Overfitting onset delayed from epoch ~1 to epoch ~6 |
| Custom CNN + Dropout + L1 | 90% | L1's sparsity effect helps, since most pixels in the motion images are 0 |
| Custom CNN + Dropout + L2 | 92% | Best single-split result |
| **Custom CNN + Dropout + L1 + K-fold (7 folds)** | **99.64%** | Best overall. Using K-fold maximizes the effective training data (166 vs. ~16 samples/class), which matters a lot at this dataset size |

**Takeaway**: transfer learning (ResNet-50, VGG-19) underperformed a much simpler custom CNN on this small, low-noise dataset. The pretrained models were too complex for the task and overfit immediately. Regularization (dropout + L1) plus K-fold cross-validation closed almost all of the remaining gap, since the biggest constraint here was dataset size, not model capacity.

## DTW vs. Deep Learning. Which is better?

- **Deep learning** is the more *practical* choice for production use, since inference is fast.
- **DTW** is more *robust to extremely small datasets* and doesn't need training at all, but is too slow to scale to a full sign dictionary or live translation.
- In practice, the two could complement each other: DTW catches cases where too little data exists to train a class, deep learning handles everything once enough data is collected.

## Known Limitations & Future Work

- **Difference of frames assumes a static background.** Any motion from the body, head, or background will be picked up by the algorithm. A fix proposed but not yet implemented: use MediaPipe Holistic to bound the hand regions and black out everything outside that box before differencing.
- **DTW only tracks joint position, not joint angle**, so it can't distinguish signs that share a position but differ in finger curl/bend.
- **Dataset is small** (5 classes) relative to the real ESL dictionary (~5,000 signs). Results here demonstrate feasibility, not production-readiness.

## How to Run

```bash
pip install -r requirements.txt
```

For the CNN, run the **Stratified K-Fold Custom NN.ipynb** notebook, making sure to bind your test image to the example variable in the last cell.

## Tech Stack

Python, MediaPipe, TensorFlow/Keras, NumPy, pandas, FastDTW

## References

This project builds on prior work in motion-based gesture detection (difference-of-frames algorithm) and DTW-based sign language recognition using MediaPipe. See citations in the full write-up.
The dataset is not included for privacy reasons, but the result of running the dataset through the difference of frames algorithm is included for training the CNN.

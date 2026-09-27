# Sleep-Stage Classification from Single-Channel EEG — a Study and Critical Analysis

**What this repository is:** a written study of an existing deep-learning pipeline for automatic
sleep staging, and a critical analysis of how it was evaluated. **It contains no experiments of
my own and no code from the studied project.** The study itself is the Arabic report in
[`docs/report_ar.pdf`](docs/report_ar.pdf) (42 pages, 2022):
*«تحليل إشارة الدماغ EEG وتصنيف حالات النوم عند الإنسان باستخدام خوارزميتي CNN و LSTM»*.

## The work studied

- **Implementation:** Carlos Fabbri, *Automatic sleep stage classification with CNN and LSTM*,
  Universidad del Pacífico, Lima, 2020 — code: https://github.com/carlosfg97/AutomaticSleepStageClassifier
  (three Google Colab notebooks; the repository carries no licence, so its code is not reproduced
  here).
- **Data:** Sleep-EDF Expanded, PhysioNet — the 20-subject sleep-cassette subset (39 overnight
  recordings), EEG channel Fpz-Cz at 100 Hz, 30-second epochs scored W, N1, N2, N3 (N3+N4 merged),
  REM. https://physionet.org/content/sleep-edfx/1.0.0/
- **Pre-processing:** epoch extraction adapted from DeepSleepNet (Supratak et al., 2017,
  https://github.com/akaraspt/deepsleepnet).

### Method in brief
1. **Two-branch 1-D CNN** on the raw 3000-sample epoch: a small-filter branch (kernel 50, stride 6)
   for temporal detail and a large-filter branch (kernel 400, stride 50) for frequency content,
   concatenated — the DeepSleepNet representation-learning design.
2. **CNN Concat:** the same CNN fed five consecutive epochs (15 000 samples) to give it context.
3. **CNN + LSTM:** CNN features of five consecutive epochs passed to a two-layer bidirectional LSTM
   with a residual dense path, to model sleep-stage transitions.
4. **Confidence threshold:** the least-confident 20 % of predictions are set aside for a human scorer,
   and accuracy is reported on the remainder.

### Results reported by the author (not reproduced here)

| Source | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Manuscript (20-fold) | CNN Concat | 81.78 % | 75.79 % |
| Manuscript (20-fold) | CNN + LSTM | 85.30 % | 78.56 % in the results section, 79.6 % in the abstract |
| Saved notebook outputs (5-fold) | CNN Concat | 82.83 % | 75.96 % |
| Saved notebook outputs (5-fold) | CNN + LSTM | 84.33 % | 77.82 % |
| Saved notebook outputs | CNN Concat, 80 % most-confident epochs only | 90.13 % | 78.08 % |

Class counts over all 42 308 epochs: W 8 285 · N1 2 804 · N2 17 799 · N3 5 703 · REM 7 717.
N1 is the weak class throughout (F1 ≈ 0.39–0.41).

## Critical analysis of the evaluation

These points concern how the numbers were obtained, not the architecture, which follows
established work.

1. **Subject leakage.** Folds are formed by `StratifiedKFold` over the *pooled epochs of all
   recordings*. Epochs from the same person — and from the same night — therefore appear in both
   training and test folds. The model can exploit person- and night-specific signal traits, so the
   accuracy is an estimate for *people it has already seen*, not for new patients. The standard
   protocol for this dataset is subject-wise cross-validation (e.g. DeepSleepNet's 20-fold,
   one subject held out per fold).
2. **Context windows built from non-neighbouring epochs.** In CNN Concat and CNN + LSTM the
   five-epoch windows are assembled *after* the fold indices are applied, so the "previous" and
   "next" epochs of a sample are often not its real temporal neighbours, and windows can span
   two different recordings. The context the model sees is partly scrambled.
3. **The 90 % figure is not an accuracy of the classifier.** It is accuracy on the 80 % of epochs
   the model was most confident about; 20 % were discarded. It measures a human–machine workflow
   and must always be reported together with its coverage.
4. **Selective illustration.** The hypnogram comparison uses a recording chosen, in the author's
   words, *because it produces an accuracy score comparable to the general accuracy*.
5. **Inconsistencies and reproducibility.** The manuscript states 20 folds while the saved
   notebooks use 5; the abstract and results section give different macro-F1 values; the saved
   code does not run as-is (`cnn_builder()` is called without its required argument;
   `StratifiedKFold(random_state=47)` without `shuffle=True` raises an error in scikit-learn
   ≥ 0.24), so the stored outputs were produced by code that differs from what is saved.

## What a sound evaluation would look like

Subject-wise cross-validation (no person in both training and test), context windows cut from
each recording *before* splitting, macro-F1 and Cohen's κ reported alongside accuracy with
per-fold spread, the confidence-threshold results reported as an accuracy–coverage curve, and
hypnograms shown for every held-out subject or for a random one.

## Next step

A re-implementation under that protocol (PyTorch, same data and channel) is planned as a
separate project, so that results can be reported as my own and compared honestly with the
figures above.

## Rights

The Arabic report is © Moustafa Afara, 2022; code excerpts quoted in it belong to their
respective authors (Fabbri 2020; Supratak et al. 2017) and are credited in the report. The
studied code is available only from its author's repository linked above.

**Author:** Moustafa Afara — signal processing and pattern recognition (audio, biosignals, vision).

# Household Animals Classification Using Deep Learning

UE24CS352A — Machine Learning Mini-Project

A cats-versus-dogs image classifier based on Lei Lin’s *Household Animals Classification Using Deep Learning*. The project compares a VGG-style CNN trained from scratch, regularization, pretrained VGG-16 feature extraction, and VGG-16 fine-tuning.

The implementation includes reproducible dataset splits, saved checkpoints, test evaluation, intermediate activation visualizations, Grad-CAM explanations, and a local Streamlit demonstration.

## Experimental Approach

The experiments follow this sequence:

1. Train VGG-6 from scratch without dropout or data augmentation.
2. Train a fresh VGG-6 with dropout and training augmentation.
3. Freeze ImageNet-pretrained VGG-16 convolutional layers and train a new binary classifier.
4. Fine-tune VGG-16 from `block5_conv1` onward, together with the classifier.
5. Evaluate the saved checkpoints on the same held-out test set.
6. Visualize intermediate activations and generate Grad-CAM heatmaps.

The implementation follows the reference paper’s main methodology. It is not an exact numerical reproduction: settings omitted by the paper are explicitly chosen for this implementation.

## Dataset

Dataset: [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

The labeled dataset contains 25,000 images: 12,500 cats and 12,500 dogs.

Dataset preparation checks image readability, identifies exact duplicates using decoded RGB pixels, and excludes images that fail its checks. Six images were excluded; their individual reasons are recorded in `data/splits/excluded.csv`.

The remaining 24,994 images are split by class using seed 42:

| Split | Cats | Dogs | Total |
|---|---:|---:|---:|
| Training | 9,997 | 9,997 | 19,994 |
| Validation | 1,999 | 1,999 | 3,998 |
| Test | 501 | 501 | 1,002 |

The target split proportions are 80% training, 16% validation, and 4% testing. Integer rounding accounts for the final counts.

- Label `0`: cat
- Label `1`: dog
- Split manifests reference the original images without moving them.
- Exact duplicate images do not cross splits.
- Near-duplicate detection is not implemented.
- The competition’s `test1/` images are unlabeled and are used only for demonstration predictions.

## Models and Training

### VGG-6

Four convolutional blocks use:

- Channels: 32, 64, 128, 128
- Convolution kernels: 3 × 3
- Same padding and ReLU activation
- 2 × 2 max pooling after each convolution

The classifier uses Flatten → Dense(512) → ReLU → optional Dropout → Dense(1).

### VGG-16

The ImageNet-pretrained convolutional feature extractor is retained. Its original classifier is replaced with Flatten → Dense(512) → ReLU → Dropout(0.5) → Dense(1).

During frozen-feature training, only the replacement classifier is updated. During fine-tuning, blocks 1–4 remain frozen and block 5 plus the classifier are trainable.

### Experiment Settings

| Setting | Initial VGG-6 | Regularized VGG-6 | Frozen VGG-16 | Fine-tuned VGG-16 |
|---|---|---|---|---|
| Epochs | 20 | 20 | 20 | 10 additional |
| Optimizer | RMSprop | RMSprop | RMSprop | RMSprop |
| Learning rate | 0.0001 | 0.0001 | 0.0001 | 0.00001 |
| Batch size | 32 | 32 | 32 | 32 |
| Dropout | None | 0.5 | 0.5 | 0.5 |
| Training augmentation | None | Enabled | Enabled | Enabled |

All experiments use:

- 224 × 224 RGB inputs
- ImageNet normalization: mean `(0.485, 0.456, 0.406)`, standard deviation `(0.229, 0.224, 0.225)`
- Binary cross-entropy with logits
- Full-precision computation
- No early stopping
- Validation-loss-based checkpoint selection

Augmentation consists of random horizontal flips with probability 0.5 and rotations within ±10 degrees. Validation and test images receive no random augmentation.

Input size, dense-layer width, dropout rate, augmentation settings, optimizer hyperparameters, and epoch counts are implementation choices where the reference does not specify exact settings.

## Results

All test results use the same 1,002 held-out images and a probability threshold of 0.5.

| Model | Evaluated checkpoint | Test accuracy | Macro F1 | Test loss | Correct |
|---|---|---:|---:|---:|---:|
| Initial VGG-6 | Best validation loss, epoch 7 | 87.13% | 0.8712 | 0.2926 | 873/1,002 |
| Regularized VGG-6 | Best validation loss, epoch 19 | 92.91% | 0.9291 | 0.1853 | 931/1,002 |
| Frozen VGG-16 | Best validation loss, epoch 2 | 98.70% | 0.9870 | 0.0322 | 989/1,002 |
| Fine-tuned VGG-16 | Final fine-tuning epoch, epoch 10 | 99.50% | 0.9950 | 0.0242 | 997/1,002 |

Precision, recall, and F1 summaries use macro averaging.

### Model Selection

Frozen VGG-16 at epoch 2 was selected before test evaluation because it had the lowest validation loss, approximately 0.0428.

Fine-tuning did not improve this validation loss. Therefore:

- The demo uses `outputs/vgg16_frozen/best.pt`.
- `outputs/vgg16_finetuned/best.pt` retains the starting model, recorded as fine-tuning epoch 0.
- The fine-tuned test comparison uses `outputs/vgg16_finetuned/last.pt`, containing epoch 10.

The final fine-tuned model achieved higher test accuracy, but the deployment selection was not changed based on test results.

### Observations

- Initial VGG-6 showed overfitting: training accuracy reached 99.59%, while final validation accuracy was 86.17%.
- Dropout and augmentation improved VGG-6 test accuracy to 92.91%.
- Pretrained VGG-16 features substantially improved classification performance.
- Fine-tuning improved the final model’s observed test performance but did not improve validation loss.

These results describe this split and training run; multiple-seed experiments were not performed.

## Project Structure

```text
app.py                         Local Streamlit demo
training.ipynb                 Training, evaluation, and visualization workflow
scripts/
    prepare_data.py            Image validation and split generation
    data.py                    Dataset and data loaders
    models.py                  VGG6 and VGG16Binary architectures
    explain.py                 Binary Grad-CAM implementation
data/
    train/                     Original labeled images; not committed
    test1/                     Unlabeled demonstration images; not committed
    splits/                    Split manifests and preparation reports
docs/
    Lin (1).pdf                Reference report
    Lin.pdf                    Reference poster
outputs/
    vgg6_initial/              Initial baseline experiment
    vgg6_regularized/          Regularized baseline experiment
    vgg16_frozen/              Frozen-feature experiment
    vgg16_finetuned/           Fine-tuning experiment
    evaluation/               Metrics, predictions, and confusion matrices
    visualizations/           Activation grids and Grad-CAM figures
pyproject.toml                 Project dependencies
uv.lock                        Locked dependency versions
README.md                      Setup and usage instructions
```

Large checkpoints and raw dataset images are distributed separately or regenerated locally.

## Environment Setup

The experiments were run on an NVIDIA GeForce RTX 3050 6 GB Laptop GPU.

Requirements:

- Python version specified in `.python-version`
- `uv`
- Compatible NVIDIA driver for GPU execution
- Access to the Kaggle dataset for training

From the cloned repository root:

```bash
uv sync --locked
```

Verify the environment:

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The training notebook expects CUDA. The Streamlit demo supports CPU fallback.

## Run the Demo

The demo requires this checkpoint:

```text
outputs/vgg16_frozen/best.pt
```

If it is not included with your checkout, obtain the checkpoint from the project authors or reproduce the training workflow below. Preserve the directory path above.

Run from the repository root:

```bash
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://localhost:8501
```

The demo supports:

1. Uploading JPG, JPEG, PNG, or WebP images.
2. Viewing a cat/dog prediction and probability scores.
3. Generating a Grad-CAM explanation of the predicted class.
4. Downloading the explanation as a PNG.

Stop the server with `Ctrl+C`.

## Reproduce the Experiments

### 1. Obtain the Dataset

Download and extract the Kaggle dataset so that labeled images have this layout:

```text
data/train/cat.0.jpg
data/train/dog.0.jpg
...
```

Unlabeled competition images may be placed in `data/test1/`.

### 2. Prepare the Splits

```bash
uv run python scripts/prepare_data.py
```

Inspect:

```text
data/splits/summary.json
data/splits/excluded.csv
```

The manifests contain paths relative to the repository root. All models must use the same manifests.

### 3. Open the Notebook

Open `training.ipynb` in Cursor or another compatible notebook interface and select the repository’s `.venv` Python kernel.

Execute the cells in their documented order:

1. Environment and data checks
2. Initial VGG-6
3. Regularized VGG-6
4. Frozen VGG-16
5. VGG-16 fine-tuning
6. Test evaluation
7. Intermediate activations
8. Grad-CAM examples

The first pretrained VGG-16 initialization downloads its ImageNet weights.

Experiment setup cells refuse to overwrite nonempty experiment directories. For a fresh reproduction, archive the existing experiment directories before running the notebook. Preserve the original results rather than deleting them.

The notebook saves checkpoints after completed epochs but does not implement automatic interrupted-run resumption.

## Evaluation and Visualizations

Evaluation outputs are stored under `outputs/evaluation/`:

- `summary.csv`: model comparison
- `model_selection.json`: validation-based deployment selection
- Per-model `metrics.json`: metrics and confusion matrix
- Per-model `predictions.csv`: individual test predictions
- `confusion_matrices.png`: visual comparison

Visualization outputs are stored under `outputs/visualizations/`:

- VGG-6 intermediate activation grids
- Grad-CAM input, heatmap, and overlay figures
- Heatmap arrays and explanation metadata
- Correctly classified and misclassified examples

The reference paper discusses CAM with global average pooling. This implementation uses **Grad-CAM**, an explicitly documented adaptation that works with the trained Flatten/Dense classifiers without retraining them.

## Limitations

- Only cats and dogs are modeled; other subjects still receive one of these labels.
- Animal illness or behavior detection is not implemented.
- Probability scores are not calibrated confidence guarantees.
- Exact duplicates are checked; near duplicates are not.
- Grad-CAM is a qualitative explanation, not segmentation or proof of causality.
- Performance on unrelated datasets, breeds, environments, and image conditions has not been established.
- Torchvision pretrained weights and preprocessing may differ from the reference implementation.

## References

1. Lei Lin. *Household Animals Classification Using Deep Learning*. Stanford CS229, Spring 2020. Local report: `docs/Lin (1).pdf`.
2. [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
3. Simonyan and Zisserman. [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
4. Selvaraju et al. [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391).
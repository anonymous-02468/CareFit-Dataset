# CareFit Dataset

This repository contains sample code for loading multimodal physiological data and running VO2 Max / VO2 AT inference. The data includes ECG signals, SmO2 signals, personal information, and ground-truth exercise test labels.

## Repository Structure

- `inference.py`: Inference entry point for VO2 Max and VO2 AT prediction.
- `model.py`: PyTorch model definitions, including ECG encoder, SmO2 encoder, and multimodal fusion block.
- `visualization.py`: Simple visualization example for ECG and SmO2 signals.
- `requirements.txt`: Python package dependencies.
- `dataset/ground_truth.csv`: Ground-truth labels and personal information.
- `dataset/*.py`: PyTorch dataset loaders and multimodal collate function.
- `data/s1/`, `data/s2/`, ...: Expected subject data folders. Each folder should contain ECG and SmO2 `.npz` files.
- `ckpts/`: Expected checkpoint folder for pretrained model weights.

## Data Format

`dataset/ground_truth.csv` contains the following columns:

- `id`: Subject identifier.
- `vo2max`: Maximum oxygen consumption.
- `vo2at`: Oxygen consumption at anaerobic threshold.
- `age`: Age of the subject.
- `height`: Height of the subject.
- `weight`: Weight of the subject.
- `bmi`: Body mass index of the subject.
- `fvc0`: Recommended forced vital capacity.
- `fev10`: Recommended forced expired volume in one second.
- `ratio0`: Recommended ratio of FEV1 to FVC.
- `pef0`: Recommended peak expiratory flow rate.
- `fvc`: Forced vital capacity.
- `fev1`: Forced expired volume in one second.
- `ratio`: Ratio of FEV1 to FVC.
- `pef`: Peak expiratory flow rate.
- `time`: Exercise duration in seconds.

The inference code expects each subject folder under `data/` to contain:

- `ecg_clean.npz`: ECG signal data. The loader reads the `data` array from this file.
- `smo2_smooth.npz`: SmO2 signal data. The loader reads the `data` array from this file.

The default inference subject list is `s1` to `s5`, configured in `inference.py`.

## Environment Setup

Create and activate a Python environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA-enabled PyTorch build, install PyTorch by following the command generated on the official PyTorch website, then install the remaining packages from `requirements.txt`.

## Prepare Data and Checkpoints

Before running inference, place the dataset and checkpoint files in the expected locations:

```text
CareFit-Dataset/
|-- ckpts/
|   |-- model_vo2max.pth
|   `-- model_vo2at.pth
|-- data/
|   |-- s1/
|   |   |-- ecg_clean.npz
|   |   `-- smo2_smooth.npz
|   |-- s2/
|   |   |-- ecg_clean.npz
|   |   `-- smo2_smooth.npz
|   `-- ...
`-- dataset/
    `-- ground_truth.csv
```

Due to IRB restrictions and privacy protections, the complete dataset will be released upon paper publication.

## Run Inference

Run VO2 Max inference:

```bash
python inference.py --flag vo2max
```

Run VO2 AT inference:

```bash
python inference.py --flag vo2at
```

The script loads `ckpts/model_<flag>.pth`, runs inference on the configured subject list, denormalizes predictions and labels, and prints MAE and MAPE:

```text
MAE: <value>, MAPE: <value>
```

## Visualization Example

`visualization.py` provides a simple ECG and SmO2 visualization example. If you use the current `data/` and `dataset/` layout, make sure the paths in `visualization.py` match your local data location before running it:

```bash
python visualization.py
```
When you run the script, you will see the following outputs:
- The shape of the ECG raw data and SmO2 data.
- The VO2 Max and VO2 AT values for the specified dataset ID.
- Plots of the ECG raw data and SmO2 data.
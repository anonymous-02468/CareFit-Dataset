# Sample Dataset Analysis

This repository contains a sample dataset and a Python script to analyze the data. The dataset includes physiological data such as ECG raw signals and SmO2 levels, along with ground truth values for VO2 Max and VO2 AT, and other personal information.

## Files

- `ground_truth.csv`: Contains ground truth data with columns:
  - `id`: Identifier for the dataset.
  - `vo2max`: Maximum oxygen consumption.
  - `vo2at`: Oxygen consumption at anaerobic threshold.
  - `age`: Age of the subject.
  - `height`: Height of the subject.
  - `weight`: Weight of the subject.
  - `bmi`: Body mass index of the subject.
  - `fvc0`: Recommendation Value of Forced vital capacity.
  - `fev10`: Recommendation Value of Forced expired volume in one second.
  - `ratio0`: Recommendation Value of Ratio of FEV1 to FVC.
  - `pef0`: Recommendation Value of Peak expiratory flow rate.
  - `fvc`: Forced vital capacity.
  - `fev1`: Forced expired volume in one second.
  - `ratio`: Ratio of FEV1 to FVC.
  - `pef`: Peak expiratory flow rate.
  - `time`: Time duration of the exercise session in seconds.

- `sample_code.py`: Python script to load, process, and visualize the data.
- `0001/`, `0002/`, ...: The id of patients. Directories containing `.npz` files for ECG raw data and SmO2 data.

## Usage

1. Ensure you have Python installed along with the required libraries:
   - `numpy`
   - `pandas`
   - `matplotlib`

2. Run the script `sample_code.py` to load and visualize the data for a specific dataset ID. By default, the script uses the dataset with ID `0001`.

3. The script performs the following steps:
   - Loads the ground truth data from `ground_truth.csv`.
   - Loads the ECG raw data and SmO2 data from the `.npz` files.
   - Extracts VO2 Max and VO2 AT values for the specified dataset ID.
   - Visualizes the ECG raw data and SmO2 data using matplotlib.

## Example Output

When you run the script, you will see the following outputs:

- The shape of the ECG raw data and SmO2 data.
- The VO2 Max and VO2 AT values for the specified dataset ID.
- Plots of the ECG raw data and SmO2 data.

## Visualization

The script generates two plots:

1. **ECG Raw Data**: A time-series plot showing the amplitude of the ECG signal over time.
2. **SmO2 Data**: A time-series plot showing the SmO2 levels over time.

Both plots are displayed in a single figure with subplots for easy comparison.

## Customization

To analyze a different dataset, modify the `id` variable in `sample_code.py` to the desired dataset ID (e.g., `0002`, `0003`, etc.).

## Release

Data requesters will be required to submit an abstract outlining their research purpose and evidence of their own institutional IRB approval or ethics committee review. A formal data use agreement will also be required, and the detailed guidelines for dataset requests will be released following the paper's publication. Due to IRB restrictions and privacy protections, complete dataset will be released upon paper publication.

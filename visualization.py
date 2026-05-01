import numpy as np
import pandas as pd

df = pd.read_csv("./dataset/ground_truth.csv")

# load data
id = "s3"

print(id)

ecg_raw = np.load(f"./data/{id}/ecg_clean.npz", allow_pickle=True)['data']
smo2 = np.load(f"./data/{id}/smo2_smooth.npz", allow_pickle=True)['data']

vo2max = df.loc[df['id'] == id, 'vo2max'].values[0]
vo2at = df.loc[df['id'] == id, 'vo2at'].values[0]

print("ECG Raw Data Shape:", ecg_raw.shape)
print("SmO2 Data Shape:", smo2.shape)
print("VO2 Max:", vo2max)
print("VO2 AT:", vo2at)

from matplotlib import pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_raw, label='ECG Raw Data')
plt.title(f'{id} ECG Raw Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(smo2, label='SmO2 Data', color='orange')
plt.title(f'{id} SmO2 Data')
plt.xlabel('Time')
plt.ylabel('SmO2 Level')
plt.legend()
plt.tight_layout()
plt.show()
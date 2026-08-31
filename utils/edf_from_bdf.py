import mne
import matplotlib.pyplot as plt


fname = r'\\MCSSERVER\DB Temp\physionet.org\files\neurotravel\test_60secs.EDF'

# Читаем файл с помощью MNE
raw = mne.io.read_raw_edf(
    fname,
    preload=True,  # загружаем данные в память
    verbose=False  # отключаем подробный вывод
)

# Настраиваем параметры отображения
raw.plot(
    block = True
)
import mne
import os

base_directory = r'\\MCSSERVER\DB Temp\physionet.org\files\neurotravel'
fout = 'test_raw_crop_from_bdf.EDF'
out = os.path.join(base_directory, fout)
fname = r'\\MCSSERVER\DB Temp\physionet.org\files\neurotravel\test_11h_from_sm.bdf'

raw = mne.io.read_raw_bdf(
    fname,
    preload=True,  # Загружаем данные в память сразу
    verbose=True  # Подробный вывод процесса
)
print(raw.ch_names)

raw_data = raw.get_data()
fs = raw.info.get('sfreq')

# cut 60 secs

raw_crop = raw.crop(tmin=0.0, tmax=60.0, verbose=True)
print('raw_crop', raw_crop )

# Сохраняем в формат EDF
raw_crop.export(
    fname=out,
    fmt='edf',
    overwrite=True
)
print(f"Данные успешно сохранены в файл: {fname}")

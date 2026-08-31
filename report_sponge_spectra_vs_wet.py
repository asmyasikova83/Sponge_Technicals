import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



base_dir = Path(r'\\MCSSERVER\DB Temp\physionet.org\files\sponge_eeg')

ch_renamed = {
    'Fp1-Ref': 'Fp1',
    'F3-Ref': 'F3',
    'F4-Ref': 'F4',
    'C3-Ref': 'C3',
    'C4-Ref': 'C4',
    'T5-Ref': 'T5',
    'T6-Ref': 'T6',
    'P3-Ref': 'P3',
    'O2-Ref': 'O2'
}
target_chs = ['Fp1', 'F3', 'F4', 'C3', 'C4', 'T5', 'T6', 'P3', 'O2']

all = []
for i in range(3):
    fname = base_dir / f'Sponge_EEG_{i+1}.bdf'
    raw = mne.io.read_raw_bdf(
        fname,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
    )

    raw.notch_filter(
    freqs=np.arange(50, 150, 50),  # частоты для фильтрации: 50, 100, 150, 200, 250 Гц
    method='fir',  # метод: FIR‑фильтр
    filter_length='auto',  # автоматическая длина фильтра
    phase='zero',  # нулевая фаза (без сдвига сигнала)
    fir_window='hamming',  # окно Хэмминга
    fir_design='firwin'  # дизайн FIR‑фильтра
    )

    raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

    if 'Fp1' not in raw.ch_names:
        raw.rename_channels(ch_renamed)
    available_target = [ch for ch in target_chs if ch in raw.ch_names]
    all_channels = raw.ch_names

    channels_to_drop = [ch for ch in all_channels if ch not in available_target]

    raw_picked = raw.drop_channels(channels_to_drop)
    data = np.mean(raw_picked.get_data(), axis = 0) #chs

    all.append(data[:39000])

base_dir = Path(r'\\MCSSERVER\DB Temp\physionet.org\files\sponge_eeg')

subjects = ['S00', 'S01', 'S03', 'S04', 'S05']

for subject in subjects:
    fname = base_dir / f'{subject}_wet_eyes_open.bdf'
    raw = mne.io.read_raw_bdf(
        fname,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
    )

    raw.notch_filter(
    freqs=np.arange(50, 150, 50),  # частоты для фильтрации: 50, 100, 150, 200, 250 Гц
    method='fir',  # метод: FIR‑фильтр
    filter_length='auto',  # автоматическая длина фильтра
    phase='zero',  # нулевая фаза (без сдвига сигнала)
    fir_window='hamming',  # окно Хэмминга
    fir_design='firwin'  # дизайн FIR‑фильтра
    )

    # Apply a high-pass filter at 0.5 Hz using a two-way FIR filter
    raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

    data = np.mean(raw.get_data(), axis = 0) #chs

    all.append(data[:39000])

# PSD for all datasets
names  = ['Sponge_EEG_1', 'Sponge_EEG_2', 'Sponge_EEG_3', 'S00', 'S01', 'S03', 'S04', 'S05']

info = mne.create_info(
    ch_names=names,  # имя канала
    sfreq=500,
    ch_types=['eeg'] * len(names)
)

raw_combined = mne.io.RawArray(data=all, info=info)

psd = raw_combined.compute_psd(method='welch', fmax=50)

# Visualize PSD
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['black', 'black', 'black', 'yellow', 'orange', 'plum', 'purple', 'magenta']
channel_names = psd.ch_names
lines = []

for i, ch_name in enumerate(channel_names):
    psd.plot(picks=ch_name, axes=ax, show=False)
    line = ax.get_lines()[-1]
    line.set_color(colors[i])
    line.set_label(ch_name)
    line.set_linewidth(2.5)
    lines.append(line)

ax.legend(
    handles=lines,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=12,
    title_fontsize=14,
    framealpha=0.9
)

ax.set_title(
    'PSD по мокрой ЭЭГ с разных чуваков + Sponge EEG',
    fontsize=16,
    fontweight='bold'
)
ax.set_xlabel(
    'Частота, Гц',
    fontsize=14,
    fontweight='semibold'
)
ax.set_ylabel(
    'Мощность, дБ',
    fontsize=14,
    fontweight='semibold'
)

# Увеличиваем размер меток на осях
ax.tick_params(
    axis='both',
    which='major',
    labelsize=12,
    width=2,
    length=6
)

ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(base_dir / 'PSD_combined.png', dpi=300, bbox_inches='tight')
plt.show()




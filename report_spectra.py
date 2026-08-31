import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path(r'\\MCSSERVER\DB Temp\msasha\Sponge')

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

    # Беру мин. длит-ть t
    all.append(data[:39000])

###########et eeg as control###################

base_dir_wet = Path(r'\\MCSSERVER\DB Temp\msasha\Sponge')

fname_wet = base_dir_wet / 'S05_wet_eyes_open.bdf'

raw = mne.io.read_raw_bdf(
        fname_wet,
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
data_wet = np.mean(raw.get_data(), axis = 0) #chs

all.append(data_wet[:39000])

# Создаём info с корректными параметрами
info = mne.create_info(
    ch_names=['Sponge 1','Sponge 2', 'Sponge 3', 'Wet'],  # имя канала
    sfreq=500,
    ch_types=['eeg', 'eeg','eeg','eeg']     # тип канала (в виде списка)
)

# Создаём Raw-объект (исправлено: create Raw array → mne.io.RawArray)
raw_combined = mne.io.RawArray(data=all, info=info)

psd = raw_combined.compute_psd(method='welch', fmax=50)  # ограничиваем частоту до 50 Гц для наглядности

# Строим график с настройками оформления
# Получаем имена каналов
channel_names = psd.ch_names

fig, ax = plt.subplots(figsize=(12, 6))

# Генерируем цвета для каждого канала
colors = ['red', 'green', 'blue', 'magenta']

lines = []  # для хранения объектов линий

for i, ch_name in enumerate(channel_names):
    psd.plot(picks=ch_name, axes=ax, show=False)
    line = ax.get_lines()[-1]
    line.set_color(colors[i])
    line.set_label(ch_name)
    line.set_linewidth(2.5)
    lines.append(line)

# Создаём легенду на основе собранных линий с увеличенным шрифтом
ax.legend(
    handles=lines,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=12,  # размер шрифта в легенде
    title_fontsize=14,  # размер шрифта заголовка легенды (если есть)
    framealpha=0.9  # небольшая прозрачность фона легенды
)

# Настройки заголовков и подписей осей с увеличенным размером шрифта
ax.set_title(
    'PSD усредненн. по каналам записей Sponge + мокрая ЭЭГ',
    fontsize=16,  # размер шрифта заголовка
    fontweight='bold'  # жирный шрифт для заголовка
)
ax.set_xlabel(
    'Частота, Гц',
    fontsize=14,  # размер шрифта подписи оси X
    fontweight='semibold'  # полужирный шрифт
)
ax.set_ylabel(
    'Мощность, дБ',
    fontsize=14,  # размер шрифта подписи оси Y
    fontweight='semibold'  # полужирный шрифт
)

# Увеличиваем размер меток на осях
ax.tick_params(
    axis='both',
    which='major',
    labelsize=12,  # размер шрифта меток осей
    width=2,  # толщина основных тиков
    length=6  # длина основных тиков
)

ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(base_dir / 'PSD.png', dpi=300, bbox_inches='tight')
plt.show()
import os
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne.channels import make_standard_montage

def average_raw(reconst_raw):
    reconst_raw_to_psd = reconst_raw.pick_channels(['O1', 'O2', 'Pz'])
    data = reconst_raw_to_psd.get_data()
    data_mean = np.mean(data, axis=0)
    info1ch = mne.create_info(
        ch_names=['Posterior'],
        sfreq=500,
        ch_types='eeg'
    )
    reconst_raw_mean = mne.io.RawArray(data_mean[np.newaxis], info1ch)

    return reconst_raw_mean

def get_events(raw):
    events_from_ann = mne.events_from_annotations(raw, verbose=False)
    events = events_from_ann[0]  # events — массив Nx3
    event_id = events_from_ann[1]  # словарь id → описание

    print(f"Найдено событий из аннотаций: {len(events)}")
    print("Event ID mapping:", event_id)

    target_id = 1
    events_open = events[events[:, 2] == target_id]
    target_id = 2
    events_closed = events[events[:, 2] == target_id]

    return events_open, events_closed

def make_x_ticks(rms_uv):
    total_time_s = rms_uv.shape[1]
    n_points = 34
    step = total_time_s / (n_points - 1)

    # Массив времени для каждой точки (от 0 до 330 с)
    time_axis = np.linspace(0, total_time_s, n_points)
    tick_labels = np.arange(0, round(raw.times[-1]) + 10, 10)

    return time_axis, tick_labels

def rms_sliding_window_per_channel(raw, window_sec=1.0, step_sec=0.1):
    """
    Считает RMS в скользящих окнах для каждого канала отдельно.
    Возвращает:
        rms_matrix: (n_channels, n_windows)
        time_centers_sec: центры окон в секундах (удобнее для heatmap)
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # (n_channels, n_samples)
    n_ch, n_samp = data.shape
    win_samp = int(window_sec * sfreq)

    step_samp = max(1, int(step_sec * sfreq))
    n_windows = int((n_samp - win_samp) / step_samp) + 1
    rms_matrix = np.empty((n_ch, n_windows), dtype=float)

    for i in range(n_windows):
        start = i * step_samp
        end = start + win_samp
        chunk = data[:, start:end]
        # RMS = sqrt(mean(x^2))
        rms_matrix[:, i] = np.sqrt(np.mean(chunk**2, axis=1))
        #Check
        #rms_matrix[:, i] =  np.sqrt(np.sum(chunk**2, axis=1) / chunk.shape[1])

    # Центры окон (для понятной оси времени на heatmap)
    time_centers_sec = (np.arange(n_windows) * step_samp + win_samp / 2) / sfreq

    return rms_matrix, time_centers_sec

def segment_and_psd(reconst_raw_mean, events_open,events_closed):
    epochs_open_clean = mne.Epochs(
        reconst_raw_mean,
        events_open,
        tmin=0,
        baseline=None,
        preload=True
    )
    psd_open_clean = epochs_open_clean.compute_psd(fmin=1, fmax=30)
    epochs_closed_clean = mne.Epochs(
        reconst_raw_mean,
        events_closed,
        tmin=0,
        baseline=None,
        preload=True
    )
    psd_closed_clean = epochs_closed_clean.compute_psd(fmin=1, fmax=30)

    psd_open_avg = psd_open_clean.get_data().mean(axis=0).mean(axis=0)
    psd_closed_avg = psd_closed_clean.get_data().mean(axis=0).mean(axis=0)
    freqs = psd_open_clean.freqs

    return psd_open_avg, psd_closed_avg, freqs

base_dir = Path(r'\\MCSSERVER\DB Temp\msasha\Sponge')

fname = base_dir / f'NeoRec_2026-08-27_16-01-28.edf'
raw = mne.io.read_raw_edf(
        fname,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
)

rms_mat, t_centers = rms_sliding_window_per_channel(raw, window_sec=1.0, step_sec=0.1)


# Перевод в мкВ (если нужно для удобства)
rms_uv = rms_mat * 1e6

time_axis, tick_labels = make_x_ticks(rms_uv)

ch_names = raw.ch_names
# RMS heatmap
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    rms_uv,
    cmap='viridis',
    cbar_kws={'label': 'RMS (µV)'},
    yticklabels=ch_names,
    vmin=0,
    vmax=100,
    ax=ax  # важно: передаём оси, чтобы seaborn рисовал туда, куда нужно
)

# Настройка меток времени
plt.xticks(time_axis)  # или используй tick_indices, если они у тебя есть
ax.set_xticklabels(tick_labels, rotation=0, fontsize=6)
plt.xlabel('Время (s)')
plt.ylabel('Канал')
plt.title('Карта шума (RMS, 1s sliding window)')

plt.tight_layout()

fname = os.path.join(base_dir, "rms_heatmap.png")
os.makedirs(base_dir, exist_ok=True)  # создадим папку, если её нет

fig.savefig(fname, dpi=300, bbox_inches='tight')
print(f"Heatmap сохранён: {fname}")

plt.close(fig)  # освобождаем память

raw_picked = raw.drop_channels(['Cz'])

raw_picked.notch_filter(
    freqs=50,
    method='fir',
    filter_length='auto',
    phase='zero',
    fir_window='hamming',
    fir_design='firwin',
    trans_bandwidth=2.5,      # ширина переходной полосы (Гц)
    n_jobs=4                 # параллельные потоки для скорости
)

#ICA

ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(raw_picked, reject=dict(eeg=200e-6))  # avoid a couple of big artifacts

raw_ica = raw_picked.copy()
montage = make_standard_montage('standard_1020')
raw_ica.set_montage(montage, on_missing='warn')  # warn покажет, какие каналы не нашли
chs_with_loc = [ch for ch, ch_info in zip(raw_ica.ch_names, raw_ica.info['chs']) if ch_info.get('loc') is not None]
print(f"Каналы с координатами: {len(chs_with_loc)} из {len(raw_ica.ch_names)}")

ica = ICA(n_components=0.95, method='fastica', random_state=42)
ica.fit(raw_ica, reject=dict(eeg=200e-6))
fig_components  = ica.plot_components(ch_type='eeg', show=True)
fname = os.path.join(base_dir, f"ica_components.png")
fig_components.savefig(fname, dpi=300, bbox_inches='tight')
print(f"Сохранено: {fname}")

fig_sources = ica.plot_sources(raw_ica, show_scrollbars=False)
fname = os.path.join(base_dir, f"ica_sources.png")
fig_sources.savefig(fname, dpi=300, bbox_inches='tight')
plt.close('all')

#ica.plot_properties(raw_ica, picks=[0, 1])
ica.exclude = [0, 1, 2, 4]
reconst_raw = raw_ica.copy()
ica.apply(reconst_raw)

#raw_ica.plot()
#reconst_raw.plot()
#plt.show()

# Spectra
events_open, events_closed = get_events(raw)

reconst_raw_mean = average_raw(reconst_raw)
psd_open_reconst_avg, psd_closed_reconst_avg, freqs = segment_and_psd(reconst_raw_mean, events_open,events_closed)

raw_mean = average_raw(raw)
psd_open_avg, psd_closed_avg, freqs = segment_and_psd(raw_mean, events_open,events_closed)

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(freqs, psd_open_reconst_avg * 10e9, linewidth=2, label='Открытые глаза после чистки')
ax.semilogy(freqs, psd_closed_reconst_avg * 10e9, linewidth=2, label='Закрытые глаза после чистки')
ax.semilogy(freqs, psd_open_avg * 10e9, linewidth=2, linestyle='--', label='Открытые глаза до чистки')
ax.semilogy(freqs, psd_closed_avg * 10e9, linewidth=2, linestyle='--', label='Закрытые глаза до чистки')

ax.set_xlabel('Частота (Hz)')
ax.set_ylabel('Мощность (uV²/Hz)')
ax.set_title('PSD: Открытые vs Закрытые глаза (усреднено по O1, O2, Pz)')
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend(loc='best')

alpha_range = (8, 12)
ax.axvspan(alpha_range[0], alpha_range[1], color='gray', alpha=0.15, label='Полоса альфы')
plt.tight_layout()

fname = os.path.join(base_dir, "Spectra_alpha.png")
fig.savefig(fname, dpi=300, bbox_inches='tight')
print(f"График сохранён: {fname}")

plt.close(fig)

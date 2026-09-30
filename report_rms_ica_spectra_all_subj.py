import os
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne.channels import make_standard_montage

ch_renamed = {
    'Fp1-Ref': 'Fp1',
    'Fp2-Ref': 'Fp2',
    'F7-Ref': 'F7',
    'F3-Ref': 'F3',
    'Fz-Ref': 'Fz',
    'F4-Ref': 'F4',
    'F8-Ref': 'F8',
    'T3-Ref': 'T3',
    'C3-Ref': 'C3',
    'C4-Ref': 'C4',
    'T5-Ref': 'T5',
    'P3-Ref': 'P3',
    'P4-Ref': 'P4',
    'Pz-Ref': 'Pz',
    'T4-Ref': 'T4',
    'T6-Ref': 'T6',
    'O1-Ref': 'O1',
    'Oz-Ref': 'Oz',
    'O2-Ref': 'O2'


}
target_chs = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'C4', 'T5', 'P3', 'Pz', 'P4', 'T4', 'T6', 'O1', 'Oz', 'O2']

def average_raw(reconst_raw, chan):
    reconst_raw_to_psd = reconst_raw.pick_channels(chan)
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

    target_id = 2
    events_open = events[events[:, 2] == target_id]
    target_id = 1
    events_closed = events[events[:, 2] == target_id]
    return events_open, events_closed

def make_x_ticks(rms_uv, fname):
    total_time_s = rms_uv.shape[1]
    if fname.stem == 'Test1_Zhenya_small_sensors':
        n_points = 38
    elif fname.stem == 'Test5_DA_big_sensors_tense':
        n_points = 28
    else:
        n_points = 26
    #n_points = 92
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

def save_psd(freqs, psd_open, psd_closed, stem, out_dir, chan):
    """Сохраняет PSD (открытые/закрытые глаза) + ось частот в .npy и .npz."""
    stem = stem.replace(" ", "_")          # на случай пробелов в именах файлов

    # вариант 1: один архив со всеми массивами (удобно грузить целиком)
    npz_path = out_dir / f"psd_{stem}.npz"
    np.savez(npz_path,                     # np.savez_compressed — если файлы большие
             freqs=np.asarray(freqs, dtype=np.float64),
             psd_open=np.asarray(psd_open, dtype=np.float64),
             psd_closed=np.asarray(psd_closed, dtype=np.float64))

    # вариант 2: отдельные .npy с нужным суффиксом (удобно для Gloo/Excel/других скриптов)
    np.save(out_dir / f"psd_open_avg_{stem}_{chan}.npy",   np.asarray(psd_open,   dtype=np.float64))
    np.save(out_dir / f"psd_closed_avg_{stem}_{chan}.npy", np.asarray(psd_closed, dtype=np.float64))
    np.save(out_dir / f"freqs_{stem}_{chan}.npy",          np.asarray(freqs,      dtype=np.float64))

    print(f"PSD сохранён: {npz_path}")
    return npz_path

def segment_and_psd(reconst_raw_mean, events_open,events_closed):
    d = reconst_raw_mean.get_data()
    print('reconst_raw_mean shape', d.shape)
    epochs_open_clean = mne.Epochs(
        reconst_raw_mean,
        events_open,
        tmin=0,
        baseline=(0, 0.5),
        preload=True
    )
    print(f"tmin={epochs_open_clean.tmin}, tmax={epochs_open_clean.tmax}")
    psd_open_clean = epochs_open_clean.compute_psd(
        method='welch',
        fmin=1.0, fmax=30.0,
        n_fft=250,
        n_overlap=125,  # 50 % перекрытия
        remove_dc=True,
        n_jobs=4,
    )
    epochs_closed_clean = mne.Epochs(
        reconst_raw_mean,
        events_closed,
        tmin=0,
        baseline=(0, 0.5),
        preload=True
    )

    psd_closed_clean = epochs_closed_clean.compute_psd(
        method='welch',
        fmin=1.0, fmax=30.0,
        n_fft=250,  # ← при 256 Гц: окно 8 с, Δf = 0.125 Гц
        n_overlap=125,  # 75 % перекрытия
        remove_dc=True,
        n_jobs=4,
    )
    psd_open_avg = psd_open_clean.get_data().mean(axis=0).mean(axis=0)
    psd_closed_avg = psd_closed_clean.get_data().mean(axis=0).mean(axis=0)
    freqs = psd_open_clean.freqs

    return psd_open_avg, psd_closed_avg, freqs

base_dir = Path(r'\\MCSSERVER\DB Temp\msasha\Sponge\Sponge_EEG_session3_18Sep2026')
pic_dir = Path(r'\\MCSSERVER\DB Temp\msasha\Sponge\Sponge_EEG_session3_18Sep2026\pics')
#base_dir = Path(r'\\MCSSERVER\DB Temp\msasha\Sleep')
#fname = base_dir / f'Sponge_EEG_session2_9Sep2026.edf'
#fname = base_dir / f'Sponge_EEG_session1_27Aug2026.edf'
#fname = base_dir / f'NeoRec_2026-08-27_16-01-28.edf'
#fname = base_dir / f'NeoRec_2026-09-08_16-03-16.edf'

#chan = ['P3', 'Pz', 'P4', 'O1', 'O2']
chan = ['Fp1']

for fname in sorted(base_dir.glob('*.bdf')):
    print(f"Обрабатываем: {fname.name}")
    try:
        raw = mne.io.read_raw_bdf(
        fname,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
    )
        if 'Fp1' not in raw.ch_names:
            raw.rename_channels(ch_renamed)
        available_target = [ch for ch in target_chs if ch in raw.ch_names]
        all_channels = raw.ch_names

        channels_to_drop = [ch for ch in all_channels if ch not in available_target]

        raw_picked = raw.drop_channels(channels_to_drop)
        print('sfreq:', raw_picked.info['sfreq'])

        # rms_mat, t_centers = rms_sliding_window_per_channel(raw_picked, window_sec=1.0, step_sec=0.1)
        rms_mat, t_centers = rms_sliding_window_per_channel(raw_picked, window_sec=1.0, step_sec=0.1)

        # Перевод в мкВ (если нужно для удобства)
        rms_uv = rms_mat * 1e6

        time_axis, tick_labels = make_x_ticks(rms_uv, fname)

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
        plt.title(f'Карта шума (RMS) - {fname.stem}')

        plt.tight_layout()

        #rms_fname = os.path.join(base_dir, "rms_heatmap.png")
        rms_fname = pic_dir / f"rms_heatmap_{fname.stem}.png"
        print('rms_fname ', rms_fname )
        #os.makedirs(base_dir, exist_ok=True)  # создадим папку, если её нет

        fig.savefig(rms_fname, dpi=300, bbox_inches='tight')
        print(f"Heatmap сохранён: {rms_fname}")

        plt.close(fig)  # освобождаем память

        #raw_picked = raw.drop_channels(['Cz'])

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

        #ica = ICA(n_components=15, max_iter="auto", random_state=97)
        #ica = ICA(n_components=5, max_iter="auto", random_state=97)
        ica = ICA(n_components=0.95, method='fastica', random_state=42)
        ica.fit(raw_picked, reject=dict(eeg=200e-6))  # avoid a couple of big artifacts

        raw_ica = raw_picked.copy()
        montage = make_standard_montage('standard_1020')
        raw_ica.set_montage(montage, on_missing='warn')  # warn покажет, какие каналы не нашли
        chs_with_loc = [ch for ch, ch_info in zip(raw_ica.ch_names, raw_ica.info['chs']) if ch_info.get('loc') is not None]
        print(f"Каналы с координатами: {len(chs_with_loc)} из {len(raw_ica.ch_names)}")

        ica = ICA(n_components=0.95, method='fastica', random_state=42)
        ica.fit(raw_ica, reject=dict(eeg=200e-6))
        fig_components  = ica.plot_components(ch_type='eeg', show=True)
        ica_fname = pic_dir / f"ica_components_{fname.stem}.png"
        fig_components.savefig(ica_fname, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {ica_fname}")

        fig_sources = ica.plot_sources(raw_ica, show_scrollbars=False)
        ica_sources_fname = pic_dir / f"ica_soures_{fname.stem}.png"
        fig_sources.savefig(ica_sources_fname, dpi=300, bbox_inches='tight')
        plt.close('all')

        #ica.plot_properties(raw_ica, picks=[0, 1])

        if fname.stem == 'Test1_Zhenya_small_sensors':
            print(f'excluding...{fname.stem}')
            ica.exclude = [3, 7, 2]
        if fname.stem == 'Test2_Zhenya_big_sensors_tense':
            print(f'excluding...{fname.stem}')
            ica.exclude = [3, 1, 2, 11, 8]
        if fname.stem == 'Test3_Zhenya_big_sensors_loose':
            print(f'excluding...{fname.stem}')
            ica.exclude = [0, 1, 3, 5, 11]
        if fname.stem == 'Test4_DA_small_sensor':
            print(f'excluding...{fname.stem}')
            ica.exclude = [0, 1, 2, 3, 4, 5, 8 ]
        if fname.stem == 'Test5_DA_big_sensors_tense':
            ica.exclude = [3, 9 ]

        reconst_raw = raw_ica.copy()
        print(reconst_raw, 'reconst_raw')
        ica.apply(reconst_raw)

        raw_ica.plot()
        reconst_raw.plot()
        #plt.show()

        # Spectra
        events_open,  events_closed  = get_events(raw)
        print('events_open', events_open)
        print('events_closed', events_closed)

        reconst_raw_mean = average_raw(reconst_raw, chan)
        psd_open_reconst_avg, psd_closed_reconst_avg, freqs = segment_and_psd(reconst_raw_mean, events_open,events_closed)
        save_psd(freqs, psd_open_reconst_avg, psd_closed_reconst_avg, fname.stem, pic_dir, chan)

        raw_mean = average_raw(raw, chan)
        psd_open_avg, psd_closed_avg, freqs = segment_and_psd(raw_mean, events_open,events_closed)


        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(freqs, psd_open_reconst_avg * 10e12, linewidth=2, label='Открытые глаза после чистки')
        ax.semilogy(freqs, psd_closed_reconst_avg * 10e12, linewidth=2, label='Закрытые глаза после чистки')
        #ax.semilogy(freqs, psd_open_avg* 10e9, linewidth=2, linestyle='--', label='Открытые глаза до чистки')
        #ax.semilogy(freqs, psd_closed_avg * 10e9, linewidth=2, linestyle='--', label='Закрытые глаза до чистки')

        ax.set_xlabel('Частота (Гц)')
        ax.set_ylabel('Мощность (мкВ²/Гц)')
        ax.set_title(f'PSD: {fname.stem} Открытые vs Закрытые глаза  ( {chan})')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(loc='best')

        alpha_range = (8, 12)
        ax.axvspan(alpha_range[0], alpha_range[1], color='gray', alpha=0.15, label='Полоса альфы')
        plt.tight_layout()

        spectra_fname = pic_dir / f"Spectra_alpha_{fname.stem}.png"
        fig.savefig(spectra_fname, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {spectra_fname}")

        plt.close(fig)
    except Exception as e:
        print(f"Ошибка при чтении {fname.name}: {e}")
        # Если нужно — продолжить обработку остальных файлов, а не падать
        continue
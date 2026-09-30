import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

chan = ['P3', 'Pz', 'P4', 'O1', 'O2']
#chan = ['Fp1']
# --- НАСТРОЙКИ ------------------------------------------------------------
data_dir = Path(fr"\\MCSSERVER\DB Temp\msasha\Sponge\Sponge_EEG_session3_18Sep2026\pics\spectra_clean_{chan}")

C_CLOSE, C_OPEN = "tab:blue", "tab:orange"
PSD_SCALE  = 1e12          # поставьте 1.0, если в .npy уже масштабированные значения
INTERP     = True          # тянуть спектры другой длины на общую частотную сетку
AVG_LOG    = True          # среднее = геометрическое (в log10), иначе — арифметическое
ALPHA_BAND = (8, 12)
FMAX       = 30.0          # <- ограничение оси частот
FMIN       = 1.0
SHOW_INDIV = True          # тонким фоном показывать все отдельные кривые

if not data_dir.exists():
    raise FileNotFoundError(f"Папка не найдена: {data_dir}")

npy = [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".npy"]

def stem_of(p: Path) -> str:
    s = re.sub(r"(?i)^psd[_\-]?", "", p.stem)
    s = re.sub(r"(?i)[_\-]?(open|closed|avg|raw|clean|reconst)[_\-]?", "_", s)
    return s.strip("_- ")

# --- 1. одна ось частот на всех -------------------------------------------
freq_files = [f for f in npy if "freq" in f.name.lower()]
if not freq_files:
    freq_files = [f for f in npy
                  if "open" not in f.name.lower() and "closed" not in f.name.lower()]
if not freq_files:
    raise FileNotFoundError("В папке нет ни одного файла с осями частот.")

freq_file = sorted(freq_files)[0]
freqs_all = np.load(freq_file).astype(np.float64)
print(f"Ось частот: {freq_file.name} (N={freqs_all.size}, "
      f"{freqs_all.min():.2f}–{freqs_all.max():.2f} Гц)")

# --- 2. собираем open / closed в матрицы -----------------------------------
groups = {"open": [], "closed": []}
for f in npy:
    low = f.name.lower()
    if   "open"   in low: groups["open"].append(f)
    elif "closed" in low: groups["closed"].append(f)

def load_matrix(files):
    """Стекает спектры в матрицу (N_сессий, N_частот) на сетке freqs_all."""
    rows = []
    for f in sorted(files):
        y = np.load(f).astype(np.float64)
        if y.ndim != 1:
            print(f"  пропуск (не 1D): {f.name}, shape={y.shape}")
            continue
        if y.size == freqs_all.size:
            yi = y
        elif INTERP:
            x_old = np.linspace(freqs_all[0], freqs_all[-1], y.size)
            yi = np.interp(freqs_all, x_old, y)
            print(f"  {f.name}: {y.size} != {freqs_all.size} → интерполяция")
        else:
            print(f"  пропуск (длина {y.size}): {f.name}")
            continue
        rows.append(np.where(yi > 0, yi, np.nan))   # лог-шкала: нули -> nan
    return np.vstack(rows) if rows else np.empty((0, freqs_all.size))

M_open   = load_matrix(groups["open"])
M_closed = load_matrix(groups["closed"])
print(f"Сессий: open={M_open.shape[0]}, closed={M_closed.shape[0]}")

if M_open.shape[0] == 0 or M_closed.shape[0] == 0:
    raise ValueError("Не найдено данных хотя бы в одной из групп — график не строим.")

# --- 3. среднее и разброс ---------------------------------------------------
def agg(mat):
    """Возвращает (среднее, среднее+sd, среднее-sd) по строкам матрицы."""
    if AVG_LOG:
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log10(mat)
        m = np.nanmean(L, axis=0)
        sd = np.nanstd(L, axis=0)
        return 10**m, 10**(m + sd), 10**(m - sd)
    m  = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    return m, m + sd, m - sd

mean_open,  hi_open,  lo_open  = agg(M_open)
mean_close, hi_close, lo_close = agg(M_closed)

# --- 4. окно 1–30 Гц --------------------------------------------------------
mask = (freqs_all >= FMIN) & (freqs_all <= FMAX)
x = freqs_all[mask]

fig, ax = plt.subplots(figsize=(10, 6))

if SHOW_INDIV:
    for row in M_closed: ax.semilogy(x, row[mask] * PSD_SCALE, color=C_CLOSE, lw=0.8, alpha=0.18, zorder=1)
    for row in M_open:   ax.semilogy(x, row[mask] * PSD_SCALE, color=C_OPEN,  lw=0.8, alpha=0.18, zorder=1)

ax.fill_between(x, lo_close * PSD_SCALE, hi_close * PSD_SCALE, color=C_CLOSE, alpha=0.18,
                label="±1 SD (закрытые)", zorder=2)
ax.fill_between(x, lo_open  * PSD_SCALE, hi_open  * PSD_SCALE, color=C_OPEN,  alpha=0.18,
                label="±1 SD (открытые)", zorder=2)

ax.semilogy(x, mean_close * PSD_SCALE, color=C_CLOSE, lw=2.5, zorder=4, label="Среднее — закрытые глаза")
ax.semilogy(x, mean_open  * PSD_SCALE, color=C_OPEN,  lw=2.5, zorder=4, label="Среднее — открытые глаза")

ax.axvspan(*ALPHA_BAND, color="gray", alpha=0.15, zorder=0)
ax.set_xlim(FMIN, FMAX)
ax.set_xlabel("Частота (Гц)")
ax.set_ylabel(f"Мощность (мкВ²/Гц)")
ax.set_title(f"Chan: {chan} {f.stem.split('_')[-4:-1]} PSD: closed (n={M_closed.shape[0]}) vs open (n={M_open.shape[0]})")
ax.set_xticks(np.arange(0, FMAX + 1, 5))
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend(loc="upper right")

plt.tight_layout()
out = data_dir / f"spectra_mean_1-30Hz_{chan}.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Сохранено: {out}")
plt.show()

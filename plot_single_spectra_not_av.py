import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

chan = 'Fp1'
# --- НАСТРОЙКИ ------------------------------------------------------------
data_dir = Path(fr"\\MCSSERVER\DB Temp\msasha\Sponge\Sponge_EEG_session3_18Sep2026\pics\spectra_clean_{chan}")

C_CLOSE, C_OPEN = "tab:blue", "tab:orange"   # закрытые — синий, открытые — оранжевый
PSD_SCALE  = 1e12     # уберите (поставьте 1.0), если в .npy уже отмасштабированные значения
INTERP     = True     # интерполировать спектры на частотную сетку из выбранного freqs
ALPHA_BAND = (8, 12)
X_LIMITS   = (1, 40)

if not data_dir.exists():
    raise FileNotFoundError(f"Папка не найдена: {data_dir}")

npy = [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".npy"]

def stem_of(p: Path) -> str:
    """Из psd_open_avg_Test3_zhenya.npy -> 'Test3_zhenya'."""
    s = re.sub(r"(?i)^psd[_\-]?", "", p.stem)
    s = re.sub(r"(?i)[_\-]?(open|closed|avg|raw|clean|reconst)[_\-]?", "_", s)
    return s.strip("_- ")

# --- 1. один freqs на всех -------------------------------------------------
freq_files = [f for f in npy if "freq" in f.name.lower()]
if not freq_files:                          # если частотный файл не назван со 'freq' — берём первый попавшийся
    freq_files = [f for f in npy
                  if "open" not in f.name.lower() and "closed" not in f.name.lower()]
if not freq_files:
    raise FileNotFoundError("В папке нет ни одного файла с осями частот.")

freq_file = sorted(freq_files)[0]
freqs = np.load(freq_file)
print(f"Ось частот взята из: {freq_file.name}  (N={freqs.size}, {freqs.min():.2f}–{freqs.max():.2f} Гц)")

# --- 2. все open / closed ---------------------------------------------------
groups = {"open": [], "closed": []}
for f in npy:
    low = f.name.lower()
    if   "open"   in low: groups["open"].append(f)
    elif "closed" in low: groups["closed"].append(f)

print(f"Найдено файлов: open={len(groups['open'])}, closed={len(groups['closed'])}")

# --- 3. построение ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
skipped = []

for kind, color in (("closed", C_CLOSE), ("open", C_OPEN)):
    for f in sorted(groups[kind]):
        y = np.load(f).astype(np.float64)

        if y.size == freqs.size:
            x, y_plot = freqs, y
        elif INTERP and y.ndim == 1:
            print(f"  {f.name}: длина {y.size} != {freqs.size} → интерполяция")
            x_old = np.linspace(freqs[0], freqs[-1], y.size)
            y_plot = np.interp(freqs, x_old, y)
            x = freqs
        else:
            skipped.append((f.name, y.size))
            continue

        # semilogy не любит нули и отрицательные значения
        y_plot = np.where(y_plot > 0, y_plot, np.nan) * PSD_SCALE
        ax.semilogy(x, y_plot, color=color, lw=1.2, alpha=0.65)
        print(f"  {kind:6s} {f.name} -> {stem_of(f)}")

if skipped:
    print(f"Пропущено (несовпадение длин, INTERP=False): {skipped}")

# --- 4. оформление ----------------------------------------------------------
ax.axvspan(*ALPHA_BAND, color="gray", alpha=0.15)
ax.set_xlim(*X_LIMITS)
ax.set_xlabel("Частота (Гц)")
ax.set_ylabel(f"Мощность (мкВ²/Гц)")
ax.set_title(f"{chan} PSD: {len(groups['open'])} open vs {len(groups['closed'])} closed")
ax.grid(True, which="both", ls="--", alpha=0.3)

# легенда из «эталонных» линий, чтобы не плодить подписи на каждую сессию
ax.legend(handles=[Line2D([0], [0], color=C_CLOSE, lw=2, label="Закрытые глаза"),
                   Line2D([0], [0], color=C_OPEN,  lw=2, label="Открытые глаза")],
          loc="upper right")

plt.tight_layout()

out = data_dir / "spectra_all.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Сохранено: {out}")
plt.show()
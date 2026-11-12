"""Edge2 DMIC event classifier v1: 34-dim window features -> {quiet,speech,keyboard,wind}.

Data: the board's own labeled clips (记录/dmic-dataset).  Evaluation is a
*temporal* 70/30 split inside each clip -- optimistic, but it catches gross
overfit.  Real acceptance needs a fresh labeled capture session.
"""
import json, wave
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[2])  # khadas-ows-node repo root
BANDS = np.array([70.7,141.4,244.9,346.4,451.7,566.8,696.5,841.7,996.8,1171.3,1370.0,
                  1595.0,1854.7,2154.1,2503.3,2916.3,3414.7,4036.1,4829.1,5824.1,7020.0,
                  8552.8,10677.1,13638.2])
VOCAL = (BANDS >= 300.0) & (BANDS <= 4000.0)
FRAME_MS = 100
WINDOW = 10
CLASSES = ["quiet", "speech", "keyboard", "wind"]

def load(path):
    w = wave.open(path, 'rb'); rate = w.getframerate(); ch = w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1, ch).mean(axis=1) / 32768.0
    w.close()
    return rate, x

def goertzel_bands(mono, rate):
    n = len(mono)
    win = np.hanning(n)
    weighted = mono * win
    seconds = np.arange(n) / float(rate)
    phases = np.exp(-2j * np.pi * np.outer(BANDS, seconds))
    spectrum = phases @ weighted
    power = np.maximum(spectrum.real**2 + spectrum.imag**2, 1e-12)
    amp = 2.0 * np.sqrt(power) / max(n, 1)
    return 20.0 * np.log10(np.maximum(amp, 1e-6))

def periodicity(mono, rate):
    x = mono - mono.mean()
    lo = max(1, int(rate * 0.002)); hi = min(len(x) - 1, int(rate * 0.012))
    step = max(1, rate // 6000)
    best = 0.0
    for lag in range(lo, hi + 1, step):
        a = x[:-lag]; b = x[lag:]
        denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b))) + 1e-12
        v = float(np.dot(a, b)) / denom
        if v > best: best = v
    return best

def band_flatness(bands):
    p = np.power(10.0, bands[VOCAL] / 10.0)
    return float(np.exp(np.mean(np.log(p + 1e-20))) / (np.mean(p) + 1e-20))

def frame_feats(x, rate, start, n):
    mono = x[start:start + n]
    if len(mono) < n: return None
    level = 20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-12)
    bands = goertzel_bands(mono, rate)
    power = np.power(10.0, bands / 10.0)
    total = max(power.sum(), 1e-20)
    cent = float((power * BANDS).sum() / total)
    flat = float(np.exp(np.log(np.maximum(power, 1e-20)).mean()) / (total / len(BANDS)))
    index = np.arange(len(BANDS), dtype=float); centered = index - index.mean()
    slope = float((centered * (bands - bands.mean())).sum() / (centered**2).sum())
    low_share = float(power[:5].sum() / total)
    return dict(bands=bands, level=level, pitch=periodicity(mono, rate),
                vflat=band_flatness(bands), cent=cent, flat=flat, slope=slope,
                low_share=low_share, env_lo=float(bands[:5].mean()),
                env_mid=float(bands[5:16].mean()))

def window_rows(path):
    rate, x = load(path)
    hop = rate * FRAME_MS // 1000
    an = min(2560, hop)
    frames = []
    rows = []
    for i in range(len(x) // hop):
        f = frame_feats(x, rate, i * hop + hop - an, an)
        if f is None: continue
        frames.append(f)
        if len(frames) < WINDOW: continue
        recent = frames[-WINDOW:]
        bands = np.mean([f["bands"] for f in recent], axis=0)
        env_lo = np.array([f["env_lo"] for f in recent]); env_mid = np.array([f["env_mid"] for f in recent])
        vector = np.concatenate([
            bands,
            [np.mean([f["level"] for f in recent])],
            [np.median([f["pitch"] for f in recent])],
            [np.median([f["vflat"] for f in recent])],
            [env_mid.std(), env_lo.std(), env_lo.max() - env_lo.min()],
            [np.mean([f["flat"] for f in recent]), np.mean([f["slope"] for f in recent]),
             np.mean([f["low_share"] for f in recent]), np.mean([f["cent"] for f in recent])],
        ])
        rows.append(vector)
    return np.array(rows)

LABELED = [
    (REPO + "/记录/dmic-dataset/quiet-20261109.wav", "quiet"),
    (REPO + "/记录/dmic-dataset/street-20261109.wav", "quiet"),
    (REPO + "/记录/dmic-dataset/near-voice-20261109.wav", "speech"),
    (REPO + "/记录/dmic-dataset/20261109-094825-01-speech.wav", "speech"),
    (REPO + "/记录/dmic-dataset/office-babble-20261109.wav", "speech"),
    (REPO + "/记录/dmic-dataset/20261109-094825-02-keyboard.wav", "keyboard"),
    (REPO + "/记录/dmic-dataset/20261109-094825-03-wind.wav", "wind"),
]

def main():
    X, y, clip = [], [], []
    for path, label in LABELED:
        rows = window_rows(path)
        if len(rows) == 0: continue
        X.append(rows); y += [CLASSES.index(label)] * len(rows); clip += [path.rsplit('/', 1)[-1]] * len(rows)
    X = np.vstack(X); y = np.array(y)
    mean = X.mean(axis=0); std = X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    # temporal 70/30 split inside every clip
    train_idx, test_idx = [], []
    offset = 0
    for rows in [window_rows(p) for p, _ in LABELED]:
        n = len(rows); cut = max(1, int(n * 0.7))
        train_idx += list(range(offset, offset + cut)); test_idx += list(range(offset + cut, offset + n)); offset += n
    W = np.zeros((len(CLASSES), Z.shape[1])); b = np.zeros(len(CLASSES))
    Xtr, ytr = Z[train_idx], y[train_idx]
    lr, lam = 0.3, 1e-2
    for _ in range(1200):
        logits = Xtr @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
        onehot = np.eye(len(CLASSES))[ytr]
        grad = (probs - onehot)
        W -= lr * (grad.T @ Xtr / len(Xtr) + lam * W)
        b -= lr * grad.mean(axis=0)
    def confusion(idx):
        logits = Z[idx] @ W.T + b
        pred = logits.argmax(axis=1)
        table = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        for t, p in zip(y[idx], pred): table[t, p] += 1
        return table, (pred == y[idx]).mean()
    tr_table, tr_acc = confusion(train_idx)
    te_table, te_acc = confusion(test_idx)
    print("classes:", CLASSES)
    print(f"windows: train={len(train_idx)} test={len(test_idx)}")
    print(f"train acc={tr_acc:.3f}  test acc={te_acc:.3f}")
    print("test confusion (rows=true, cols=pred):")
    print("           " + " ".join(f"{c:>9s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>9s}  " + " ".join(f"{te_table[i, j]:9d}" for j in range(len(CLASSES))))
    model = {"schema": 1, "classes": CLASSES, "mean": mean.tolist(), "std": std.tolist(),
             "w": W.tolist(), "b": b.tolist(),
             "note": "temporal 70/30 split; labels from board clips; wind/keyboard clips contaminated by office babble"}
    json.dump(model, open("/tmp/event_lr_v1.json", "w"), indent=1)
    print("saved /tmp/event_lr_v1.json")

if __name__ == "__main__":
    main()

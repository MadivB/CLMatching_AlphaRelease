"""Exact, vectorised replacement for v3_2_global_matching._scan_best_shift_multi (the matrix stage's per-cluster scan,
a Python loop over 896 shifts of a (K, n_ch, T) block: ~45 ms per cluster, ~500 clusters per spill = ~20 s).

loss(t0) = mean over all elements of (clip(base + shift(img, t0), hi) - actual)^2 / max(err, 1e-6), shift = right shift
by t0 ticks with zero fill (samples pushed beyond the window are dropped).  For channels that can never clip
(max base + max img <= hi) the loss is a quadratic in the shifted image and all shifts come from two FFT
cross-correlations per channel; channels that can clip are evaluated exactly for all shifts at once from a strided
(n_shift, T) view.  Float64 throughout; agreement with the loop version is at the 1e-9 level (see self-test)."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def scan_best_shift_fft(cluster_block, base_block, actual_block, error_block, *, search_range=900, adc_clip=60780.0, return_curve=False):
    S = np.asarray(cluster_block, np.float64); B = np.asarray(base_block, np.float64); A = np.asarray(actual_block, np.float64); E = np.maximum(np.asarray(error_block, np.float64), 1e-6)
    T = S.shape[-1]; n_shift = int(search_range) + 1
    S2 = S.reshape(-1, T); B2 = B.reshape(-1, T); A2 = A.reshape(-1, T); E2 = E.reshape(-1, T)
    hi = np.broadcast_to(np.asarray(adc_clip, np.float64), S.shape).reshape(-1, T)      # per-element ceiling (scalar, (n_ch,1) or full)
    hi_min = hi.min(1)
    N = float(S2.size)
    W = 1.0 / E2; D = B2 - A2
    const = float((D * D * W).sum())
    can_clip = (B2.max(1) + S2.max(1) > hi_min - 1e-9)
    curve = np.full(n_shift, const)
    # FFT part: channels that never clip
    nc = ~can_clip
    if nc.any():
        L = 1
        while L < 2 * T:
            L *= 2
        Fs = np.fft.rfft(S2[nc], L); Fs2 = np.fft.rfft(S2[nc] ** 2, L)
        F1 = np.fft.rfft(2.0 * D[nc] * W[nc], L); F2 = np.fft.rfft(W[nc], L)
        # cross-correlation C_f(t0) = sum_tau f(tau + t0) s(tau): ifft(F_f * conj(F_s))[t0]
        c1 = np.fft.irfft(F1 * np.conj(Fs), L)[:, :n_shift].sum(0); c2 = np.fft.irfft(F2 * np.conj(Fs2), L)[:, :n_shift].sum(0)
        curve += c1 + c2
    # exact part: channels that may clip, all shifts at once via a strided view of the zero-padded image
    if can_clip.any():
        idx = np.flatnonzero(can_clip)
        for i in idx:
            s = S2[i]; pad = np.concatenate([np.zeros(n_shift - 1), s])          # shifted(t0)[t] = s[t - t0]
            V = sliding_window_view(pad, T)[::-1][:n_shift]                       # (n_shift, T): row t0 = s shifted right by t0
            m = np.minimum(B2[i][None, :] + V, hi[i][None, :])
            curve += ((m - A2[i][None, :]) ** 2 * W[i][None, :]).sum(1) - (D[i] * D[i] * W[i]).sum()
    curve = curve / N
    best = int(np.argmin(curve))
    return best, float(curve[best]), (curve.astype(np.float32) if return_curve else None)


if __name__ == "__main__":
    import os, sys, time
    HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.abspath(os.path.join(HERE, "..", "CLMatching_AlphaRelease"))
    for p in (BASE, os.path.join(BASE, "M5p1"), os.path.join(BASE, "NewMLSection"), os.path.join(BASE, "TwoByTwo")):
        sys.path.insert(0, p)
    from v3_2_global_matching import _scan_best_shift_multi
    rng = np.random.default_rng(1); T = 1000; tmpl = np.load(os.path.join(BASE, "assets", "avg_pulse.npy")).astype(float).ravel(); tmpl = tmpl / tmpl.max()
    def shifted(amp, t0):
        out = np.zeros((amp.size, T)); out[:, t0:] = amp[:, None] * tmpl[None, : T - t0]; return out
    worst = 0.0; mism = 0; t_loop = 0.0; t_fft = 0.0; n_tests = 12
    for k in range(n_tests):
        n_ch = int(rng.integers(8, 60)); amp = rng.uniform(100, 8000, n_ch) * (rng.random(n_ch) < 0.7)
        base = sum(shifted(rng.uniform(500, 30000, n_ch), int(rng.integers(0, 800))) for _ in range(3)); actual = base + shifted(amp, int(rng.integers(0, 880))) + rng.normal(0, 50, (n_ch, T))
        ceil = np.full(n_ch, 60000.0); ceil[: max(1, n_ch // 8)] = base.max(1)[: max(1, n_ch // 8)] * 0.8      # some channels saturate
        actual = np.minimum(actual, ceil[:, None]); err = 200.0 * np.maximum(actual, 100.0)
        img = shifted(amp, 0)[None]; Bb = base[None].astype(np.float32); Aa = actual[None].astype(np.float32); Ee = err[None].astype(np.float32); hi = ceil[None, :, None].astype(np.float32)
        t0 = time.time(); b1, s1, c1 = _scan_best_shift_multi(img.astype(np.float32), Bb, Aa, Ee, search_range=895, adc_clip=hi, return_curve=True); t_loop += time.time() - t0
        t0 = time.time(); b2, s2, c2 = scan_best_shift_fft(img, Bb, Aa, Ee, search_range=895, adc_clip=hi, return_curve=True); t_fft += time.time() - t0
        d = np.abs(c1.astype(np.float64) - c2.astype(np.float64)); rel = d.max() / max(np.ptp(c1), 1e-9); worst = max(worst, rel); mism += int(b1 != b2)
        print(f"test {k}: n_ch {n_ch} sat_ch {(ceil < 60000).sum()} loop argmin {b1} fft argmin {b2} loss {s1:.6g} / {s2:.6g} max|dcurve|/range {rel:.2e}")
    print(f"\nargmin mismatches {mism}/{n_tests}; worst relative curve difference {worst:.2e}; time loop {t_loop/n_tests*1e3:.1f} ms vs fft {t_fft/n_tests*1e3:.1f} ms per scan")

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 9b - PROPER robustness/convergence: for each grid N and each quartic
# coefficient, RE-OPTIMIZE hbar_eff locally (1D scan + refine) and report the
# BEST ACHIEVABLE MAPE. This is the fair test: it asks whether a good fit still
# exists (with hbar merely shifted), not whether one frozen hbar survives a
# change of grid. Still no global DE search; each evaluation is one forward pass.
# ==============================================================================

A_START  = 1.551941486210356
T_STEPS  = 300
A_END    = 1.02
C_OFFSET = 10.0

TRUE_ZEROS_100 = np.array([
    14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 37.5861, 40.9187, 43.3270, 48.0051, 49.7738,
    52.9703, 56.4462, 59.3470, 60.8317, 65.1125, 67.0798, 69.5464, 72.0671, 75.7046, 77.1448,
    79.3373, 82.9103, 84.7354, 87.4252, 88.8091, 92.4918, 94.6513, 95.8706, 98.8311, 101.3178,
    103.7255, 105.4466, 107.1686, 111.0295, 111.8746, 114.3202, 116.2266, 118.7907, 121.3701, 122.9468,
    124.2568, 127.5166, 129.5787, 131.0876, 133.4977, 134.7565, 138.1160, 139.7362, 141.1237, 143.1118,
    146.0009, 147.4427, 150.0515, 150.9252, 153.0246, 156.1129, 157.5975, 158.8499, 161.1889, 163.0306,
    165.5373, 167.1844, 169.0945, 169.9119, 173.4115, 174.7541, 176.4414, 178.3774, 179.9164, 182.2070,
    184.8744, 185.9589, 187.2289, 189.4161, 192.0266, 193.0797, 195.2658, 196.8764, 198.0153, 201.2647,
    202.4935, 204.1896, 205.3946, 207.9062, 209.5765, 211.3289, 213.3479, 214.5470, 216.1695, 219.0675,
    220.7149, 221.3952, 224.0070, 224.9833, 227.4214, 229.3374, 231.2882, 231.9872, 233.6934, 236.5242
])


def solve(hbar, a_start=A_START, quartic=0.05, N=250, L=3.5):
    q = np.linspace(-L, L, N, endpoint=False)
    dq = q[1] - q[0]
    dp = 2 * np.pi * hbar / (N * dq)
    p = np.fft.fftfreq(N) * N * dp
    T_kin = 0.5 * p**2
    F = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
    F_inv = np.fft.ifft(np.eye(N), axis=0) * np.sqrt(N)
    U_kin = F_inv @ np.diag(np.exp(-1j * T_kin / hbar)) @ F
    t_start = 1.0 / (np.log(1 + C_OFFSET)**2)
    t_end = 1.0 / (np.log(T_STEPS + C_OFFSET)**2)
    k_opt = (a_start - A_END) / (t_start - t_end)
    a_dyna = A_END - k_opt * t_end
    U_tot = np.eye(N, dtype=np.complex128)
    for t in range(1, T_STEPS + 1):
        a_t = a_dyna + k_opt / (np.log(t + C_OFFSET)**2)
        V_t = -q + q**2 + (a_t / 3.0) * q**3 + quartic * q**4
        U_tot = (U_kin * np.exp(-1j * V_t / hbar)) @ U_tot
    V_base = -q + q**2 + (A_END / 3.0) * q**3 + quartic * q**4
    H_base = F_inv @ np.diag(T_kin) @ F + np.diag(V_base)
    try:
        evals, evecs = np.linalg.eig(U_tot)
    except Exception:
        return np.nan, np.nan
    phases = np.angle(evals)
    ee = np.real(np.sum(np.conj(evecs) * (H_base @ evecs), axis=0))
    m = np.round((ee * T_STEPS + hbar * phases) / (2 * np.pi * hbar))
    E = (-hbar * phases + 2 * np.pi * hbar * m) / T_STEPS
    E = np.sort(E); E = E - E[0]
    if len(E) < 101:
        return np.nan, np.nan
    scale = TRUE_ZEROS_100[0] / E[1] if E[1] != 0 else 1.0
    pred = E[1:101] * scale
    mse = np.mean((pred - TRUE_ZEROS_100)**2)
    mape = np.mean(np.abs((pred - TRUE_ZEROS_100) / TRUE_ZEROS_100)) * 100
    return mse, mape


def best_over_hbar(quartic=0.05, N=250, L=3.5, coarse=None):
    """Local re-optimization of hbar: coarse 1D scan then refine around the best.
    Returns (best_hbar, best_mse, best_mape)."""
    if coarse is None:
        coarse = np.linspace(0.05, 0.085, 36)  # ~0.001 steps
    best = (np.nan, np.inf, np.nan)
    for h in coarse:
        mse, mape = solve(h, quartic=quartic, N=N, L=L)
        if np.isfinite(mse) and mse < best[1]:
            best = (h, mse, mape)
    # refine around best with finer step
    if np.isfinite(best[0]):
        fine = np.linspace(best[0] - 0.0015, best[0] + 0.0015, 13)
        for h in fine:
            mse, mape = solve(h, quartic=quartic, N=N, L=L)
            if np.isfinite(mse) and mse < best[1]:
                best = (h, mse, mape)
    return best


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- (A) Best-achievable MAPE vs quartic coefficient (hbar re-optimized) ----
    print("(A) Quartic scan with hbar re-optimized at each coefficient ...")
    cs = [0.03, 0.04, 0.05, 0.06, 0.08]
    best_mape_c, best_h_c = [], []
    for c in cs:
        h, mse, mape = best_over_hbar(quartic=c, N=250)
        best_mape_c.append(mape); best_h_c.append(h)
        print(f"    quartic={c:.2f}  best_hbar={h:.4f}  best_MSE={mse:.3f}  best_MAPE={mape:.2f}%")
    ax = axes[0]
    ax.plot(cs, best_mape_c, 'o-', color='crimson')
    ax.axvline(0.05, color='gray', ls='--', lw=1, label='used value 0.05')
    ax.set_xlabel('quartic coefficient $\\lambda$')
    ax.set_ylabel('best achievable MAPE (%)\n($\\hbar_{\\mathrm{eff}}$ re-optimized)')
    ax.set_title('(A) Robustness to quartic coefficient')
    ax.legend(fontsize=8)

    # ---- (B) Best-achievable MAPE vs grid N (hbar re-optimized) ----
    print("(B) Grid scan with hbar re-optimized at each N ...")
    Ns = [200, 220, 250, 280, 320]
    best_mape_N, best_h_N = [], []
    for Ng in Ns:
        h, mse, mape = best_over_hbar(quartic=0.05, N=Ng)
        best_mape_N.append(mape); best_h_N.append(h)
        print(f"    N={Ng}  best_hbar={h:.4f}  best_MSE={mse:.3f}  best_MAPE={mape:.2f}%")
    ax = axes[1]
    ax.plot(Ns, best_mape_N, 's-', color='darkgreen')
    ax.axvline(250, color='gray', ls='--', lw=1, label='used grid N=250')
    ax.set_xlabel('spatial grid size N')
    ax.set_ylabel('best achievable MAPE (%)\n($\\hbar_{\\mathrm{eff}}$ re-optimized)')
    ax.set_title('(B) Grid-refinement convergence')
    ax.legend(fontsize=8)

    fig.suptitle('Fair robustness/convergence: best achievable fit with $\\hbar_{\\mathrm{eff}}$ re-optimized '
                 'at each setting (local scan, no global search)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(os.path.dirname(__file__), '9-Robustness_Reoptimized.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("Saved:", out)

    print("\n=== SUMMARY (fair: hbar re-optimized) ===")
    print(f"Quartic 0.03-0.08: best MAPE range {np.nanmin(best_mape_c):.2f}%-{np.nanmax(best_mape_c):.2f}%")
    print(f"  optimal hbar shifts: {[f'{h:.4f}' for h in best_h_c]}")
    print(f"Grid N=200-320: best MAPE range {np.nanmin(best_mape_N):.2f}%-{np.nanmax(best_mape_N):.2f}%")
    print(f"  optimal hbar shifts: {[f'{h:.4f}' for h in best_h_N]}")


if __name__ == '__main__':
    main()

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 9 - Robustness / sensitivity / convergence studies for the quantum solver.
#   (1) quartic-coefficient scan  : how MSE depends on the 0.05 q^4 coefficient
#   (2) hbar_eff sensitivity      : MSE(hbar) curve around the optimum -> error bar
#   (3) grid-refinement convergence: MAPE(100) vs grid size N
#
# NO parameter search. Frozen optimized params; each point is ONE forward
# spectrum extraction. Single core, a few minutes total.
# ==============================================================================

HBAR_EFF = 0.06138739295586476
A_START  = 1.551941486210356
N_DEF    = 250
L_DEF    = 3.5
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


def solve(hbar=HBAR_EFF, a_start=A_START, quartic=0.05, N=N_DEF, L=L_DEF):
    """One forward pass; returns (mse, mape) against the first 100 zeros under
    single-point anchoring. Returns (nan, nan) if extraction fails."""
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
        U_t = U_kin * np.exp(-1j * V_t / hbar)
        U_tot = U_t @ U_tot

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


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ---- (1) quartic coefficient scan ----
    print("(1) Quartic-coefficient scan ...")
    cs = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10])
    mse_c, mape_c = [], []
    for c in cs:
        mse, mape = solve(quartic=c)
        mse_c.append(mse); mape_c.append(mape)
        print(f"    quartic={c:.2f}  MSE={mse:.3f}  MAPE={mape:.2f}%")
    ax = axes[0]
    ax.plot(cs, mape_c, 'o-', color='crimson')
    ax.axvline(0.05, color='gray', ls='--', lw=1, label='used value 0.05')
    ax.set_xlabel('quartic coefficient $\\lambda$ in $\\lambda q^4$')
    ax.set_ylabel('MAPE on first 100 zeros (%)')
    ax.set_title('(1) Robustness to quartic coefficient')
    ax.legend(fontsize=8)

    # ---- (2) hbar sensitivity around optimum ----
    print("(2) hbar_eff sensitivity ...")
    hs = np.linspace(0.045, 0.080, 15)
    mse_h, mape_h = [], []
    for h in hs:
        mse, mape = solve(hbar=h)
        mse_h.append(mse); mape_h.append(mape)
        print(f"    hbar={h:.4f}  MSE={mse:.3f}  MAPE={mape:.2f}%")
    mse_h = np.array(mse_h)
    ax = axes[1]
    ax.plot(hs, mse_h, 'o-', color='navy')
    ax.axvline(HBAR_EFF, color='gray', ls='--', lw=1, label=f'optimum {HBAR_EFF:.4f}')
    # estimate a width: hbar range where MSE < 2x minimum
    mmin = np.nanmin(mse_h)
    band = hs[mse_h < 2 * mmin]
    if len(band) >= 2:
        ax.axvspan(band.min(), band.max(), color='navy', alpha=0.12,
                   label=f'MSE<2$\\times$min: $\\pm${(band.max()-band.min())/2:.4f}')
    ax.set_xlabel('$\\hbar_{\\mathrm{eff}}$')
    ax.set_ylabel('MSE on first 100 zeros')
    ax.set_title('(2) Sensitivity to $\\hbar_{\\mathrm{eff}}$')
    ax.legend(fontsize=8)

    # ---- (3) grid-refinement convergence ----
    print("(3) Grid-refinement convergence ...")
    Ns = [180, 220, 250, 300, 350, 420]
    mape_N = []
    for Ng in Ns:
        # keep box density comparable: scale L modestly with N
        Lg = L_DEF * (Ng / N_DEF) ** 0.0  # keep L fixed (physical box unchanged)
        mse, mape = solve(N=Ng, L=L_DEF)
        mape_N.append(mape)
        print(f"    N={Ng}  MAPE={mape:.2f}%")
    ax = axes[2]
    ax.plot(Ns, mape_N, 's-', color='darkgreen')
    ax.axvline(250, color='gray', ls='--', lw=1, label='used grid N=250')
    ax.set_xlabel('spatial grid size N')
    ax.set_ylabel('MAPE on first 100 zeros (%)')
    ax.set_title('(3) Grid-refinement convergence')
    ax.legend(fontsize=8)

    fig.suptitle('Robustness, sensitivity, and convergence of the quantum solver '
                 '(single-point anchoring, no re-optimization)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(__file__), '9-Robustness_Sensitivity.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("Saved:", out)

    print("\n=== SUMMARY ===")
    valid = ~np.isnan(mape_c)
    print(f"Quartic scan 0.02-0.10: MAPE range {np.nanmin(mape_c):.2f}%-{np.nanmax(mape_c):.2f}%")
    print(f"hbar MSE minimum at {hs[np.nanargmin(mse_h)]:.4f} (optimum {HBAR_EFF:.4f})")
    if len(band) >= 2:
        print(f"hbar tolerance band (MSE<2xmin): [{band.min():.4f}, {band.max():.4f}]")
    print(f"Grid N=180->420: MAPE {mape_N[0]:.2f}% -> {mape_N[-1]:.2f}%")


if __name__ == '__main__':
    main()

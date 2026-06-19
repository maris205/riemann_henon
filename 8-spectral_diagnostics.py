import os
# Keep BLAS single-threaded: this VM has 1 physical core.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 8 - Standard spectral diagnostics (NNSD, number variance, spectral rigidity,
#     pair correlation) for the area-preserving Henon-map model spectrum,
#     compared to GUE / Poisson and to the true Riemann zeros.
#
# IMPORTANT: this script does NOT run any parameter search. It uses the already
# optimized parameters (hbar_eff, a_start) found in 6-henon_micro_param_scan_100
# and simply does ONE forward spectrum extraction, then computes diagnostics.
# Runs in a few seconds on a single core.
# ==============================================================================

# ---- Frozen, previously-optimized parameters (see 6-..._100_zeros.log) -------
HBAR_EFF = 0.06138739295586476
A_START  = 1.551941486210356
N        = 250
L        = 3.5
T_STEPS  = 300
A_END    = 1.02
C_OFFSET = 10.0

# First 100 nontrivial Riemann zero ordinates (ground truth reference).
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


def build_floquet(hbar=HBAR_EFF, a_start=A_START):
    """Build the one-period Floquet propagator U for the time-dependent,
    time-reversal-breaking drive a(t). Its EIGENPHASES are the appropriate
    spectral object for level-statistics: a periodically-driven (Floquet)
    chaotic system is the discrete-time analogue of a GUE Hamiltonian.
    One forward pass; no optimization."""
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
        V_t = -q + q**2 + (a_t / 3.0) * q**3 + 0.05 * q**4
        U_t = U_kin * np.exp(-1j * V_t / hbar)
        U_tot = U_t @ U_tot
    return U_tot


def extract_floquet_phases(drop_edge=50):
    """Eigenphases of the Floquet operator, sorted on the unit circle.
    Edge phases are dropped to avoid the few states pushed against the
    grid/box cutoff. These phases are the spectral object whose fluctuation
    statistics we compare to GUE."""
    U = build_floquet()
    theta = np.sort(np.angle(np.linalg.eigvals(U)))
    if drop_edge > 0:
        theta = theta[drop_edge:-drop_edge]
    return theta


# ------------------------------------------------------------------------------
# Unfolding: remove the smooth (global) density so the mean spacing is 1.
# We use a polynomial fit to each spectrum's own staircase (standard empirical
# local unfolding). This is exactly the step the reviewers emphasized: GUE
# statistics describe the UNFOLDED spectrum, not the raw counting function.
# ------------------------------------------------------------------------------
def unfold(levels, deg=6):
    levels = np.sort(np.asarray(levels, dtype=float))
    staircase = np.arange(1, len(levels) + 1)
    coeffs = np.polyfit(levels, staircase, deg)
    unfolded = np.polyval(coeffs, levels)
    return unfolded


def nn_spacings(unfolded):
    s = np.diff(np.sort(unfolded))
    return s / np.mean(s)  # enforce unit mean exactly


# ---- Reference curves --------------------------------------------------------
def wigner_gue(s):
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


def poisson(s):
    return np.exp(-s)


def number_variance(unfolded, Lvals):
    """Sigma^2(L): variance of the count of unfolded levels in windows of width L,
    averaged over window positions sliding across the spectrum."""
    u = np.sort(unfolded)
    span = u[-1] - u[0]
    out = []
    for Lw in Lvals:
        if Lw >= span:
            out.append(np.nan)
            continue
        # slide window center over available range; sample densely
        starts = np.linspace(u[0], u[-1] - Lw, 400)
        counts = np.array([np.sum((u >= a) & (u < a + Lw)) for a in starts])
        out.append(np.var(counts))
    return np.array(out)


def spectral_rigidity_from_sigma2(Lvals, sigma2):
    """Dyson-Mehta Delta_3(L) computed from Sigma^2 via the standard integral
    Delta_3(L) = (2/L^4) \int_0^L (L^3 - 2 L^2 r + r^3) Sigma^2(r) dr."""
    Lvals = np.asarray(Lvals, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    d3 = np.full_like(Lvals, np.nan)
    for i, Lw in enumerate(Lvals):
        mask = (Lvals <= Lw) & np.isfinite(sigma2)
        if mask.sum() < 3:
            continue
        r = Lvals[mask]
        s2 = sigma2[mask]
        kernel = (Lw**3 - 2 * Lw**2 * r + r**3)
        d3[i] = (2.0 / Lw**4) * np.trapezoid(kernel * s2, r)
    return d3


def sigma2_gue(Lw):
    g = 0.5772156649015329  # Euler-Mascheroni
    return (1.0 / np.pi**2) * (np.log(2 * np.pi * Lw) + g + 1.0)


def delta3_gue(Lw):
    g = 0.5772156649015329
    return (1.0 / np.pi**2) * (np.log(2 * np.pi * Lw) + g - 5.0/4.0 - np.pi**2 / 8.0)


def pair_correlation(unfolded, rmax=3.0, nbins=30):
    """Two-level correlation R2(r) of the unfolded spectrum (histogram of all
    pairwise gaps), normalized to fluctuate around 1."""
    u = np.sort(unfolded)
    diffs = []
    for i in range(len(u)):
        d = u[i+1:] - u[i]
        d = d[d < rmax]
        diffs.extend(d.tolist())
    diffs = np.array(diffs)
    edges = np.linspace(0, rmax, nbins + 1)
    hist, _ = np.histogram(diffs, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    n = len(u)
    # expected pairs per bin for uncorrelated (density 1): n * width (per ref point)
    expected = n * width
    R2 = hist / expected
    return centers, R2


def R2_gue(r):
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    nz = r != 0
    s = np.pi * r[nz]
    out[nz] = 1.0 - (np.sin(s) / s)**2
    return out


# ==============================================================================
def main():
    print("Extracting Floquet eigenphases (single forward pass, no optimization)...")
    theta = extract_floquet_phases(drop_edge=50)
    print(f"  Floquet phases kept: {len(theta)}")

    # Floquet phases sit on the unit circle with uniform mean density, so the
    # 'unfolding' is just a linear rescale to unit mean spacing.
    n = len(theta)
    u_model = (theta - theta[0]) * (n - 1) / (theta[-1] - theta[0])
    # True zeros: standard polynomial unfolding of the staircase.
    u_zeros = unfold(TRUE_ZEROS_100)

    s_model = nn_spacings(u_model)
    s_zeros = nn_spacings(u_zeros)

    # --- Figure: 2x2 panel of diagnostics ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) NNSD
    ax = axes[0, 0]
    sgrid = np.linspace(0, 3.5, 300)
    bins = np.linspace(0, 3.5, 16)
    ax.hist(s_zeros, bins=bins, density=True, alpha=0.45, color='black',
            label='Riemann zeros (first 100)')
    ax.hist(s_model, bins=bins, density=True, alpha=0.45, color='crimson',
            label='Henon Floquet phases')
    ax.plot(sgrid, wigner_gue(sgrid), 'b-', lw=2, label='GUE (Wigner surmise)')
    ax.plot(sgrid, poisson(sgrid), 'g--', lw=1.5, label='Poisson')
    ax.set_xlabel('s (unfolded spacing)'); ax.set_ylabel('P(s)')
    ax.set_title('(a) Nearest-neighbor spacing distribution')
    ax.legend(fontsize=8)

    # (b) Pair correlation
    ax = axes[0, 1]
    c_m, R2_m = pair_correlation(u_model)
    c_z, R2_z = pair_correlation(u_zeros)
    rgrid = np.linspace(1e-3, 3.0, 300)
    ax.plot(c_z, R2_z, 'ko-', ms=3, label='Riemann zeros')
    ax.plot(c_m, R2_m, 'r^-', ms=3, label='Henon Floquet')
    ax.plot(rgrid, R2_gue(rgrid), 'b-', lw=2, label='GUE  $1-\\mathrm{sinc}^2(\\pi r)$')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('r (unfolded separation)'); ax.set_ylabel('$R_2(r)$')
    ax.set_title('(b) Two-level (pair) correlation')
    ax.legend(fontsize=8)

    # (c) Number variance
    ax = axes[1, 0]
    Lvals = np.linspace(0.5, 12, 24)
    s2_model = number_variance(u_model, Lvals)
    s2_zeros = number_variance(u_zeros, Lvals)
    ax.plot(Lvals, s2_zeros, 'ko-', ms=3, label='Riemann zeros')
    ax.plot(Lvals, s2_model, 'r^-', ms=3, label='Henon Floquet')
    ax.plot(Lvals, sigma2_gue(Lvals), 'b-', lw=2, label='GUE')
    ax.plot(Lvals, Lvals, 'g--', lw=1.2, label='Poisson ($\\Sigma^2=L$)')
    ax.set_xlabel('L'); ax.set_ylabel('$\\Sigma^2(L)$')
    ax.set_title('(c) Number variance')
    ax.legend(fontsize=8)

    # (d) Spectral rigidity
    ax = axes[1, 1]
    d3_model = spectral_rigidity_from_sigma2(Lvals, s2_model)
    d3_zeros = spectral_rigidity_from_sigma2(Lvals, s2_zeros)
    ax.plot(Lvals, d3_zeros, 'ko-', ms=3, label='Riemann zeros')
    ax.plot(Lvals, d3_model, 'r^-', ms=3, label='Henon Floquet')
    ax.plot(Lvals, delta3_gue(Lvals), 'b-', lw=2, label='GUE')
    ax.plot(Lvals, Lvals / 15.0, 'g--', lw=1.2, label='Poisson ($L/15$)')
    ax.set_xlabel('L'); ax.set_ylabel('$\\Delta_3(L)$')
    ax.set_title('(d) Spectral rigidity (Dyson-Mehta)')
    ax.legend(fontsize=8)

    fig.suptitle('Standard spectral diagnostics: Henon-map Floquet phases vs. Riemann zeros vs. GUE / Poisson\n'
                 '(unfolded; first 100 levels, limited-statistics regime)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(os.path.dirname(__file__), '8-Spectral_Diagnostics.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")

    # --- Quantitative summary (KS-like distance of NNSD to GUE) ---
    def cdf_dist(s, ref_cdf):
        ss = np.sort(s)
        emp = np.arange(1, len(ss)+1) / len(ss)
        return np.max(np.abs(emp - ref_cdf(ss)))

    # GUE Wigner-surmise CDF via numerical integration
    grid = np.linspace(0, 6, 6000)
    pdf = wigner_gue(grid)
    cdf = np.cumsum(pdf) * (grid[1]-grid[0])
    cdf /= cdf[-1]
    gue_cdf = lambda x: np.interp(x, grid, cdf)
    pois_cdf = lambda x: 1 - np.exp(-x)

    print("\n=== Nearest-neighbor spacing: max-CDF-distance (smaller = closer) ===")
    print(f"  zeros vs GUE     : {cdf_dist(s_zeros, gue_cdf):.4f}")
    print(f"  zeros vs Poisson : {cdf_dist(s_zeros, pois_cdf):.4f}")
    print(f"  model vs GUE     : {cdf_dist(s_model, gue_cdf):.4f}")
    print(f"  model vs Poisson : {cdf_dist(s_model, pois_cdf):.4f}")
    print(f"\n  mean spacing (model) = {np.mean(np.diff(np.sort(u_model))):.4f}")
    print(f"  <s> zeros={np.mean(s_zeros):.3f}  <s^2> zeros={np.mean(s_zeros**2):.3f} "
          f"(GUE <s^2>=1.273, Poisson=2.0)")
    print(f"  <s> model={np.mean(s_model):.3f}  <s^2> model={np.mean(s_model**2):.3f}")


if __name__ == '__main__':
    main()

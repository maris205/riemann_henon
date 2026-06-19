import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np

# Explore which spectral OBJECT (if any) shows GUE-type level repulsion.
# No parameter search; frozen params. Fast.

HBAR_EFF = 0.06138739295586476
A_START  = 1.551941486210356
N        = 250
L        = 3.5
T_STEPS  = 300
A_END    = 1.02
C_OFFSET = 10.0

g_euler = 0.5772156649015329

def build_static_H(a=A_END, n=N, Lbox=L, hbar=HBAR_EFF):
    q = np.linspace(-Lbox, Lbox, n, endpoint=False)
    dq = q[1]-q[0]
    dp = 2*np.pi*hbar/(n*dq)
    p = np.fft.fftfreq(n)*n*dp
    T_kin = 0.5*p**2
    F = np.fft.fft(np.eye(n), axis=0)/np.sqrt(n)
    F_inv = np.fft.ifft(np.eye(n), axis=0)*np.sqrt(n)
    V = -q + q**2 + (a/3.0)*q**3 + 0.05*q**4
    H = F_inv @ np.diag(T_kin) @ F + np.diag(V)
    return H

def build_floquet():
    q = np.linspace(-L, L, N, endpoint=False)
    dq = q[1]-q[0]
    dp = 2*np.pi*HBAR_EFF/(N*dq)
    p = np.fft.fftfreq(N)*N*dp
    T_kin = 0.5*p**2
    F = np.fft.fft(np.eye(N), axis=0)/np.sqrt(N)
    F_inv = np.fft.ifft(np.eye(N), axis=0)*np.sqrt(N)
    U_kin = F_inv @ np.diag(np.exp(-1j*T_kin/HBAR_EFF)) @ F
    t_start = 1.0/(np.log(1+C_OFFSET)**2)
    t_end = 1.0/(np.log(T_STEPS+C_OFFSET)**2)
    k_opt = (A_START-A_END)/(t_start-t_end)
    a_dyna = A_END - k_opt*t_end
    U = np.eye(N, dtype=np.complex128)
    for t in range(1, T_STEPS+1):
        a_t = a_dyna + k_opt/(np.log(t+C_OFFSET)**2)
        V_t = -q + q**2 + (a_t/3.0)*q**3 + 0.05*q**4
        U = (U_kin*np.exp(-1j*V_t/HBAR_EFF)) @ U
    return U

def unfold_poly(levels, deg=7):
    levels = np.sort(np.asarray(levels, float))
    st = np.arange(1, len(levels)+1)
    c = np.polyfit(levels, st, deg)
    return np.polyval(c, levels)

def nnsd(unf):
    s = np.diff(np.sort(unf)); return s/np.mean(s)

def wigner_gue(s): return (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)

def dist_to(s):
    grid = np.linspace(0,6,6000)
    pdf = wigner_gue(grid); cdf=np.cumsum(pdf)*(grid[1]-grid[0]); cdf/=cdf[-1]
    gue = lambda x: np.interp(x,grid,cdf); pois=lambda x: 1-np.exp(-x)
    ss=np.sort(s); emp=np.arange(1,len(ss)+1)/len(ss)
    dg=np.max(np.abs(emp-gue(ss))); dp=np.max(np.abs(emp-pois(ss)))
    return dg, dp, np.mean(s**2)

def report(name, s):
    dg,dp,s2 = dist_to(s)
    verdict = 'GUE-like' if dg<dp else 'Poisson-like'
    print(f"{name:45s} n={len(s):3d}  d_GUE={dg:.3f} d_Pois={dp:.3f} <s^2>={s2:.3f}  -> {verdict}")

print("GUE <s^2>=1.273, Poisson <s^2>=2.0\n")

# (A) static H @ a=1.02, first 100 eigenvalues (1D -> expect integrable)
H = build_static_H()
ev = np.sort(np.linalg.eigvalsh(0.5*(H+H.conj().T)).real)[:120]
report("A) static H(a=1.02), lowest 100 eig", nnsd(unfold_poly(ev[:100])))

# (B) static H but bound states only (below barrier) - take lowest 60
report("B) static H, lowest 60 eig", nnsd(unfold_poly(ev[:60])))

# (C) Floquet eigenphases of U_tot, ALL N, circle-unfolded
U = build_floquet()
th = np.sort(np.angle(np.linalg.eigvals(U)))
s_phase = np.diff(th); s_phase = s_phase/np.mean(s_phase)
report("C) Floquet eigenphases (all N, raw)", s_phase)

# (D) Floquet eigenphases, drop edges (keep central 150)
th2 = th[50:-50]
sp2 = np.diff(th2); sp2=sp2/np.mean(sp2)
report("D) Floquet eigenphases (central 150)", sp2)

# (E) static H @ a=1.02 with LARGER box/grid (more genuine levels)
H2 = build_static_H(n=400, Lbox=4.5)
ev2 = np.sort(np.linalg.eigvalsh(0.5*(H2+H2.conj().T)).real)[:150]
report("E) static H, N=400 grid, lowest 120", nnsd(unfold_poly(ev2[:120])))

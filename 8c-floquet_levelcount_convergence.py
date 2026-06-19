import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

# Convergence of the Floquet-phase GUE signature with the number of levels.
# No parameter search; frozen optimized params. Vary only grid size N (=> more
# eigenphases) and box L. Pure forward passes.

HBAR_EFF = 0.06138739295586476
A_START  = 1.551941486210356
T_STEPS  = 300
A_END    = 1.02
C_OFFSET = 10.0

def floquet_phases(N, L, hbar=HBAR_EFF, a_start=A_START):
    q = np.linspace(-L, L, N, endpoint=False)
    dq = q[1]-q[0]
    dp = 2*np.pi*hbar/(N*dq)
    p = np.fft.fftfreq(N)*N*dp
    T_kin = 0.5*p**2
    F = np.fft.fft(np.eye(N), axis=0)/np.sqrt(N)
    F_inv = np.fft.ifft(np.eye(N), axis=0)*np.sqrt(N)
    U_kin = F_inv @ np.diag(np.exp(-1j*T_kin/hbar)) @ F
    t_start = 1.0/(np.log(1+C_OFFSET)**2)
    t_end = 1.0/(np.log(T_STEPS+C_OFFSET)**2)
    k_opt = (a_start-A_END)/(t_start-t_end)
    a_dyna = A_END - k_opt*t_end
    U = np.eye(N, dtype=np.complex128)
    for t in range(1, T_STEPS+1):
        a_t = a_dyna + k_opt/(np.log(t+C_OFFSET)**2)
        V_t = -q + q**2 + (a_t/3.0)*q**3 + 0.05*q**4
        U = (U_kin*np.exp(-1j*V_t/hbar)) @ U
    return np.sort(np.angle(np.linalg.eigvals(U)))

def wigner_gue(s): return (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)

def dist(s):
    grid=np.linspace(0,6,6000); pdf=wigner_gue(grid); cdf=np.cumsum(pdf)*(grid[1]-grid[0]); cdf/=cdf[-1]
    gue=lambda x:np.interp(x,grid,cdf); pois=lambda x:1-np.exp(-x)
    ss=np.sort(s); emp=np.arange(1,len(ss)+1)/len(ss)
    return np.max(np.abs(emp-gue(ss))), np.max(np.abs(emp-pois(ss))), np.mean(s**2)

print("Convergence of Floquet-phase NNSD toward GUE with level count")
print("GUE <s^2>=1.273, Poisson=2.0\n")
print(f"{'N':>5} {'L':>5} {'#levels':>8} {'d_GUE':>7} {'d_Pois':>7} {'<s^2>':>7}  verdict")
for N, L in [(250,3.5),(400,4.0),(600,4.5),(800,5.0)]:
    th = floquet_phases(N, L)
    th = th[40:-40]  # drop edge states near the box cutoff
    s = np.diff(th); s/=np.mean(s)
    dg, dp, s2 = dist(s)
    v = 'GUE' if dg<dp else 'Poisson'
    print(f"{N:>5} {L:>5.1f} {len(s):>8} {dg:>7.3f} {dp:>7.3f} {s2:>7.3f}  {v}")

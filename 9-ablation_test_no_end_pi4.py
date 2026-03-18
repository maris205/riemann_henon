import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import time
import multiprocessing
from scipy.optimize import differential_evolution

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

global_start_time = 0
generation = 0
best_mse_global = float('inf')

def holy_grail_objective_3d_nocutoff(x):
    hbar, a_start, a_end = x
    
    if hbar <= 0.02 or hbar >= 0.25 or a_end < 0.90 or a_end > 1.15:
        return 1e6
    if a_start <= a_end + 0.05:
        return 1e6 + (a_end - a_start) * 1000

    N = 250         
    L = 3.5         
    T_steps = 300   
    c_offset = 10.0 
    
    q = np.linspace(-L, L, N, endpoint=False)
    dq = q[1] - q[0]
    dp = 2 * np.pi * hbar / (N * dq)
    p = np.fft.fftfreq(N) * N * dp
    
    T_kin = 0.5 * p**2
    F = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
    F_inv = np.fft.ifft(np.eye(N), axis=0) * np.sqrt(N)
    U_kin = F_inv @ np.diag(np.exp(-1j * T_kin / hbar)) @ F
    
    t_start = 1.0 / (np.log(1 + c_offset)**2)
    t_end   = 1.0 / (np.log(T_steps + c_offset)**2)
    k_opt = (a_start - a_end) / (t_start - t_end)
    a_dyna = a_end - k_opt * t_end  
    
    U_tot = np.eye(N, dtype=np.complex128)
    
    for t in range(1, T_steps + 1):
        a_t = a_dyna + k_opt / (np.log(t + c_offset)**2)
        # 💣 致命剥除：去掉 + 0.05 * q**4，暴露纯三次势阱悬崖！
        V_t = -q + q**2 + (a_t / 3.0) * q**3 
        U_t = U_kin * np.exp(-1j * V_t / hbar)
        U_tot = U_t @ U_tot 
        
    # 💣 测量基态同样剥除
    V_base = -q + q**2 + (a_end / 3.0) * q**3 
    H_base = F_inv @ np.diag(T_kin) @ F + np.diag(V_base)
    
    try:
        evals, evecs = np.linalg.eig(U_tot)
    except Exception:
        return 1e6
        
    phases = np.angle(evals)
    expected_energies = np.real(np.sum(np.conj(evecs) * (H_base @ evecs), axis=0))
    m = np.round((expected_energies * T_steps + hbar * phases) / (2 * np.pi * hbar))
    E_quantum = (-hbar * phases + 2 * np.pi * hbar * m) / T_steps
    E_quantum = np.sort(E_quantum)
    E_quantum = E_quantum - E_quantum[0]

    if len(E_quantum) < 101:
        return 1e6 + (101 - len(E_quantum)) * 1000 

    scale_k = TRUE_ZEROS_100[0] / E_quantum[1] if E_quantum[1] != 0 else 1.0
    E_pred = E_quantum[1:101] * scale_k
    mse = np.mean((E_pred - TRUE_ZEROS_100)**2)
    
    if np.isnan(mse) or mse > 1e6:
        return 1e6
    return mse

def callback_fn(xk, convergence):
    global generation, global_start_time, best_mse_global
    generation += 1
    current_mse = holy_grail_objective_3d_nocutoff(xk)
    if current_mse < best_mse_global:
        best_mse_global = current_mse
    elapsed = time.time() - global_start_time
    print(f" [💥 2号机 3D 自由裸奔测试: 第 {generation} 代] 耗时: {elapsed/60:.2f}分钟 | 收敛度: {convergence*100:.2f}%")
    print(f"    👉 当代挣扎参数: hbar={xk[0]:.6f}, 起点={xk[1]:.6f}, 终点={xk[2]:.6f} | 绝望 MSE = {current_mse:.4f}\n")

def run_epyc_quantum_shift_nocutoff():
    global global_start_time
    max_cores = multiprocessing.cpu_count()
    WORKERS = max(1, max_cores - 2)
    
    print("====================================================================")
    print(f" 🚀 AMD EPYC [2号母舰消融测试] 点火！物理算力核心: {max_cores} ")
    print("    [实验代号]: 无保护的自由漂移灾难 (Free Drift Catastrophe) ")
    print("    [引擎模式]: 移除 0.05*q**4，允许 3D 空间自由漂移寻找救命稻草！ ")
    print("====================================================================\n")
    
    bounds = [
        (0.05, 0.25),   
        (1.05, 1.80),   
        (0.90, 1.10)     
    ]
    
    global_start_time = time.time()
    
    result_de = differential_evolution(
        holy_grail_objective_3d_nocutoff, bounds, strategy='best1bin', 
        maxiter=10000, popsize=60, tol=1e-13, mutation=(0.5, 1.5), recombination=0.75, 
        workers=-1, updating='deferred', callback=callback_fn, polish=True, disp=False
    )
    
    print("\n" + "💣 2号机 崩溃验证结束 " + "="*50)
    print(f"【3D 自由空间 (无 phi^4 保护) 挣扎极限】:")
    print(f"[*] 畸变大自然平滑尺度 (hbar)        = {repr(result_de.x[0])}")
    print(f"[*] 畸变大爆炸退火起点 (a_start)     = {repr(result_de.x[1])}")
    print(f"[*] 畸变重正化偏移拓扑终点 (a_end)   = {repr(result_de.x[2])}")
    print(f"[*] 失去保护后的发散 MSE             = {repr(result_de.fun)}")
    print("="*64 + "\n")

if __name__ == '__main__':
    run_epyc_quantum_shift_nocutoff()
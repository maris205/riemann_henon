import os
# ==============================================================================
# 【极其致命的超算保护锁】：必须在 import numpy 前锁死底层线程！
# 保证 64 个核心完美运行独立宇宙，防止矩阵多线程引发系统级死锁海啸！
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import time
import multiprocessing
from scipy.optimize import differential_evolution

# 全局恒定真值基准 (常驻内存，防止跨进程拷贝损耗)
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

# ==============================================================================
# 3D 自由空间目标函数：hbar, a_start, a_end 全部由算力引擎掌控
# ==============================================================================
def holy_grail_objective_3d(x):
    hbar, a_start, a_end = x
    
    # 【热力学时间之箭屏障】：大爆炸起点必须高于冷却终点！
    # 并且两者至少保持 0.05 的对数冷却落差，彻底杜绝无物理意义的盲目游走。
    if hbar <= 0.02 or hbar >= 0.25 or a_end < 0.95 or a_end > 1.10:
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
    
    # 动能矩阵外提，极速预计算
    U_kin = F_inv @ np.diag(np.exp(-1j * T_kin / hbar)) @ F
    
    t_start = 1.0 / (np.log(1 + c_offset)**2)
    t_end   = 1.0 / (np.log(T_steps + c_offset)**2)
    k_opt = (a_start - a_end) / (t_start - t_end)
    a_dyna = a_end - k_opt * t_end  
    
    U_tot = np.eye(N, dtype=np.complex128)
    
    for t in range(1, T_steps + 1):
        a_t = a_dyna + k_opt / (np.log(t + c_offset)**2)
        # 带有 phi^4 保护的安全物理势阱
        V_t = -q + q**2 + (a_t / 3.0) * q**3 + 0.05 * q**4
        # O(N^2) 极速广播乘法
        U_t = U_kin * np.exp(-1j * V_t / hbar)
        U_tot = U_t @ U_tot 
        
    # 以自由寻优得到的动态 a_end 作为基准测量态！
    V_base = -q + q**2 + (a_end / 3.0) * q**3 + 0.05 * q**4
    H_base = F_inv @ np.diag(T_kin) @ F + np.diag(V_base)
    
    try:
        evals, evecs = np.linalg.eig(U_tot)
    except Exception:
        return 1e6
        
    phases = np.angle(evals)
    
    # 物理期望能量与半经典解包
    expected_energies = np.real(np.sum(np.conj(evecs) * (H_base @ evecs), axis=0))
    m = np.round((expected_energies * T_steps + hbar * phases) / (2 * np.pi * hbar))
    E_quantum = (-hbar * phases + 2 * np.pi * hbar * m) / T_steps
    E_quantum = np.sort(E_quantum)
    E_quantum = E_quantum - E_quantum[0]

    if len(E_quantum) < 101:
        return 1e6 + (101 - len(E_quantum)) * 1000 

    # 原生硬刚：首点单点线性对齐
    scale_k = TRUE_ZEROS_100[0] / E_quantum[1] if E_quantum[1] != 0 else 1.0
    E_pred = E_quantum[1:101] * scale_k
    
    mse = np.mean((E_pred - TRUE_ZEROS_100)**2)
    
    if np.isnan(mse) or mse > 1e6:
        return 1e6
        
    return mse

def callback_fn(xk, convergence):
    global generation, global_start_time, best_mse_global
    generation += 1
    # 动态求值以获取真实当代极小 MSE
    current_mse = holy_grail_objective_3d(xk)
    if current_mse < best_mse_global:
        best_mse_global = current_mse
        
    elapsed = time.time() - global_start_time
    print(f" [🌌 自由拓扑多重宇宙: 第 {generation} 代] 耗时: {elapsed/60:.2f}分钟 | 种群收敛度: {convergence*100:.2f}%")
    print(f"    👉 当代霸主参数: hbar={xk[0]:.6f}, 起点={xk[1]:.6f}, 终点={xk[2]:.6f} | 极限 MSE = {current_mse:.4f}\n")

def run_epyc_quantum_shift_search():
    global global_start_time
    max_cores = multiprocessing.cpu_count()
    WORKERS = max(1, max_cores - 2)
    
    print("====================================================================")
    print(f" 🚀 AMD EPYC [2号母舰] 点火！物理算力核心: {max_cores} ")
    print("    [实验代号]: 宇宙自由意志 (Free Will Quantum Shift) ")
    print("    [引擎模式]: 3D 连续空间全自由差分进化 (解锁 a_end) ")
    print("    [科学目标]: 寻找由普朗克常数引发的经典 KAM 边界微扰偏移！ ")
    print("====================================================================\n")
    
    # 3D 边界设定：
    # hbar: 覆盖更宽泛的共振区
    # a_start: 大爆炸起点
    # a_end: 在经典边界 1.02 附近上下浮动，允许系统发生自发量子漂移！
    # 【二号机：彻夜焚烧 3D 自由版参数】
    bounds = [
        (0.05, 0.25),    # hbar: 大尺度网格与极细网格并存
        (1.05, 1.80),    # a_start: 从微温到大爆炸极高热态
        (0.90, 1.10)     # a_end: 彻底放开 1.02，允许发生剧烈的量子相变偏移！
    ]
    print(f"[*] 【3D 进化启动】将使用 workers=-1 并行塞满所有核心！")
    print(f"[*] 请放任其燃烧！长达几个小时的极客算力远征开始了...\n")
    
    global_start_time = time.time()
    
    # 启动 3D 差分进化 (Differential Evolution)
    # 3D空间下，popsize=20 意味着每代 20 * 3(变量) = 60 个种群个体
    # 这个规模极其完美地填满了你的 64 个核心，保证 CPU 没有任何空闲！
    result_de = differential_evolution(
        holy_grail_objective_3d, 
        bounds, 
        strategy='best1bin', 
        maxiter=10000,         # 万代探索
        popsize=60,            # 3D空间，每代 60*3=180 个平行宇宙
        tol=1e-13,             
        mutation=(0.5, 1.5), 
        recombination=0.75, 
        workers=-1,            
        updating='deferred',   
        callback=callback_fn,
        polish=True,           
        disp=False
    )
    
    total_time = time.time() - global_start_time
    
    print("\n" + "🏆 量子偏移圣杯已现 " + "="*50)
    print(f"总耗时: {total_time/60:.2f} 分钟 (共历经 {result_de.nit} 代繁衍进化)")
    print(f"抛光与收敛状态: {result_de.message}\n")
    
    # 获取最高精度表示
    print("【2号机：3D 自由空间黄金参数 (16-bit Float Precision)】:")
    print(f"[*] 极限大自然平滑尺度 (hbar)        = {repr(result_de.x[0])}")
    print(f"[*] 极限大爆炸退火起点 (a_start)     = {repr(result_de.x[1])}")
    print(f"[*] 量子重正化偏移拓扑终点 (a_end)   = {repr(result_de.x[2])}")
    print(f"[*] 放开一切物理束缚后的终极 MSE   = {repr(result_de.fun)}")
    print("="*64 + "\n")
    
    best_hbar, best_astart, best_aend = result_de.x
    mse_check = holy_grail_objective_3d([best_hbar, best_astart, best_aend])
    print(f"👉 终极验证 MSE: {mse_check:.4f}")

if __name__ == '__main__':
    run_epyc_quantum_shift_search()
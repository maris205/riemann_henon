import os
# ==============================================================================
# 【极其致命的超算保护锁】：必须在 import numpy 前锁死底层线程！
# 防止 Scipy 多进程与 Numpy 多线程引发“线程海啸”，造成 64 核 EPYC 瞬间死锁！
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

# 记录全局启动时间
global_start_time = 0
generation = 0
best_mse_global = float('inf')

# ==============================================================================
# 目标函数：专为 Scipy 连续空间优化器设计 (返回单一的 float MSE)
# ==============================================================================
def holy_grail_objective(x):
    hbar, a_start = x
    
    # 防止优化算法盲目游走导致物理意义丧失
    if hbar <= 0.01 or hbar >= 0.3 or a_start <= 1.03:
        return 1e6

    # 物理保真度保持与百零点原生硬刚一致
    N = 250         
    L = 3.5         
    T_steps = 300   
    a_end = 1.02    # 【死锁经典孪生同构边界】
    c_offset = 10.0 
    
    q = np.linspace(-L, L, N, endpoint=False)
    dq = q[1] - q[0]
    dp = 2 * np.pi * hbar / (N * dq)
    p = np.fft.fftfreq(N) * N * dp
    
    T_kin = 0.5 * p**2
    F = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
    F_inv = np.fft.ifft(np.eye(N), axis=0) * np.sqrt(N)
    
    # 极速向量化预计算：将动能矩阵外提
    U_kin = F_inv @ np.diag(np.exp(-1j * T_kin / hbar)) @ F
    
    t_start = 1.0 / (np.log(1 + c_offset)**2)
    t_end   = 1.0 / (np.log(T_steps + c_offset)**2)
    k_opt = (a_start - a_end) / (t_start - t_end)
    a_dyna = a_end - k_opt * t_end  
    
    U_tot = np.eye(N, dtype=np.complex128)
    
    for t in range(1, T_steps + 1):
        a_t = a_dyna + k_opt / (np.log(t + c_offset)**2)
        V_t = -q + q**2 + (a_t / 3.0) * q**3 + 0.05 * q**4
        # 极速广播乘法：O(N^2) 碾压 O(N^3)
        U_t = U_kin * np.exp(-1j * V_t / hbar)
        U_tot = U_t @ U_tot 
        
    V_base = -q + q**2 + (a_end / 3.0) * q**3 + 0.05 * q**4
    H_base = F_inv @ np.diag(T_kin) @ F + np.diag(V_base)
    
    try:
        evals, evecs = np.linalg.eig(U_tot)
    except Exception:
        return 1e6 # 严重惩罚：矩阵奇异
        
    phases = np.angle(evals)
    
    # 全向量化提取物理期待能量
    expected_energies = np.real(np.sum(np.conj(evecs) * (H_base @ evecs), axis=0))

    m = np.round((expected_energies * T_steps + hbar * phases) / (2 * np.pi * hbar))
    E_quantum = (-hbar * phases + 2 * np.pi * hbar * m) / T_steps
    E_quantum = np.sort(E_quantum)
    E_quantum = E_quantum - E_quantum[0]

    if len(E_quantum) < 101:
        return 1e6 + (101 - len(E_quantum)) * 1000 # 给予平滑的逃逸惩罚梯度

    # 绝不妥协的首点线性对齐！
    scale_k = TRUE_ZEROS_100[0] / E_quantum[1] if E_quantum[1] != 0 else 1.0
    E_pred = E_quantum[1:101] * scale_k
    
    mse = np.mean((E_pred - TRUE_ZEROS_100)**2)
    
    if np.isnan(mse) or mse > 1e6:
        return 1e6
        
    return mse

# 用于打印遗传算法进度的回调函数
def callback_fn(xk, convergence):
    global generation, global_start_time, best_mse_global
    generation += 1
    current_mse = holy_grail_objective(xk)
    if current_mse < best_mse_global:
        best_mse_global = current_mse
        
    elapsed = time.time() - global_start_time
    print(f" [🧬 达尔文天择中: 第 {generation} 代] 耗时: {elapsed/60:.2f}分钟 | 种群收敛度: {convergence*100:.2f}%")
    print(f"    👉 当前霸主基因: hbar={xk[0]:.8f}, a_start={xk[1]:.8f} | 当代极限 MSE = {current_mse:.4f}\n")

def run_epyc_holy_grail_search():
    global global_start_time
    max_cores = multiprocessing.cpu_count()
    WORKERS = max(1, max_cores - 2) # 留2个核给系统日常调度，防止卡顿
    
    print("====================================================================")
    print(f" 🚀 AMD EPYC 巨兽点火！检测到系统并发线程数: {max_cores} ")
    print("    [引擎模式]: 连续空间全局微分进化 (Differential Evolution) ")
    print("    [末端抛光]: 触发 L-BFGS-B 单纯形法死磕 16 位极限精度 ")
    print("    [正在 16 位浮点数的汪洋大海中，搜寻大自然的终极真理常数...] ")
    print("====================================================================\n")
    
    # 缩小包围圈：我们将战区极其精准地锁定在你刚才网格扫出的极小值盆地附近！
    # 【一号机：彻夜焚烧版参数】
    bounds = [(0.05, 0.25), (1.10, 1.80)]
    
    print(f"[*] 【全局进化启动】将使用 workers=-1 并行塞满所有核心！")
    print(f"[*] 请去喝杯咖啡，每完成一代 (Generation) 都会自动播报战况...\n")
    
    global_start_time = time.time()
    
    # 启动差分进化算法 (Differential Evolution)
    # popsize=30: 每一代会产生 30 * 2(维度) = 60 个宇宙参数组合，恰好全量塞满你的 60 多个物理核心！效率最大化！
    # updating='deferred': 强制启用超算多进程池的必备设置！
    # polish=True: 进化完成后，使用微积分进行极值抛光！
    result_de = differential_evolution(
        holy_grail_objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=10000,         # 万代封印解除！跑到死为止！
        popsize=80,            # 种群密度拉满，一次迭代撒网几百个宇宙！
        tol=1e-13,             # 13 位双精度收敛死线
        mutation=(0.5, 1.5),   # 加强变异，防止早熟陷入局部陷阱
        recombination=0.7, 
        workers=-1,            # 算力全开
        updating='deferred',   
        callback=callback_fn,
        polish=True,           
        disp=False
    )
    
    total_time = time.time() - global_start_time
    
    print("\n" + "🏆 圣杯已现 " + "="*50)
    print(f"总耗时: {total_time/60:.2f} 分钟 (共历经 {result_de.nit} 代繁衍进化)")
    print(f"抛光与收敛状态: {result_de.message}\n")
    
    # 🔥 核心要求：极高精度的 repr 完整浮点数输出！拒绝任何字符串截断！
    print("【宇宙本源黄金参数 (16-bit Float Precision)】:")
    print(f"[*] 极限大自然平滑尺度 (hbar)    = {repr(result_de.x[0])}")
    print(f"[*] 极限对数大爆炸起点 (a_start) = {repr(result_de.x[1])}")
    print(f"[*] 强制 a_end=1.02 的终极原生 MSE = {repr(result_de.fun)}")
    print("="*64 + "\n")
    
    # 复验一次生成对比数据
    best_hbar, best_astart = result_de.x
    mse_check = holy_grail_objective([best_hbar, best_astart])
    
    print(f"👉 终极验证 MSE: {mse_check:.4f} (如果低于之前的 46.6，这就是真正的物理极限！)")

if __name__ == '__main__':
    run_epyc_holy_grail_search()
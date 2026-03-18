import os
# 【极其重要】：彻底切断底层库的多线程，防止 128 核发生灾难性的线程上下文切换风暴
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from numba import njit
from scipy.optimize import differential_evolution
import time
import mpmath
import multiprocessing
import gc

LOG_FILE = "macro_2d_de_N6_LR_overnight.log"

def log_msg(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ================== 1. HPC 极限优化算子 (Float32 内存压缩) ==================
@njit(fastmath=True, nogil=True)
def build_2d_transitions_anchored(eps, delta_a, steps, n_bins, limit, c_offset):
    a_c_star = 1.02  
    
    t_start = 1.0 / (np.log(1 + c_offset)**2)
    t_end   = 1.0 / (np.log(steps + c_offset)**2)
    k_opt = delta_a / (t_start - t_end)
    a_dyna = a_c_star - k_opt * t_end
    
    n_states = n_bins * n_bins
    
    # 【极致内存优化】：使用 float32，单矩阵占用约 2GB
    transitions = np.zeros((n_states, n_states), dtype=np.float32)
    V = np.zeros(n_states, dtype=np.float64)
    
    dx = (2.0 * limit) / n_bins
    inv_2eps2 = 1.0 / (2.0 * eps**2)
    radius = int(3.0 * eps / dx) + 1
    
    center_idx = int(limit / dx)
    start_state = center_idx * n_bins + center_idx
    V[start_state] = 1.0
    
    for n in range(1, steps + 1):
        a_n = a_dyna + k_opt / (np.log(n + c_offset)**2.0)
        V_next = np.zeros(n_states, dtype=np.float64)
        
        for state in range(n_states):
            if V[state] < 1e-12: continue
            
            i_x = state // n_bins
            i_y = state % n_bins
            x_curr = -limit + dx*0.5 + i_x * dx
            x_prev = -limit + dx*0.5 + i_y * dx
            
            x_next = 1.0 - a_n * x_curr**2 - x_prev
            y_next = x_curr
            
            if abs(x_next) > limit or abs(y_next) > limit:
                V_next[start_state] += V[state]
                transitions[state, start_state] += V[state]
                continue
                
            j_x_center = int((x_next + limit) / dx)
            j_y_center = int((y_next + limit) / dx)
            
            jx_start = max(0, j_x_center - radius)
            jx_end   = min(n_bins - 1, j_x_center + radius)
            jy_start = max(0, j_y_center - radius)
            jy_end   = min(n_bins - 1, j_y_center + radius)
            
            w_sum = 0.0
            for jx in range(jx_start, jx_end + 1):
                cx_val = -limit + dx*0.5 + jx * dx
                wx = np.exp(-(cx_val - x_next)**2 * inv_2eps2)
                for jy in range(jy_start, jy_end + 1):
                    cy_val = -limit + dx*0.5 + jy * dx
                    wy = np.exp(-(cy_val - y_next)**2 * inv_2eps2)
                    w_sum += wx * wy
            
            if w_sum > 1e-18:
                inv_sum = 1.0 / w_sum
                for jx in range(jx_start, jx_end + 1):
                    cx_val = -limit + dx*0.5 + jx * dx
                    wx = np.exp(-(cx_val - x_next)**2 * inv_2eps2)
                    for jy in range(jy_start, jy_end + 1):
                        cy_val = -limit + dx*0.5 + jy * dx
                        wy = np.exp(-(cy_val - y_next)**2 * inv_2eps2)
                        
                        prob = wx * wy * inv_sum
                        flow = V[state] * prob
                        
                        target_state = jx * n_bins + jy
                        V_next[target_state] += flow
                        transitions[state, target_state] += flow
            else:
                jxc = min(max(0, j_x_center), n_bins-1)
                jyc = min(max(0, j_y_center), n_bins-1)
                target_state = jxc * n_bins + jyc
                V_next[target_state] += V[state]
                transitions[state, target_state] += V[state]
                
        V = V_next
    return transitions

# ================== 2. 升维稀疏相位提取器 ==================
def extract_phases_2d(trans_matrix):
    P_sparse = sp.csr_matrix(trans_matrix, dtype=np.float64)
    
    sums = np.array(P_sparse.sum(axis=1)).flatten()
    sums[sums == 0] = 1.0
    P_sparse.data /= sums[P_sparse.indices]
    
    # 【终极防翻车配置】：k=250 保证正能量态数量，which='LR' 死死锁住低频物理锚点！
    vals, _ = eigs(P_sparse, k=250, which='LR', tol=1e-5)
    
    pos_vals = vals[vals.imag > 1e-4]
    phases = np.unwrap(np.sort(np.angle(pos_vals)))
    return phases

# ================== 3. DE 进化核 ==================
def objective_2d(params, target_zeros, steps, n_bins, limit, offset):
    eps, delta_a = params[0], params[1]
    t0 = time.time()
    
    try:
        trans = build_2d_transitions_anchored(eps, delta_a, steps, n_bins, limit, offset)
        sys_phases = extract_phases_2d(trans)
        
        # 救命操作：粉碎 2GB 稠密矩阵，释放内存
        del trans
        gc.collect()
        
        N_actual = len(sys_phases)
        # 兜底：如果砍掉共轭态后连 6 个都不够，直接枪毙
        if N_actual < 6: 
            return 1e6 + (10 - N_actual) * 1e4
            
        # 锚点定标（因为有了 which='LR'，这里的 sys_phases[0] 绝对纯正！）
        scale = target_zeros[0] / sys_phases[0]
        
        # 【回归物理极限】：只提取前 6 阶进行 MAE 计算
        predicted = sys_phases[:6] * scale
        mae = np.mean(np.abs(predicted - target_zeros[:6]))
        
        log_msg(f"[Worker] eps={eps:.6f} | delta_a={delta_a:.6f} | MAE(6)={mae:.5f} | N={N_actual} | 耗时:{time.time()-t0:.1f}s")
        return mae
        
    except Exception as e:
        log_msg(f"[Worker Error] eps={eps:.6f} | 报错: {str(e)}")
        gc.collect() 
        return 1e8

# ================== 4. 怪兽级发射阵列 ==================
if __name__ == '__main__':
    # Linux HPC 环境下强制隔离进程内存
    multiprocessing.set_start_method('forkserver', force=True)
    
    mpmath.mp.dps = 25
    # 生成前 10 阶零点（用前 6 阶测算足矣，留几阶做备用观察）
    zeros = np.array([float(mpmath.zetazero(i).imag) for i in range(1, 11)])
    
    log_msg("🚀🚀 2D 终极保守宇宙 (前 6 阶深挖·通宵挂机版) 启动 🚀🚀")
    
    MAX_WORKERS = 50
    log_msg(f"硬件自检：并发 {MAX_WORKERS} 核火力全开，内存屏障已激活。")
    
    # 【收网战术】：极其克制地缩小包围圈，直击火山口！
    bounds = [
        (0.042, 0.050),   # eps：在普朗克常数 0.044~0.048 附近深挖
        (0.001, 0.020)    # delta_a：防止陷入过深的局部冰窟窿
    ]
    
    res = differential_evolution(
        func=objective_2d, 
        bounds=bounds,
        args=(zeros, 50000, 150, 3.0, 10.0), # 150 网格，5万步淬火
        strategy='best1bin', 
        maxiter=15,        
        popsize=50,        
        workers=MAX_WORKERS, 
        polish=False,      
        disp=True
    )
    
    log_msg("\n========================================================")
    log_msg(f"🏆 黎曼 2D 马赛克宇宙·最终裁决 🏆")
    log_msg(f"最优离散尺度 (BEST_EPS) : {res.x[0]!r}")
    log_msg(f"最优冷却深度 (DELTA_A)  : {res.x[1]!r}")
    log_msg(f"极限最小 MAE (N=1~6)    : {res.fun!r}")
    log_msg("========================================================\n")
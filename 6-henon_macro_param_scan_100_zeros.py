import os
# 【极其重要】：彻底切断底层库的多线程，防止 HPC 发生灾难性的上下文切换风暴
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
import warnings

warnings.filterwarnings('ignore')

LOG_FILE = "macro_2d_de_N100_Grid200_overnight.log"

def log_msg(message):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# ================== 1. HPC 极限优化算子 (Float32 6.4GB 巨型内存版) ==================
@njit(fastmath=True, nogil=True)
def build_2d_transitions_anchored(eps, delta_a, steps, n_bins, limit, c_offset):
    a_c_star = 1.02  
    
    t_start = 1.0 / (np.log(1 + c_offset)**2)
    t_end   = 1.0 / (np.log(steps + c_offset)**2)
    k_opt = delta_a / (t_start - t_end)
    a_dyna = a_c_star - k_opt * t_end
    
    n_states = n_bins * n_bins
    
    # 【核武级内存分配】：200x200网格下，单矩阵严格且刚性占用 6.4 GB 物理内存！
    transitions = np.zeros((n_states, n_states), dtype=np.float32)
    V = np.zeros(n_states, dtype=np.float64)
    
    dx = (2.0 * limit) / n_bins
    inv_2eps2 = 1.0 / (2.0 * eps**2)
    # 物理扩散半径自动适配更细的网格
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

# ================== 2. 神级内存拯救术 (Numba CSR 零拷贝拦截) ==================
@njit(fastmath=True, nogil=True)
def extract_csr_components(trans):
    """
    直接在 Numba 层生成 CSR 稀疏格式！彻底干掉 Scipy 转换时的内存翻倍！
    """
    n = trans.shape[0]
    nnz = 0
    # 第一遍扫描：统计非零元素个数
    for i in range(n):
        for j in range(n):
            if trans[i, j] > 1e-12:
                nnz += 1
                
    data = np.zeros(nnz, dtype=np.float64)
    indices = np.zeros(nnz, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    
    idx = 0
    # 第二遍扫描：填充 CSR 数组
    for i in range(n):
        for j in range(n):
            val = trans[i, j]
            if val > 1e-12:
                data[idx] = val
                indices[idx] = j
                idx += 1
        indptr[i+1] = idx
        
    return data, indices, indptr

# ================== 3. DE 进化核 (百阶大考版) ==================
def objective_2d(params, target_zeros, steps, n_bins, limit, offset):
    eps, delta_a = params[0], params[1]
    t0 = time.time()
    
    try:
        # 1. 构建 6.4GB 巨型稠密矩阵
        trans = build_2d_transitions_anchored(eps, delta_a, steps, n_bins, limit, offset)
        n_states = trans.shape[0]
        
        # 2. 调用 Numba 极速提取 CSR 三元组 (仅需几十 MB 内存)
        data, indices, indptr = extract_csr_components(trans)
        
        # 3. 【救命级操作】：极速粉碎 6.4GB 巨型稠密矩阵！防 OOM 核心！
        del trans
        gc.collect()
        
        # 4. 在极低内存下构建 Scipy 稀疏矩阵
        P_sparse = sp.csr_matrix((data, indices, indptr), shape=(n_states, n_states))
        del data, indices, indptr
        gc.collect()
        
        # 概率流归一化
        sums = np.array(P_sparse.sum(axis=1)).flatten()
        sums[sums == 0] = 1.0
        P_sparse.data /= sums[P_sparse.indices]
        
        # 注入固定的初始向量，防止优化器被 ARPACK 的随机起点带偏！
        np.random.seed(42)
        v0_fixed = np.random.rand(n_states)

        # 【算力狂飙】：网格达 200，态密度极高！k 必须提升至 400！
        vals, _ = eigs(P_sparse, k=400, which='LR', tol=1e-5, v0=v0_fixed)
        
        pos_vals = vals[vals.imag > 1e-4]
        sys_phases = np.unwrap(np.sort(np.angle(pos_vals)))
        
        N_actual = len(sys_phases)
        
        # 【严苛斩杀线】：如果提取出的态不到 100 个，直接按短缺数量给予极其暴力的惩罚！
        if N_actual < 100: 
            penalty = 1e6 + (100 - N_actual) * 1e4
            log_msg(f"[Worker Penalty] eps={eps:.6f} | delta_a={delta_a:.6f} | 物理容量不足: {N_actual} 阶 (耗时:{time.time()-t0:.1f}s)")
            return penalty
            
        # 绝对单点锚定法则
        scale = target_zeros[0] / sys_phases[0]
        
        # 【终极大考】：直接提取前 100 阶进行全局 MAE 计算！让算法感受韦尔律的压迫感！
        predicted = sys_phases[:100] * scale
        mae_100 = np.mean(np.abs(predicted - target_zeros[:100]))

        log_msg(f"[Worker SUCCESS] eps={eps} | delta_a={delta_a} | MAE(100)={mae_100} | 容量={N_actual}阶 | 耗时:{time.time()-t0:.2f}s")
        return mae_100
        
    except Exception as e:
        log_msg(f"[Worker Error] eps={eps:.6f} | 报错: {str(e)}")
        gc.collect() 
        return 1e8

# ================== 4. EPYC 480GB 满血防爆调度阵列 ==================
if __name__ == '__main__':
    # 强制隔离进程内存，防止 Linux 内核发生 COW 导致内存翻倍死机
    multiprocessing.set_start_method('forkserver', force=True)
    
    mpmath.mp.dps = 25
    zeros_100 = np.array([float(mpmath.zetazero(i).imag) for i in range(1, 101)])
    
    log_msg("🚀🚀 2D 终极保守宇宙 (Grid=200 超高清抗锯齿·EPYC 防爆版) 启动 🚀🚀")
    
    # 【生死内存墙控制】：48 并发 * 6.5GB ≈ 312GB。完美卡在 480GB 绝对安全区！绝不宕机！
    MAX_WORKERS = 48  
    log_msg(f"硬件自检：AMD EPYC 9754 并发严密封锁在 {MAX_WORKERS} 核，Numba 零拷贝结界已开启！")
    
    bounds = [
        (0.042, 0.052),   # eps：大自然分辨极限
        (0.001, 0.020)    # delta_a：绝热微扰深度
    ]
    
    # 完美心跳映射：popsize=24 -> 每代繁衍 48 个个体 -> 刚好填满 48 个并发内核！不留一毫秒空闲！
    res = differential_evolution(
        func=objective_2d, 
        bounds=bounds,
        args=(zeros_100, 50000, 200, 3.0, 10.0), # 🚀 网格史诗级突破至 200！
        strategy='best1bin', 
        maxiter=30,          # 既然算力到位了，进化深度放开到 30 代！
        popsize=24,          # 【核心配比】：完美压榨 48 核
        updating='deferred', # 【极其关键】：世代必须同步更新，杜绝抢锁内耗
        workers=MAX_WORKERS, 
        polish=False,      
        disp=True
    )
    
    log_msg("\n========================================================")
    log_msg(f"🏆 黎曼 2D 马赛克宇宙·百阶大一统 (Grid=200 高保真版) 🏆")
    log_msg(f"最优离散尺度 (BEST_EPS) : {res.x[0]!r}")
    log_msg(f"最优冷却深度 (DELTA_A)  : {res.x[1]!r}")
    log_msg(f"极限最小 MAE (N=1~100)  : {res.fun!r}")
    log_msg("========================================================\n")
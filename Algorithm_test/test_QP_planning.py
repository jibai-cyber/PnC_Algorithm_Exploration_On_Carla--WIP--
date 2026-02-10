import numpy as np
import scipy.sparse as sp
import osqp
import matplotlib.pyplot as plt

def build_second_diff_matrix(n):
    """构建二阶差分矩阵（针对交错排列的变量：[x1, y1, x2, y2, ...]）"""
    total_rows = 2 * (n-2)  # x和y各(n-2)行
    total_cols = 2 * n      # 每个点有x,y两个变量
    
    rows = []
    cols = []
    data = []
    
    # x坐标的二阶差分（在偶数行）
    for i in range(n-2):
        row_idx = 2 * i
        # 系数：x_i - 2x_{i+1} + x_{i+2}
        rows.append(row_idx)
        cols.append(2 * i)          # x_i (列索引: 2*i)
        data.append(1)
        
        rows.append(row_idx)
        cols.append(2 * (i+1))      # x_{i+1} (列索引: 2*(i+1))
        data.append(-2)
        
        rows.append(row_idx)
        cols.append(2 * (i+2))      # x_{i+2} (列索引: 2*(i+2))
        data.append(1)
    
    # y坐标的二阶差分（在奇数行）
    for i in range(n-2):
        row_idx = 2 * i + 1
        # 系数：y_i - 2y_{i+1} + y_{i+2}
        rows.append(row_idx)
        cols.append(2 * i + 1)      # y_i (列索引: 2*i+1)
        data.append(1)
        
        rows.append(row_idx)
        cols.append(2 * (i+1) + 1)  # y_{i+1} (列索引: 2*(i+1)+1)
        data.append(-2)
        
        rows.append(row_idx)
        cols.append(2 * (i+2) + 1)  # y_{i+2} (列索引: 2*(i+2)+1)
        data.append(1)
    
    D2 = sp.csc_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    return D2

def build_first_diff_matrix(n):
    """构建一阶差分矩阵（针对交错排列的变量）"""
    total_rows = 2 * (n-1)
    total_cols = 2 * n
    
    rows = []
    cols = []
    data = []
    
    # x坐标的一阶差分（在偶数行）
    for i in range(n-1):
        row_idx = 2 * i
        # 系数：x_{i+1} - x_i
        rows.append(row_idx)
        cols.append(2 * i)          # x_i
        data.append(-1)
        
        rows.append(row_idx)
        cols.append(2 * (i+1))      # x_{i+1}
        data.append(1)
    
    # y坐标的一阶差分（在奇数行）
    for i in range(n-1):
        row_idx = 2 * i + 1
        # 系数：y_{i+1} - y_i
        rows.append(row_idx)
        cols.append(2 * i + 1)      # y_i
        data.append(-1)
        
        rows.append(row_idx)
        cols.append(2 * (i+1) + 1)  # y_{i+1}
        data.append(1)
    
    D1 = sp.csc_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    return D1

def build_qp_matrices(original_points, w1=1.0, w2=0.1, w3=0.01):
    """
    构建QP问题的P和q矩阵
    
    代价函数：
    J = w1 * Σ(x_i - x_ir)^2 + (y_i - y_ir)^2        # 相似代价
        + w2 * Σ(x_i - 2x_{i+1} + x_{i+2})^2         # 平滑代价（二阶差分）
        + w3 * Σ(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2 # 累减代价（一阶差分）
    
    参数：
    original_points: (n, 2) 原始路径点
    w1, w2, w3: 权重系数
    
    返回：
    P: 二次项矩阵 (2n×2n)
    q: 一次项向量 (2n×1)
    """
    n = len(original_points)
    total_vars = 2 * n
    
    # 1. 相似代价的贡献
    P_similar = 2 * w1 * sp.eye(total_vars)
    q = -2 * w1 * original_points.flatten()
    
    # 2. 平滑代价的贡献（二阶差分）
    D2 = build_second_diff_matrix(n)
    P_smooth = 2 * w2 * (D2.T @ D2)
    
    # 3. 累减代价的贡献（一阶差分）
    D1 = build_first_diff_matrix(n)
    P_reduction = 2 * w3 * (D1.T @ D1)
    
    # 4. 合并所有项
    P = P_similar + P_smooth + P_reduction
    
    return P, q

def add_path_constraints(original_points, fix_start_end=True, deviation_limit=0.5):
    """
    参数：
    original_points: (n, 2) 原始路径点
    fix_start_end: 是否固定起点和终点
    deviation_limit: 最大允许偏差
    
    返回：
    A: 约束矩阵
    l: 约束下界
    u: 约束上界
    """
    n = len(original_points)
    total_vars = 2 * n
    
    rows = []
    cols = []
    data = []
    l_list = []
    u_list = []
    constraint_idx = 0
    
    # 1. 固定起点和终点（等式约束）
    if fix_start_end:
        # 起点 x1 = x1_original (等式约束：l=u)
        rows.append(constraint_idx); cols.append(0); data.append(1)
        l_list.append(original_points[0, 0])
        u_list.append(original_points[0, 0])
        constraint_idx += 1
        
        # 起点 y1 = y1_original
        rows.append(constraint_idx); cols.append(1); data.append(1)
        l_list.append(original_points[0, 1])
        u_list.append(original_points[0, 1])
        constraint_idx += 1
        
        # 终点 xn = xn_original
        rows.append(constraint_idx); cols.append(2*(n-1)); data.append(1)
        l_list.append(original_points[-1, 0])
        u_list.append(original_points[-1, 0])
        constraint_idx += 1
        
        # 终点 yn = yn_original
        rows.append(constraint_idx); cols.append(2*(n-1)+1); data.append(1)
        l_list.append(original_points[-1, 1])
        u_list.append(original_points[-1, 1])
        constraint_idx += 1
    
    # 2. 边界约束：x_original - d ≤ x ≤ x_original + d
    if deviation_limit > 0:
        for i in range(n):
            # x_i 的双界约束
            rows.append(constraint_idx)
            cols.append(2*i)
            data.append(1)
            l_list.append(original_points[i, 0] - deviation_limit)
            u_list.append(original_points[i, 0] + deviation_limit)
            constraint_idx += 1
            
            # y_i 的双界约束
            rows.append(constraint_idx)
            cols.append(2*i + 1)
            data.append(1)
            l_list.append(original_points[i, 1] - deviation_limit)
            u_list.append(original_points[i, 1] + deviation_limit)
            constraint_idx += 1
    
    # 3. 构建约束矩阵
    m = constraint_idx
    A = sp.csc_matrix((data, (rows, cols)), shape=(m, total_vars))
    l = np.array(l_list)
    u = np.array(u_list)
    
    return A, l, u

def smooth_path(original_points, w1=1.0, w2=0.5, w3=0.1, 
                fix_start_end=True, deviation_limit=0.5, verbose=False):
    """
    使用QP平滑路径
    
    参数：
    original_points: (n, 2) 原始路径点
    w1: 相似权重（控制与原始路径的接近程度）
    w2: 平滑权重（控制曲率平滑度）
    w3: 累减权重（控制点间距离均匀度）
    fix_start_end: 是否固定起点和终点
    deviation_limit: 最大允许偏差
    verbose: 是否显示求解信息
    
    返回：
    smoothed_points: (n, 2) 平滑后的路径点
    result: QP求解结果
    """
    n = len(original_points)
    
    # 1. 构建QP矩阵
    print(f"构建QP矩阵... (n={n})")
    P, q = build_qp_matrices(original_points, w1, w2, w3)
    
    # 2. 添加约束
    print("添加路径约束...")
    A, l, u = add_path_constraints(original_points, fix_start_end, deviation_limit)
    
    # 3. 求解QP问题
    print("求解QP问题...")
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=verbose, eps_abs=1e-6, eps_rel=1e-6)
    result = prob.solve()
    
    # 4. 检查求解状态
    if result.info.status_val != 1:  # 1表示solved
        print(f"警告: 求解未完全成功，状态: {result.info.status}")
        if result.x is None:
            print("返回原始路径")
            return original_points, result
    
    # 5. 提取结果
    smoothed_points = result.x.reshape(-1, 2)
    
    print(f"求解完成！目标值: {result.info.obj_val:.4f}")
    print(f"迭代次数: {result.info.iter}, 求解时间: {result.info.run_time:.4f}秒")
    
    return smoothed_points, result

def compute_path_metrics(points):
    """计算路径指标"""
    if len(points) < 3:
        return {}
    
    # 计算曲率
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    
    # 一阶差分（梯度）
    ds = np.sqrt(dx**2 + dy**2)
    
    # 二阶差分
    if len(points) >= 3:
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        
        # 曲率公式: κ = |dx*ddy - ddy*ddx| / (dx² + dy²)^(3/2)
        numerator = np.abs(dx[1:] * ddy - dy[1:] * ddx)
        denominator = (dx[1:]**2 + dy[1:]**2)**1.5
        
        # 避免除零
        valid_idx = denominator > 1e-10
        curvature = np.zeros(len(numerator))
        curvature[valid_idx] = numerator[valid_idx] / denominator[valid_idx]
        
        curvature_metrics = {
            '平均曲率': np.mean(curvature),
            '曲率标准差': np.std(curvature),
            '最大曲率': np.max(curvature),
        }
    else:
        curvature_metrics = {}
    
    # 路径长度
    total_length = np.sum(ds)
    
    # 点间距离变化
    distance_var = np.std(ds) if len(ds) > 1 else 0
    
    metrics = {
        '总长度': total_length,
        '点间距离标准差': distance_var,
        '平均点间距': np.mean(ds),
    }
    metrics.update(curvature_metrics)
    
    return metrics

def visualize_results(original_points, smoothed_points, title="路径平滑结果"):
    """可视化原始路径和平滑路径"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 路径对比图
    ax1 = axes[0]
    ax1.plot(original_points[:, 0], original_points[:, 1], 'ro-', 
             label='原始路径', linewidth=2, markersize=8, alpha=0.7)
    ax1.plot(smoothed_points[:, 0], smoothed_points[:, 1], 'bs-', 
             label='平滑路径', linewidth=2, markersize=8, alpha=0.7)
    
    # 标记起点和终点
    ax1.scatter(original_points[0, 0], original_points[0, 1], 
                c='green', s=200, marker='*', label='起点', zorder=5)
    ax1.scatter(original_points[-1, 0], original_points[-1, 1], 
                c='red', s=200, marker='*', label='终点', zorder=5)
    
    # 添加偏差范围示意
    for i in range(len(original_points)):
        circle = plt.Circle(original_points[i], 0.5, color='gray', 
                           alpha=0.2, fill=True)
        ax1.add_patch(circle)
    
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 曲率对比图
    ax2 = axes[1]
    
    # 计算曲率（简化版）
    if len(original_points) >= 3:
        def simple_curvature(points):
            curvatures = []
            for i in range(1, len(points)-1):
                # 使用三点计算近似曲率
                p0 = points[i-1]
                p1 = points[i]
                p2 = points[i+1]
                
                # 向量
                v1 = p1 - p0
                v2 = p2 - p1
                
                # 角度变化
                angle1 = np.arctan2(v1[1], v1[0])
                angle2 = np.arctan2(v2[1], v2[0])
                angle_diff = np.abs(angle2 - angle1)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                
                curvatures.append(angle_diff)
            return np.array(curvatures)
        
        orig_curvature = simple_curvature(original_points)
        smooth_curvature = simple_curvature(smoothed_points)
        
        indices = np.arange(1, len(original_points)-1)
        ax2.plot(indices, orig_curvature, 'ro-', label='原始路径曲率', alpha=0.7)
        ax2.plot(indices, smooth_curvature, 'bs-', label='平滑路径曲率', alpha=0.7)
        ax2.set_xlabel('路径点索引')
        ax2.set_ylabel('曲率（角度变化）')
        ax2.set_title('路径曲率对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '需要至少3个点计算曲率', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('曲率对比（点数不足）')
    
    plt.tight_layout()
    return fig

def main():
    """主函数：演示路径平滑"""
    np.random.seed(42)
    
    # 生成示例路径（锯齿状路径）
    n = 15  # 路径点数量
    t = np.linspace(0, 10, n)
    
    # 添加噪声的锯齿路径
    original_x = t + 0.8 * np.random.randn(n)
    original_y = 0.5 * t + 0.6 * np.random.randn(n)
    
    # 添加一个明显的拐弯
    original_x[5:8] += 2.0
    original_y[8:11] += 1.5
    
    original_points = np.column_stack([original_x, original_y])
    
    print("="*60)
    print("路径平滑演示")
    print(f"路径点数: {n}")
    print("="*60)
    
    # 计算原始路径指标
    orig_metrics = compute_path_metrics(original_points)
    print("\n原始路径指标:")
    for key, value in orig_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 平滑路径
    print("\n" + "="*60)
    print("开始路径平滑...")
    
    # 设置权重参数
    w1 = 1.0   # 相似权重：控制与原始路径的接近程度
    w2 = 0.5   # 平滑权重：控制曲率平滑度
    w3 = 0.1   # 累减权重：控制点间距离均匀度
    
    print(f"权重参数: w1={w1}, w2={w2}, w3={w3}")
    
    smoothed_points, result = smooth_path(
        original_points, 
        w1=w1, w2=w2, w3=w3,
        fix_start_end=True,
        deviation_limit=0.8,
        verbose=False
    )
    
    # 计算平滑后路径指标
    print("\n" + "="*60)
    print("平滑路径指标:")
    smooth_metrics = compute_path_metrics(smoothed_points)
    for key, value in smooth_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 计算改进百分比
    print("\n改进百分比:")
    for key in ['平均曲率', '曲率标准差', '最大曲率', '点间距离标准差']:
        if key in orig_metrics and key in smooth_metrics:
            improvement = (orig_metrics[key] - smooth_metrics[key]) / orig_metrics[key] * 100
            print(f"  {key}: {improvement:+.1f}%")
    
    # 可视化
    print("\n" + "="*60)
    print("生成可视化结果...")
    
    fig = visualize_results(original_points, smoothed_points, 
                           f"路径平滑结果 (w1={w1}, w2={w2}, w3={w3})")
    
    # 添加文本信息
    info_text = (
        f"原始路径长度: {orig_metrics.get('总长度', 0):.2f}\n"
        f"平滑路径长度: {smooth_metrics.get('总长度', 0):.2f}\n"
        f"曲率改善: {((orig_metrics.get('平均曲率', 0) - smooth_metrics.get('平均曲率', 0)) / orig_metrics.get('平均曲率', 1) * 100):+.1f}%"
    )
    
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()
    
    # 保存结果
    save_results = input("\n是否保存结果到文件？ (y/n): ")
    if save_results.lower() == 'y':
        np.save('original_path.npy', original_points)
        np.save('smoothed_path.npy', smoothed_points)
        fig.savefig('path_smoothing_result.png', dpi=300, bbox_inches='tight')
        print("结果已保存到文件！")
    
    return original_points, smoothed_points, result

if __name__ == "__main__":
    # 检查依赖
    try:
        import osqp
        import matplotlib
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装所需库: pip install osqp matplotlib")
    else:
        original_points, smoothed_points, result = main()
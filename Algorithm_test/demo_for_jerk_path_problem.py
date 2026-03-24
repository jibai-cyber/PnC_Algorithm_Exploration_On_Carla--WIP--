import numpy as np
import scipy.sparse as sp

class FrenetPathPlanner:
    def __init__(self, n, delta_s, 
                 vehicle_length=4.5, vehicle_width=2.0,
                 max_steering_angle=0.6,  # 最大方向盘转角(rad)
                 max_steering_rate=0.5,    # 最大转角变化率(rad/s)
                 wheelbase=2.8):           # 轴距(m)
        """
        初始化Frenet坐标系路径规划器
        
        参数：
        n: 路径点数量
        delta_s: 相邻点弧长间隔(m)
        vehicle_length: 车长(m)
        vehicle_width: 车宽(m)
        max_steering_angle: 最大方向盘转角(rad)
        max_steering_rate: 最大转角变化率(rad/s)
        wheelbase: 轴距(m)
        """
        self.n = n
        self.delta_s = delta_s
        self.total_vars = 3 * n
        
        # 车辆参数
        self.half_length = vehicle_length / 2
        self.half_width = vehicle_width / 2
        self.d1 = self.half_length  # 前半轴长
        self.d2 = self.half_length  # 后半轴长
        self.w = vehicle_width
        self.L = wheelbase
        
        # 运动学限制
        self.max_kappa = np.tan(max_steering_angle) / wheelbase
        self.max_kappa_rate = max_steering_rate / wheelbase  # 曲率变化率限制
        
        # 约束收集
        self.A_list = []
        self.l_list = []
        self.u_list = []

    def build_frenet_qp_3n(n, delta_s, 
                       w_ref=1.0, w_l=0.1, w_dl=0.1, w_ddl=0.1, w_dddl=0.01,
                       l_ref=None, 
                       l_end=0.0, l_end_dl=0.0, l_end_ddl=0.0,
                       w_end_l=1.0, w_end_dl=0.1, w_end_ddl=0.01):
        """
        构建Frenet坐标系下路径规划QP问题的P和q矩阵
        状态向量维度：3n×1 [l0, l0', l0'', l1, l1', l1'', ..., ln-1, ln-1', ln-1'']
        
        参数：
        n: 路径点数量
        delta_s: 相邻点间的弧长间隔
        w_ref: 参考线跟踪权重
        w_l: 横向位置权重
        w_dl: 横向速度权重
        w_ddl: 横向加速度权重
        w_dddl: 横向加加速度权重
        l_ref: 参考横向偏移序列 (n维向量)
        l_end: 终点横向偏移目标
        l_end_dl: 终点横向速度目标
        l_end_ddl: 终点横向加速度目标
        w_end_l: 终点位置权重
        w_end_dl: 终点速度权重
        w_end_ddl: 终点加速度权重
        
        返回：
        P: 二次项矩阵 (3n×3n)
        q: 一次项向量 (3n×1)
        """
        
        total_vars = 3 * n
        
        # 默认参考线为0（车道中心线）
        if l_ref is None:
            l_ref = np.zeros(n)
        
        # 1. 构建位置提取矩阵 E_pos (n × 3n)
        rows_pos = np.arange(n)
        cols_pos = 3 * rows_pos  # l_i 的索引是 3i
        data_pos = np.ones(n)
        E_pos = sp.csc_matrix((data_pos, (rows_pos, cols_pos)), shape=(n, total_vars))
        
        # 2. 构建速度提取矩阵 E_vel (n × 3n)
        rows_vel = np.arange(n)
        cols_vel = 3 * rows_vel + 1  # l_i' 的索引是 3i+1
        data_vel = np.ones(n)
        E_vel = sp.csc_matrix((data_vel, (rows_vel, cols_vel)), shape=(n, total_vars))
        
        # 3. 构建加速度提取矩阵 E_acc (n × 3n)
        rows_acc = np.arange(n)
        cols_acc = 3 * rows_acc + 2  # l_i'' 的索引是 3i+2
        data_acc = np.ones(n)
        E_acc = sp.csc_matrix((data_acc, (rows_acc, cols_acc)), shape=(n, total_vars))
        
        # 4. 构建加加速度矩阵 E_jerk ((n-1) × 3n)
        # l_i''' = (l_{i+1}'' - l_i'')/delta_s
        rows_jerk = []
        cols_jerk = []
        data_jerk = []
        
        for i in range(n-1):
            # -1/delta_s * l_i''
            rows_jerk.append(i)
            cols_jerk.append(3*i + 2)
            data_jerk.append(-1.0 / delta_s)
            
            # 1/delta_s * l_{i+1}''
            rows_jerk.append(i)
            cols_jerk.append(3*(i+1) + 2)
            data_jerk.append(1.0 / delta_s)
        
        E_jerk = sp.csc_matrix((data_jerk, (rows_jerk, cols_jerk)), shape=(n-1, total_vars))
        
        # 5. 构建P矩阵
        # 初始化P为零矩阵
        P = sp.lil_matrix((total_vars, total_vars))
        
        # 参考线跟踪项 + 位置项
        P += 2 * (w_ref + w_l) * (E_pos.T @ E_pos)
        
        # 速度项
        if w_dl > 0:
            P += 2 * w_dl * (E_vel.T @ E_vel)
        
        # 加速度项
        if w_ddl > 0:
            P += 2 * w_ddl * (E_acc.T @ E_acc)
        
        # 加加速度项
        if w_dddl > 0 and n > 1:
            P += 2 * w_dddl * (E_jerk.T @ E_jerk)
        
        # 终点项
        if w_end_l > 0:
            # 终点位置
            idx = 3*(n-1)
            P[idx, idx] += 2 * w_end_l
        
        if w_end_dl > 0 and n > 0:
            # 终点速度
            idx = 3*(n-1) + 1
            P[idx, idx] += 2 * w_end_dl
        
        if w_end_ddl > 0 and n > 0:
            # 终点加速度
            idx = 3*(n-1) + 2
            P[idx, idx] += 2 * w_end_ddl
        
        # 6. 构建q向量
        q = np.zeros(total_vars)
        
        # 参考线跟踪项
        q += -2 * w_ref * (E_pos.T @ l_ref).flatten()
        
        # 终点项
        if w_end_l > 0:
            q[3*(n-1)] += -2 * w_end_l * l_end
        
        if w_end_dl > 0:
            q[3*(n-1) + 1] += -2 * w_end_dl * l_end_dl
        
        if w_end_ddl > 0:
            q[3*(n-1) + 2] += -2 * w_end_ddl * l_end_ddl
        
        # 转换为CSC格式（OSQP要求）
        P = P.tocsc()
        
        return P, q
        
    def _add_constraint(self, A_row, l, u):
        """添加单行约束"""
        if isinstance(A_row, np.ndarray):
            A_row = A_row.reshape(1, -1)
        self.A_list.append(sp.csr_matrix(A_row))
        self.l_list.append(l)
        self.u_list.append(u)
    
    def _add_constraint_matrix(self, A, l, u):
        """添加多行约束矩阵"""
        self.A_list.append(sp.csr_matrix(A))
        self.l_list.extend(l)
        self.u_list.extend(u)
    
    # ========== 约束条件0：道路边界约束 ==========
    def add_boundary_constraints(self, path_boundary):
        """
        添加道路边界约束
        
        参数：
        path_boundary: (n, 2)数组，每行[l_min, l_max]
        """
        assert path_boundary.shape == (self.n, 2)
        
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        
        for i in range(self.n):
            # 提取l_i
            rows.append(i)
            cols.append(3 * i)
            data.append(1.0)
            l_list.append(path_boundary[i, 0])
            u_list.append(path_boundary[i, 1])
        
        A = sp.csc_matrix((data, (rows, cols)), shape=(self.n, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)
        print(f"添加道路边界约束: {self.n}个")
    
    # ========== 约束条件1：二阶导不等式约束 ==========
    def add_curvature_constraints(self, kappa_ref, v_current):
        """
        添加曲率限制（二阶导约束）
        
        参数：
        kappa_ref: (n,)数组，参考线曲率
        v_current: 当前车速(m/s)
        """
        assert len(kappa_ref) == self.n
        
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        
        # 考虑车速对最大曲率的影响
        v_factor = min(1.0, v_current / 10.0)  # 车速因子，低速时可转向更大角度
        kappa_max = self.max_kappa * v_factor
        
        for i in range(self.n):
            rows.append(i)
            cols.append(3 * i + 2)  # l_i''
            data.append(1.0)
            
            # l_i'' ∈ [-kappa_max - kappa_ref, kappa_max - kappa_ref]
            l_list.append(-kappa_max - kappa_ref[i])
            u_list.append(kappa_max - kappa_ref[i])
        
        A = sp.csc_matrix((data, (rows, cols)), shape=(self.n, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)
        print(f"添加曲率约束: {self.n}个")
    
    # ========== 约束条件2：三阶导不等式约束 ==========
    def add_jerk_constraints(self, v_current):
        """
        添加加加速度限制（三阶导约束）
        
        参数：
        v_current: 当前车速(m/s)
        """
        if self.n < 2:
            return
        
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        
        # 加加速度上限
        jerk_max = self.max_kappa_rate / v_current if v_current > 0.1 else self.max_kappa_rate * 10
        
        for i in range(self.n - 1):
            # -l_i''/Δs
            rows.append(i)
            cols.append(3 * i + 2)
            data.append(-1.0 / self.delta_s)
            
            # +l_{i+1}''/Δs
            rows.append(i)
            cols.append(3 * (i + 1) + 2)
            data.append(1.0 / self.delta_s)
            
            l_list.append(-jerk_max)
            u_list.append(jerk_max)
        
        A = sp.csc_matrix((data, (rows, cols)), shape=(self.n - 1, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)
        print(f"添加加加速度约束: {self.n-1}个")
    
    # ========== 约束条件3：物理连续性等式约束 ==========
    def add_continuity_constraints(self):
        """添加运动学连续性约束"""
        if self.n < 2:
            return
        
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        
        for i in range(self.n - 1):
            # === 速度连续性约束 ===
            # l_{i+1}' - l_i' - (Δs/2)*l_i'' - (Δs/2)*l_{i+1}'' = 0
            row_idx = 2 * i
            
            # -l_i'
            rows.append(row_idx)
            cols.append(3 * i + 1)
            data.append(-1.0)
            
            # -(Δs/2)*l_i''
            rows.append(row_idx)
            cols.append(3 * i + 2)
            data.append(-self.delta_s / 2)
            
            # +l_{i+1}'
            rows.append(row_idx)
            cols.append(3 * (i + 1) + 1)
            data.append(1.0)
            
            # -(Δs/2)*l_{i+1}''
            rows.append(row_idx)
            cols.append(3 * (i + 1) + 2)
            data.append(-self.delta_s / 2)
            
            l_list.append(0.0)
            u_list.append(0.0)
            
            # === 位置连续性约束 ===
            # l_{i+1} - l_i - Δs*l_i' - (Δs^2/3)*l_i'' - (Δs^2/6)*l_{i+1}'' = 0
            row_idx = 2 * i + 1
            
            # -l_i
            rows.append(row_idx)
            cols.append(3 * i)
            data.append(-1.0)
            
            # -Δs*l_i'
            rows.append(row_idx)
            cols.append(3 * i + 1)
            data.append(-self.delta_s)
            
            # -(Δs^2/3)*l_i''
            rows.append(row_idx)
            cols.append(3 * i + 2)
            data.append(-self.delta_s**2 / 3)
            
            # +l_{i+1}
            rows.append(row_idx)
            cols.append(3 * (i + 1))
            data.append(1.0)
            
            # -(Δs^2/6)*l_{i+1}''
            rows.append(row_idx)
            cols.append(3 * (i + 1) + 2)
            data.append(-self.delta_s**2 / 6)
            
            l_list.append(0.0)
            u_list.append(0.0)
        
        A = sp.csc_matrix((data, (rows, cols)), shape=(2 * (self.n - 1), self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)
        print(f"添加连续性约束: {2*(self.n-1)}个")
    
    # ========== 约束条件4：起点等式约束 ==========
    def add_initial_state_constraints(self, l_init, dl_init, ddl_init):
        """
        添加起点状态约束
        
        参数：
        l_init: 初始横向偏移
        dl_init: 初始横向速度
        ddl_init: 初始横向加速度
        """
        constraints = [
            (0, l_init),      # l0
            (1, dl_init),     # l0'
            (2, ddl_init)     # l0''
        ]
        
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        
        for i, (col, val) in enumerate(constraints):
            rows.append(i)
            cols.append(col)
            data.append(1.0)
            l_list.append(val)
            u_list.append(val)
        
        A = sp.csc_matrix((data, (rows, cols)), shape=(3, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)
        print(f"添加起点约束: 3个")
    
    # ========== 约束条件5：车辆角点约束 ==========
    def add_vehicle_corner_constraints(self, path_boundary):
        """
        添加车辆四个角点的道路边界约束
        
        参数：
        path_boundary: (n, 2)数组，每行[l_min, l_max]
        """
        assert path_boundary.shape == (self.n, 2)
        
        # 每个点有4个角点，每个角点有上下界两个约束 → 8个约束每点
        total_constraints = 8 * self.n
        rows = []
        cols = []
        data = []
        l_list = []
        u_list = []
        
        for i in range(self.n):
            l_min = path_boundary[i, 0]
            l_max = path_boundary[i, 1]
            
            # 基础行索引
            base_row = 8 * i
            
            # === 左前角 (l + d1*l' + w/2) ===
            # 下界约束
            rows.append(base_row)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row)
            cols.append(3 * i + 1)
            data.append(self.d1)
            l_list.append(l_min - self.half_width)
            u_list.append(np.inf)
            
            # 上界约束
            rows.append(base_row + 1)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 1)
            cols.append(3 * i + 1)
            data.append(self.d1)
            l_list.append(-np.inf)
            u_list.append(l_max - self.half_width)
            
            # === 右前角 (l + d1*l' - w/2) ===
            # 下界约束
            rows.append(base_row + 2)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 2)
            cols.append(3 * i + 1)
            data.append(self.d1)
            l_list.append(l_min + self.half_width)
            u_list.append(np.inf)
            
            # 上界约束
            rows.append(base_row + 3)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 3)
            cols.append(3 * i + 1)
            data.append(self.d1)
            l_list.append(-np.inf)
            u_list.append(l_max + self.half_width)
            
            # === 左后角 (l - d2*l' + w/2) ===
            # 下界约束
            rows.append(base_row + 4)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 4)
            cols.append(3 * i + 1)
            data.append(-self.d2)
            l_list.append(l_min - self.half_width)
            u_list.append(np.inf)
            
            # 上界约束
            rows.append(base_row + 5)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 5)
            cols.append(3 * i + 1)
            data.append(-self.d2)
            l_list.append(-np.inf)
            u_list.append(l_max - self.half_width)
            
            # === 右后角 (l - d2*l' - w/2) ===
            # 下界约束
            rows.append(base_row + 6)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 6)
            cols.append(3 * i + 1)
            data.append(-self.d2)
            l_list.append(l_min + self.half_width)
            u_list.append(np.inf)
            
            # 上界约束
            rows.append(base_row + 7)
            cols.append(3 * i)
            data.append(1.0)
            rows.append(base_row + 7)
            cols.append(3 * i + 1)
            data.append(-self.d2)
            l_list.append(-np.inf)
            u_list.append(l_max + self.half_width)
        
        A = sp.csc_matrix((data, (rows, cols)), shape=(8 * self.n, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)
        print(f"添加车辆角点约束: {8*self.n}个")
    
    # ========== 构建最终约束 ==========
    def build_constraints(self):
        """合并所有约束"""
        if not self.A_list:
            return None, None, None
        
        A_total = sp.vstack(self.A_list, format='csc')
        l_total = np.array(self.l_list)
        u_total = np.array(self.u_list)
        
        print(f"\n总约束矩阵形状: {A_total.shape}")
        print(f"总约束数量: {len(l_total)}")
        
        return A_total, l_total, u_total

# ========== 使用示例 ==========
def example_usage():
    """示例用法"""
    
    # 参数设置
    n = 20
    delta_s = 1.0
    
    # 创建规划器
    planner = FrenetPathPlanner(
        n=n,
        delta_s=delta_s,
        vehicle_length=4.5,
        vehicle_width=2.0,
        max_steering_angle=0.6,
        max_steering_rate=0.5,
        wheelbase=2.8
    )
    
    # 生成示例数据
    s = np.arange(n) * delta_s
    
    # 道路边界（示例：直线道路）
    path_boundary = np.zeros((n, 2))
    path_boundary[:, 0] = -2.0  # 左边界
    path_boundary[:, 1] = 2.0   # 右边界
    
    # 参考线曲率（示例：直线道路曲率为0）
    kappa_ref = np.zeros(n)
    
    # 当前车速
    v_current = 10.0  # 10 m/s
    
    # 起点状态
    l_init = 0.0
    dl_init = 0.0
    ddl_init = 0.0
    
    print("=" * 60)
    print("开始添加约束...")
    print("=" * 60)
    
    # 添加所有约束
    planner.add_boundary_constraints(path_boundary)
    planner.add_curvature_constraints(kappa_ref, v_current)
    planner.add_jerk_constraints(v_current)
    planner.add_continuity_constraints()
    planner.add_initial_state_constraints(l_init, dl_init, ddl_init)
    planner.add_vehicle_corner_constraints(path_boundary)
    
    # 构建最终约束矩阵
    A, l, u = planner.build_constraints()
    
    print("\n" + "=" * 60)
    print("约束添加完成！")
    print(f"最终约束矩阵非零元素: {A.nnz}")
    print(f"约束矩阵密度: {A.nnz / (A.shape[0] * A.shape[1]):.4%}")
    
    return planner, A, l, u

if __name__ == "__main__":
    example_usage()
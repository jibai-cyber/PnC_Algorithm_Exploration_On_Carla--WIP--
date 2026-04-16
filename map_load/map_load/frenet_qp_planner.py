import numpy as np
import scipy.sparse as sp

try:
    import osqp
except ImportError:
    osqp = None

# 起点为软约束（l_init / dl_init / ddl_init）
DEFAULT_INIT_TOL_L = 0.05
DEFAULT_INIT_TOL_DL = 0.02
DEFAULT_INIT_TOL_DDL = 0.06


class FrenetQPPlanner:
    """
    基于三次 jerk QP 的 Frenet 路径规划器。
    """

    def __init__(
        self,
        n: int,
        delta_s: float,
        vehicle_length: float = 4.5,
        vehicle_width: float = 2.0,
        max_steering_angle: float = 0.6,  # rad
        max_steering_rate: float = 0.5,   # rad/s
        wheelbase: float = 2.8,           # m
    ) -> None:
        self.n = n
        self.delta_s = float(delta_s)
        self.total_vars = 3 * n

        # 车辆参数（用于 ego 四角点约束与曲率/jerk 上界）
        self.half_length = vehicle_length / 2.0
        self.half_width = vehicle_width / 2.0
        self.d1 = self.half_length  # 前半轴到参考点的距离
        self.d2 = self.half_length  # 后半轴到参考点的距离
        self.w = vehicle_width
        self.L = wheelbase

        # 运动学极限：wheelbase/angle 异常会导致 max_kappa 爆炸（如 -15e6）
        wb_safe = max(float(wheelbase), 0.5)
        ang_safe = np.clip(float(max_steering_angle), 0.05, 1.5)
        self.max_kappa = np.tan(ang_safe) / wb_safe
        self.max_kappa_rate = max_steering_rate / wb_safe

        # 约束累积容器
        self.A_list: list[sp.csr_matrix] = []
        self.l_list: list[float] = []
        self.u_list: list[float] = []

    # ---------- P, q 构造 ----------
    @staticmethod
    def build_frenet_qp_3n(
        n: int,
        delta_s: float,
        w_ref: float = 0.3,
        w_dl: float = 0.1,
        w_ddl: float = 0.1,
        w_dddl: float = 0.01,
        l_ref: np.ndarray | None = None,
        l_end: float = 0.0,
        l_end_dl: float = 0.0,
        l_end_ddl: float = 0.0,
        w_end_l: float = 1.0,
        w_end_dl: float = 0.1,
        w_end_ddl: float = 0.01,
    ) -> tuple[sp.csc_matrix, np.ndarray]:
        """
        构建 Frenet QP 的二次项 P 与一次项 q。
        """
        total_vars = 3 * n

        if l_ref is None:
            l_ref = np.zeros(n)
        else:
            l_ref = np.asarray(l_ref, dtype=float).reshape(-1)
            assert l_ref.shape[0] == n

        # 位置提取矩阵 E_pos (n × 3n)
        rows_pos = np.arange(n)
        cols_pos = 3 * rows_pos
        data_pos = np.ones(n)
        E_pos = sp.csc_matrix((data_pos, (rows_pos, cols_pos)),
                              shape=(n, total_vars))

        # 速度提取矩阵 E_vel (n × 3n)
        rows_vel = np.arange(n)
        cols_vel = 3 * rows_vel + 1
        data_vel = np.ones(n)
        E_vel = sp.csc_matrix((data_vel, (rows_vel, cols_vel)),
                              shape=(n, total_vars))

        # 加速度提取矩阵 E_acc (n × 3n)
        rows_acc = np.arange(n)
        cols_acc = 3 * rows_acc + 2
        data_acc = np.ones(n)
        E_acc = sp.csc_matrix((data_acc, (rows_acc, cols_acc)),
                              shape=(n, total_vars))

        # jerk 提取矩阵 E_jerk ((n-1) × 3n)：l'''_i = (l''_{i+1} - l''_i)/delta_s
        rows_jerk: list[int] = []
        cols_jerk: list[int] = []
        data_jerk: list[float] = []
        for i in range(n - 1):
            rows_jerk.append(i)
            cols_jerk.append(3 * i + 2)
            data_jerk.append(-1.0 / delta_s)
            rows_jerk.append(i)
            cols_jerk.append(3 * (i + 1) + 2)
            data_jerk.append(1.0 / delta_s)
        if n > 1:
            E_jerk = sp.csc_matrix(
                (data_jerk, (rows_jerk, cols_jerk)),
                shape=(n - 1, total_vars),
            )
        else:
            E_jerk = sp.csc_matrix((0, total_vars))

        # --- P ---
        P = sp.lil_matrix((total_vars, total_vars))

        # 横向位置
        if w_ref > 0.0:
            P += 2.0 * w_ref * (E_pos.T @ E_pos)

        # 速度项
        if w_dl > 0.0:
            P += 2.0 * w_dl * (E_vel.T @ E_vel)

        # 加速度项
        if w_ddl > 0.0:
            P += 2.0 * w_ddl * (E_acc.T @ E_acc)

        # jerk 项
        if w_dddl > 0.0 and n > 1:
            P += 2.0 * w_dddl * (E_jerk.T @ E_jerk)

        # 终点软约束
        if w_end_l > 0.0:
            idx = 3 * (n - 1)
            P[idx, idx] += 2.0 * w_end_l
        if w_end_dl > 0.0:
            idx = 3 * (n - 1) + 1
            P[idx, idx] += 2.0 * w_end_dl
        if w_end_ddl > 0.0:
            idx = 3 * (n - 1) + 2
            P[idx, idx] += 2.0 * w_end_ddl

        # --- q ---
        q = np.zeros(total_vars, dtype=float)

        # 参考线跟踪一次项
        if w_ref > 0.0:
            q += -2.0 * w_ref * (E_pos.T @ l_ref).reshape(-1)

        # 终点一次项
        if w_end_l > 0.0:
            q[3 * (n - 1)] += -2.0 * w_end_l * l_end
        if w_end_dl > 0.0:
            q[3 * (n - 1) + 1] += -2.0 * w_end_dl * l_end_dl
        if w_end_ddl > 0.0:
            q[3 * (n - 1) + 2] += -2.0 * w_end_ddl * l_end_ddl

        return P.tocsc(), q

    # ---------- 约束构造 ----------
    def _reset_constraints(self) -> None:
        self.A_list.clear()
        self.l_list.clear()
        self.u_list.clear()

    def _add_constraint_matrix(
        self, A: sp.spmatrix, l: list[float], u: list[float]
    ) -> None:
        self.A_list.append(sp.csr_matrix(A))
        self.l_list.extend(l)
        self.u_list.extend(u)

    def add_boundary_constraints(self, path_boundary: np.ndarray) -> None:
        """中心点 l_i ∈ [l_min, l_max] 约束。"""
        assert path_boundary.shape == (self.n, 2)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        l_list: list[float] = []
        u_list: list[float] = []
        for i in range(self.n):
            rows.append(i)
            cols.append(3 * i)
            data.append(1.0)
            l_list.append(float(path_boundary[i, 0]))
            u_list.append(float(path_boundary[i, 1]))
        A = sp.csc_matrix((data, (rows, cols)),
                          shape=(self.n, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)

    def add_curvature_constraints(self, kappa_ref: np.ndarray,
                                  v_current: float) -> None:
        """曲率限制：l''_i ∈ [-kappa_max - kappa_ref, kappa_max - kappa_ref]。"""
        assert kappa_ref.shape[0] == self.n
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        l_list: list[float] = []
        u_list: list[float] = []

        # 这里暂时不对 max_kappa 做速度缩放，直接用静态极限
        kappa_max = self.max_kappa

        for i in range(self.n):
            rows.append(i)
            cols.append(3 * i + 2)
            data.append(1.0)
            l_list.append(float(-kappa_max - kappa_ref[i]))
            u_list.append(float(kappa_max - kappa_ref[i]))

        A = sp.csc_matrix((data, (rows, cols)),
                          shape=(self.n, self.total_vars))
        self._add_constraint_matrix(A, l_list, u_list)

    def add_jerk_constraints(self, v_current: float) -> None:
        """三阶导（曲率变化率）限制： (l''_{i+1} - l''_i)/Δs ∈ [-jerk_max, jerk_max]。"""
        if self.n < 2:
            return
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        l_list: list[float] = []
        u_list: list[float] = []

        if v_current > 0.1:
            jerk_max = self.max_kappa_rate / v_current
        else:
            jerk_max = self.max_kappa_rate * 10.0

        for i in range(self.n - 1):
            rows.append(i)
            cols.append(3 * i + 2)
            data.append(-1.0 / self.delta_s)

            rows.append(i)
            cols.append(3 * (i + 1) + 2)
            data.append(1.0 / self.delta_s)

            l_list.append(-jerk_max)
            u_list.append(jerk_max)

        A = sp.csc_matrix(
            (data, (rows, cols)), shape=(self.n - 1, self.total_vars)
        )
        self._add_constraint_matrix(A, l_list, u_list)

    def add_continuity_constraints(self) -> None:
        """位置 / 速度连续性等式约束。"""
        if self.n < 2:
            return
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        l_list: list[float] = []
        u_list: list[float] = []

        for i in range(self.n - 1):
            # 速度连续性：l'_{i+1} - l'_i - (Δs/2) l''_i - (Δs/2) l''_{i+1} = 0
            row = 2 * i
            rows.extend([row, row, row, row])
            cols.extend([
                3 * i + 1,
                3 * i + 2,
                3 * (i + 1) + 1,
                3 * (i + 1) + 2,
            ])
            data.extend([
                -1.0,
                -self.delta_s / 2.0,
                1.0,
                -self.delta_s / 2.0,
            ])
            l_list.append(0.0)
            u_list.append(0.0)

            # 位置连续性：
            # l_{i+1} - l_i - Δs l'_i - (Δs^2/3) l''_i - (Δs^2/6) l''_{i+1} = 0
            row = 2 * i + 1
            rows.extend([row, row, row, row, row])
            cols.extend([
                3 * i,
                3 * i + 1,
                3 * i + 2,
                3 * (i + 1),
                3 * (i + 1) + 2,
            ])
            data.extend([
                -1.0,
                -self.delta_s,
                -(self.delta_s ** 2) / 3.0,
                1.0,
                -(self.delta_s ** 2) / 6.0,
            ])
            l_list.append(0.0)
            u_list.append(0.0)

        A = sp.csc_matrix(
            (data, (rows, cols)), shape=(2 * (self.n - 1), self.total_vars)
        )
        self._add_constraint_matrix(A, l_list, u_list)

    def add_initial_state_constraints(
        self,
        l_init: float,
        dl_init: float,
        ddl_init: float,
        *,
        tol_l: float = DEFAULT_INIT_TOL_L,
        tol_dl: float = DEFAULT_INIT_TOL_DL,
        tol_ddl: float = DEFAULT_INIT_TOL_DDL,
    ) -> None:
        """起点状态盒约束：l0, l0', l0'' 各允许在名义值 ±tol 内（tol=0 即退化为等式）。"""
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        l_list: list[float] = []
        u_list: list[float] = []
        specs = [
            (0, float(l_init), max(0.0, float(tol_l))),
            (1, float(dl_init), max(0.0, float(tol_dl))),
            (2, float(ddl_init), max(0.0, float(tol_ddl))),
        ]
        for row_idx, (col, val, tol) in enumerate(specs):
            rows.append(row_idx)
            cols.append(col)
            data.append(1.0)
            l_list.append(val - tol)
            u_list.append(val + tol)
        A = sp.csc_matrix(
            (data, (rows, cols)), shape=(3, self.total_vars)
        )
        self._add_constraint_matrix(A, l_list, u_list)

    def add_vehicle_corner_constraints(self, raw_path_boundary: np.ndarray) -> None:
        """
        ego 四角点约束，使用未施加 ADC bound / lat_buffer 的 raw PathBoundary。
        raw_path_boundary: (n,2)，每行 [l_min_raw, l_max_raw]
        """
        assert raw_path_boundary.shape == (self.n, 2)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        l_list: list[float] = []
        u_list: list[float] = []

        for i in range(self.n):
            l_min = float(raw_path_boundary[i, 0])
            l_max = float(raw_path_boundary[i, 1])
            base = 8 * i

            # 左前角: l + d1*l' + w/2
            # 下界: >= l_min  → l + d1*l' >= l_min - w/2
            rows.extend([base, base])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, self.d1])
            l_list.append(l_min - self.half_width)
            u_list.append(np.inf)

            # 上界: <= l_max → l + d1*l' <= l_max - w/2
            rows.extend([base + 1, base + 1])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, self.d1])
            l_list.append(-np.inf)
            u_list.append(l_max - self.half_width)

            # 右前角: l + d1*l' - w/2
            # 下界: >= l_min → l + d1*l' >= l_min + w/2
            rows.extend([base + 2, base + 2])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, self.d1])
            l_list.append(l_min + self.half_width)
            u_list.append(np.inf)

            # 上界: <= l_max → l + d1*l' <= l_max + w/2
            rows.extend([base + 3, base + 3])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, self.d1])
            l_list.append(-np.inf)
            u_list.append(l_max + self.half_width)

            # 左后角: l - d2*l' + w/2
            rows.extend([base + 4, base + 4])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, -self.d2])
            l_list.append(l_min - self.half_width)
            u_list.append(np.inf)

            rows.extend([base + 5, base + 5])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, -self.d2])
            l_list.append(-np.inf)
            u_list.append(l_max - self.half_width)

            # 右后角: l - d2*l' - w/2
            rows.extend([base + 6, base + 6])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, -self.d2])
            l_list.append(l_min + self.half_width)
            u_list.append(np.inf)

            rows.extend([base + 7, base + 7])
            cols.extend([3 * i, 3 * i + 1])
            data.extend([1.0, -self.d2])
            l_list.append(-np.inf)
            u_list.append(l_max + self.half_width)

        A = sp.csc_matrix(
            (data, (rows, cols)), shape=(8 * self.n, self.total_vars)
        )
        self._add_constraint_matrix(A, l_list, u_list)

    def build_constraints_matrix(self) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        """合并所有已添加的约束，得到总的 A, l, u。"""
        if not self.A_list:
            return sp.csc_matrix((0, self.total_vars)), np.zeros(0), np.zeros(0)
        A_total = sp.vstack(self.A_list, format="csc")
        l_total = np.array(self.l_list, dtype=float)
        u_total = np.array(self.u_list, dtype=float)
        return A_total, l_total, u_total

    # ---------- 一站式构建与求解 ----------
    def solve(
        self,
        path_boundary: np.ndarray,
        raw_path_boundary: np.ndarray,
        kappa_ref: np.ndarray,
        v_current: float,
        l_init: float,
        dl_init: float,
        ddl_init: float,
        l_ref: np.ndarray | None = None,
        l_end: float = 0.0,
        l_end_dl: float = 0.0,
        l_end_ddl: float = 0.0,
        weights: dict | None = None,
        init_state_tol: tuple[float, float, float] | None = None,
        debug_callback=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        构建并求解 QP，返回每个采样点上的 (l, dl, ddl)。
        若无可行解或 osqp 不可用，则返回 None。

        l_ref: w_ref 跟踪项的目标；每行对应 path_boundary 同一采样站。
               None 时使用 path_boundary 的横向可行域中心 (l_lower + l_upper) / 2（与 vehicle_perception
               中 ADC 后的走廊一致），使代价偏向凸空间几何中心而非 Frenet 参考线 l=0。
               若需仍贴合几何参考线，传入全零向量 np.zeros(n)。
        """
        if osqp is None:
            return None

        self._reset_constraints()

        if weights is None:
            weights = {}
        w_ref = float(weights.get("w_ref", 0.3))
        w_dl = float(weights.get("w_dl", 0.1))
        w_ddl = float(weights.get("w_ddl", 0.1))
        w_dddl = float(weights.get("w_dddl", 0.01))
        w_end_l = float(weights.get("w_end_l", 1.0))
        w_end_dl = float(weights.get("w_end_dl", 0.1))
        w_end_ddl = float(weights.get("w_end_ddl", 0.01))

        # w_ref*(l - l_ref)^2：默认跟踪 PathBoundary（ADC）各站中心，避免障碍侵入时仍强拉向 l=0
        if l_ref is None:
            assert path_boundary.shape == (self.n, 2)
            l_ref_resolved = (
                path_boundary[:, 0].astype(np.float64)
                + path_boundary[:, 1].astype(np.float64)
            ) * 0.5
        else:
            l_ref_resolved = np.asarray(l_ref, dtype=np.float64).reshape(-1)
            assert l_ref_resolved.shape[0] == self.n

        # P, q
        P, q = self.build_frenet_qp_3n(
            self.n,
            self.delta_s,
            w_ref=w_ref,
            w_dl=w_dl,
            w_ddl=w_ddl,
            w_dddl=w_dddl,
            l_ref=l_ref_resolved,
            l_end=l_end,
            l_end_dl=l_end_dl,
            l_end_ddl=l_end_ddl,
            w_end_l=w_end_l,
            w_end_dl=w_end_dl,
            w_end_ddl=w_end_ddl,
        )

        # 约束
        self.add_boundary_constraints(path_boundary)
        self.add_curvature_constraints(kappa_ref, v_current)
        self.add_jerk_constraints(v_current)
        self.add_continuity_constraints()
        if init_state_tol is None:
            self.add_initial_state_constraints(l_init, dl_init, ddl_init)
        else:
            tl, tdl, tddl = (
                float(init_state_tol[0]),
                float(init_state_tol[1]),
                float(init_state_tol[2]),
            )
            self.add_initial_state_constraints(
                l_init, dl_init, ddl_init, tol_l=tl, tol_dl=tdl, tol_ddl=tddl
            )
        self.add_vehicle_corner_constraints(raw_path_boundary)
        A, l_vec, u_vec = self.build_constraints_matrix()

        def _diag(reason: str, l_vec_diag=None, u_vec_diag=None, **extra) -> None:
            if debug_callback:
                jerk_max = (
                    self.max_kappa_rate / v_current if v_current > 0.1
                    else self.max_kappa_rate * 10.0
                )
                debug_callback(
                    reason=reason,
                    n=self.n, delta_s=self.delta_s,
                    path_boundary=path_boundary,
                    raw_path_boundary=raw_path_boundary,
                    kappa_ref=kappa_ref, v_current=v_current,
                    l_vec=l_vec_diag if l_vec_diag is not None else l_vec,
                    u_vec=u_vec_diag if u_vec_diag is not None else u_vec,
                    half_width=self.half_width, half_length=self.half_length,
                    max_kappa=self.max_kappa, jerk_max=jerk_max,
                    **extra,
                )

        # 约束边界清理：OSQP 对 inf/NaN 或 l>u 敏感
        OSQP_LARGE = 1e8
        l_vec = np.where(np.isnan(l_vec), -OSQP_LARGE, l_vec)
        u_vec = np.where(np.isnan(u_vec), OSQP_LARGE, u_vec)
        l_vec = np.where(l_vec == -np.inf, -OSQP_LARGE, l_vec)
        l_vec = np.where(l_vec == np.inf, OSQP_LARGE, l_vec)
        u_vec = np.where(u_vec == np.inf, OSQP_LARGE, u_vec)
        u_vec = np.where(u_vec == -np.inf, -OSQP_LARGE, u_vec)
        bad_idx = np.where(l_vec > u_vec)[0]
        if len(bad_idx) > 0:
            _diag("l>u", l_vec, u_vec)
            return None

        # 调用 OSQP
        prob = osqp.OSQP()
        try:
            prob.setup(P=P, q=q, A=A, l=l_vec, u=u_vec,
                       verbose=False, polish=True)
        except Exception as e:
            _diag("setup_fail")
            raise
        res = prob.solve()
        if res.info.status_val not in (1,):  # 1: solved
            status_str = getattr(res.info, "status", "") or str(res.info.status_val)
            _diag(f"osqp_status={res.info.status_val} {status_str}", res_info=res.info)
            return None

        z = res.x.reshape(-1)
        if z.size != self.total_vars:
            return None

        l_sol = z[0::3]
        dl_sol = z[1::3]
        ddl_sol = z[2::3]
        return l_sol, dl_sol, ddl_sol


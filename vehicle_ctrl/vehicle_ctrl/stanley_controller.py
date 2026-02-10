#!/usr/bin/env python3
"""Stanley控制器模块"""

import math
import numpy as np


class StanleyController:
    def __init__(self, k=3.5, epsilon=0.3, max_steer=1.22, 
                 filter_alpha=0.2, lookahead_base=1.5, lookahead_gain=0.5, curvature_feedforward_gain=0.0):
        """
        Args:
            lookahead_base: 基础前视距离 (m)
            lookahead_gain: 前视距离与车速的增益系数
            k: Stanley核心增益
            epsilon: 低速软化系数
            max_steer: 最大转向角 (rad)
            filter_alpha: 一阶低通滤波系数 (0~1，越大滤波越弱)
            curvature_feedforward_gain: 曲率前馈增益 (0~1，建议先从0.5开始)
        """
        self.k = k
        self.epsilon = epsilon
        self.max_steer = max_steer
        self.filter_alpha = filter_alpha
        self.curvature_feedforward_gain = curvature_feedforward_gain
        
        # 滤波初始化
        self.filtered_cross_track_error = 0.0
        self.filtered_heading_error = 0.0

    # 线性插值
    def _interpolate_curvature(self, path, path_curvatures, seg_start_idx, proj_ratio):
        # 边界保护：确保索引不越界
        seg_start_idx = np.clip(seg_start_idx, 0, len(path)-2)
        seg_end_idx = seg_start_idx + 1
        
        # 取线段两端点的曲率
        k0 = path_curvatures[seg_start_idx]
        k1 = path_curvatures[seg_end_idx]
        
        # 线性插值计算前视点曲率
        interpolated_kappa = k0 * (1 - proj_ratio) + k1 * proj_ratio
        
        return interpolated_kappa

    def compute_steering(self, current_x, current_y, current_yaw, current_v, path, dt=0.05, wheelbase=2.8, path_curvatures=None, start_idx=0):
        """
        计算转向角（复用前视点线段信息，优化曲率插值）
        
        Returns:
            steering_angle: 转向角 (rad)
            target_point: 目标点坐标
            cross_track_error: 横向误差
            curvature: 路径曲率 (1/m)
            nearest_idx: 最近路径点索引
        """
        if len(path) < 3:
            return 0.0, (current_x, current_y), 0.0, 0.0, 0
        
        current_pos = np.array([current_x, current_y])
        
        # 1. 找到最近路径段（原有逻辑）
        min_dist = float('inf')
        nearest_idx = 0
        target_point = path[0]
        nearest_seg_start = None
        nearest_seg_vec = None
        nearest_seg_len = None
        nearest_seg_unitvec = None


        # TODO: 优化查找最近路径段
        min_search_idx = max(0, start_idx - 50)
        max_search_idx = min(len(path)-1, start_idx + 50)
        for i in range(min_search_idx, max_search_idx):
            start_point = np.array(path[i])
            end_point = np.array(path[i + 1])
            
            line_vec = end_point - start_point
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                continue
                
            line_unitvec = line_vec / line_len
            vehicle_to_start = current_pos - start_point
            
            projection = np.dot(vehicle_to_start, line_unitvec)
            projection = np.clip(projection, 0, line_len)
            
            nearest_point = start_point + projection * line_unitvec
            dist = np.linalg.norm(current_pos - nearest_point)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                target_point = nearest_point
                nearest_seg_start = start_point
                nearest_seg_end = end_point
                nearest_seg_vec = line_vec
                nearest_seg_len = line_len
                nearest_seg_unitvec = line_unitvec
        
        # 2. 计算前视点（核心：记录线段索引和投影比例）
        lookahead_point = target_point  # 仅用于迭代，计算未来lookahead_distance的曲率
        curvature = 0.0
        cross_track_error = 0.0
        heading_error = 0.0
        
        if nearest_seg_len is not None:
            # 计算横向误差
            vehicle_vec = current_pos - nearest_seg_start
            cross_track_error = np.cross(vehicle_vec, nearest_seg_vec) / nearest_seg_len
            
            heading_error = math.atan2(nearest_seg_unitvec[1], nearest_seg_unitvec[0]) - current_yaw
            heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        self.current_heading_error = heading_error
        self.current_cross_track_error = cross_track_error
        
        if current_v < 0.6:
            steer_correction = heading_error + np.arctan2(self.k * cross_track_error, current_v + self.epsilon)
        else:
            steer_correction = heading_error + np.arctan2(self.k * cross_track_error, current_v)
        steering_angle = np.clip(steer_correction, -self.max_steer, self.max_steer)
        
        
        return steering_angle, (float(lookahead_point[0]), float(lookahead_point[1])), cross_track_error, curvature, nearest_idx

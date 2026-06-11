#!/usr/bin/env python3
"""PID控制器模块"""

import numpy as np


class PIDController:
    """PID速度控制器"""
    def __init__(self, kp, ki, kd, dt, output_limits=(-1, 1), integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_limit = integral_limit
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        
    def compute(self, error, deadband=None):
        # if deadband is not None and abs(error) < deadband:
        #     error = 0.0
        proportional = self.kp * error
        
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        integral = self.ki * self.integral
        
        derivative = self.kd * (error - self.previous_error) / self.dt
        
        output = proportional + integral + derivative
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # # 清积分
        # if integral_clear_limit is not None:
        #     if abs(error) < integral_clear_limit:
        #         self.integral /= 5.0
        
        self.previous_error = error
        self.previous_output = output
        
        return output

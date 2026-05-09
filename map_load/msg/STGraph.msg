std_msgs/Header header
float64 t_horizon
STObstacleStamp[] stamps_display
# 细时间分辨率 (如 0.1s)，供后续 QP / 细 ST 约束使用
STObstacleStamp[] stamps_qp
# 粗时间分辨率 (0.5s)，供 DP 纵向规划使用
STObstacleStamp[] stamps_dp

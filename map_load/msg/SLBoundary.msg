# SL边界消息类型 - 用于传递障碍物在Frenet坐标系中的边界信息

std_msgs/Header header

# 边界范围
float64 start_s    # s方向起始值
float64 end_s      # s方向结束值
float64 start_l    # l方向起始值
float64 end_l      # l方向结束值

# 详细边界点
float64[] boundary_s   # 边界点s坐标数组
float64[] boundary_l   # 边界点l坐标数组

# 车辆ID
string vehicle_id

uint8 decision
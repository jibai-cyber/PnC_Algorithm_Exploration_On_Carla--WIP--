uint8 DECISION_STATIC_NUDGE = 1
uint8 DECISION_STATIC_STOP = 2
uint8 DECISION_DYN_OVERTAKE = 3
uint8 DECISION_DYN_YIELD = 4
uint8 DECISION_DYN_FOLLOW = 5

string obstacle_id
bool is_static
uint8 decision
derived_object_msgs/Object object

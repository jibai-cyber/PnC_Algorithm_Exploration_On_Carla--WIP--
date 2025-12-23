# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla
from agents.tools.misc import get_speed
from scipy import linalg


class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """


    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3,
                 max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, offset, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lat_controller.change_parameters(**args_lateral)


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = waypoint.location

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class LQRLateralController():
    """
    LQRLateralController implements lateral control using Linear Quadratic Regulator (LQR).
    Uses a simplified bicycle model for vehicle dynamics.
    """

    def __init__(self, vehicle, offset=0, dt=0.03, Q=None, R=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line
            :param dt: time differential in seconds
            :param Q: State cost matrix (4x4) for [e, e_dot, theta_e, theta_e_dot]
            :param R: Control cost matrix (scalar or 1x1) for steering angle
        """
        self._vehicle = vehicle
        self._dt = dt
        self._offset = offset
        
        # Vehicle parameters (typical for CARLA vehicles)
        self._wheelbase = 2.89  # meters (distance between front and rear axles)
        
        # LQR cost matrices
        # State: [lateral_error, lateral_error_rate, heading_error, heading_error_rate]
        if Q is None:
            # Higher weight on lateral and heading errors
            self._Q = np.diag([10.0, 1.0, 10.0, 1.0])
        else:
            self._Q = Q
            
        if R is None:
            # Control effort weight (steering angle)
            self._R = np.array([[1.0]])
        else:
            self._R = R
        
        # State history for derivative calculation
        self._prev_lateral_error = 0.0
        self._prev_heading_error = 0.0
        self._initialized = False
        
    def run_step(self, waypoint):
        """
        Execute one step of lateral control using LQR to steer
        the vehicle towards a certain waypoint.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1]
        """
        return self._lqr_control(waypoint, self._vehicle.get_transform())
    
    def _lqr_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on LQR

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get vehicle state
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])
        
        # Get vehicle speed (in m/s)
        vel = self._vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        # Minimum speed to avoid singularity
        if speed < 0.1:
            speed = 0.1
        
        # Get target waypoint location
        if self._offset != 0:
            w_tran = waypoint
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(
                x=self._offset * r_vec.x,
                y=self._offset * r_vec.y
            )
        else:
            w_loc = waypoint.location
        
        # Calculate lateral error (perpendicular distance to path)
        w_vec = np.array([w_loc.x - ego_loc.x, w_loc.y - ego_loc.y, 0.0])
        
        # Calculate heading error
        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            heading_error = 0
        else:
            heading_error = math.acos(np.clip(np.dot(w_vec, v_vec) / wv_linalg, -1.0, 1.0))
        
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            heading_error *= -1.0
        
        # Lateral error is the distance to the target point
        lateral_error = np.linalg.norm(w_vec) * np.sin(heading_error)
        
        # Calculate derivatives (error rates)
        if self._initialized:
            lateral_error_rate = (lateral_error - self._prev_lateral_error) / self._dt
            heading_error_rate = (heading_error - self._prev_heading_error) / self._dt
        else:
            lateral_error_rate = 0.0
            heading_error_rate = 0.0
            self._initialized = True
        
        # Update history
        self._prev_lateral_error = lateral_error
        self._prev_heading_error = heading_error
        
        # State vector: [lateral_error, lateral_error_rate, heading_error, heading_error_rate]
        state = np.array([[lateral_error],
                         [lateral_error_rate],
                         [heading_error],
                         [heading_error_rate]])
        
        # Linearized bicycle model matrices (continuous time)
        # State: [e, e_dot, theta_e, theta_e_dot]
        # Control: [delta] (steering angle)
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, speed, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        B = np.array([
            [0],
            [0],
            [0],
            [speed / self._wheelbase]
        ])
        
        # Discretize the system using zero-order hold
        A_d, B_d = self._discretize_system(A, B, self._dt)
        
        # Solve discrete-time algebraic Riccati equation (DARE)
        try:
            K = self._solve_dare(A_d, B_d, self._Q, self._R)
        except:
            # If LQR fails, fall back to simple proportional control
            return np.clip(-1.0 * heading_error - 0.5 * lateral_error, -1.0, 1.0)
        
        # Calculate control input: u = -Kx
        steering = -np.dot(K, state)[0, 0]
        
        # Convert steering angle to normalized control [-1, 1]
        # Assuming max steering angle is about 70 degrees (1.22 radians)
        max_steering_angle = 1.22  # radians
        steering_normalized = steering / max_steering_angle
        
        return np.clip(steering_normalized, -1.0, 1.0)
    
    def _discretize_system(self, A, B, dt):
        """
        Discretize continuous-time system using zero-order hold
        
            :param A: Continuous-time state matrix
            :param B: Continuous-time input matrix
            :param dt: Time step
            :return: Discretized A_d and B_d matrices
        """
        n = A.shape[0]
        m = B.shape[1]
        
        # Method: Matrix exponential
        # [A_d  B_d] = exp([A  B] * dt)
        # [0    I  ]      [0  0]
        Mat = np.zeros((n + m, n + m))
        Mat[:n, :n] = A * dt
        Mat[:n, n:] = B * dt
        
        Mat_exp = linalg.expm(Mat)
        
        A_d = Mat_exp[:n, :n]
        B_d = Mat_exp[:n, n:]
        
        return A_d, B_d
    
    def _solve_dare(self, A, B, Q, R):
        """
        Solve the discrete-time algebraic Riccati equation (DARE)
        and return the LQR gain matrix K
        
            :param A: Discrete-time state matrix
            :param B: Discrete-time input matrix
            :param Q: State cost matrix
            :param R: Control cost matrix
            :return: LQR gain matrix K
        """
        # Solve DARE: A'XA - X - A'XB(B'XB + R)^(-1)B'XA + Q = 0
        P = linalg.solve_discrete_are(A, B, Q, R)
        
        # Calculate LQR gain: K = (B'PB + R)^(-1)B'PA
        K = np.dot(np.linalg.inv(np.dot(np.dot(B.T, P), B) + R),
                   np.dot(np.dot(B.T, P), A))
        
        return K
    
    def change_parameters(self, Q=None, R=None, dt=None):
        """Changes the LQR parameters"""
        if Q is not None:
            self._Q = Q
        if R is not None:
            self._R = R
        if dt is not None:
            self._dt = dt
        # Reset initialization
        self._initialized = False


class VehicleLQRPIDController():
    """
    VehicleLQRPIDController combines LQR lateral control with PID longitudinal control
    for vehicle control
    """

    def __init__(self, vehicle, args_lateral_lqr, args_longitudinal, offset=0, 
                 max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral_lqr: dictionary of arguments to set the LQR lateral controller:
            dt -- time step
            Q -- State cost matrix (optional)
            R -- Control cost matrix (optional)
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: lateral offset from center line
        :param max_throttle: maximum throttle value
        :param max_brake: maximum brake value
        :param max_steering: maximum steering value
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        
        # Use PID for longitudinal control
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        
        # Use LQR for lateral control
        self._lat_controller = LQRLateralController(self._vehicle, offset, **args_lateral_lqr)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking LQR lateral and PID longitudinal
        controllers to reach a target waypoint at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: control command
        """

        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control

    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_LQR(self, args_lateral_lqr):
        """Changes the parameters of the LQRLateralController"""
        self._lat_controller.change_parameters(**args_lateral_lqr)
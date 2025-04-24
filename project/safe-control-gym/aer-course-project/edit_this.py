"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
import numpy as np

from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
        
        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        ## visualization
        # Plot trajectory in each dimension and 3D.
        plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


    def planning(self, use_firmware, initial_info):
        
        # 1. Starting and goal positions (2D)
        start_xy = (self.initial_obs[0], self.initial_obs[2])
        goal_xy = (initial_info["x_reference"][0], initial_info["x_reference"][2])
        Z_FIXED = 1.0

        # 2. Build gate waypoints using pre/post offsets
        raw_gates = self.NOMINAL_GATES
        custom_order = [0, 1, 2, 3, 1, 3]
        selected_gates = [raw_gates[i] for i in custom_order]

        # Compute (pre, post) offset points for each gate
        gate_offsets = []
        for gate in selected_gates:
            gx, gy, gyaw = gate[0], gate[1], gate[5]
            pre, post = ecu.offset_from_yaw(gx, gy, gyaw, dist=0.25)
            gate_offsets.append((pre, post))

        # Base obstacles from YAML
        base_obstacles = [(obs[0], obs[1], 0.4) for obs in self.NOMINAL_OBSTACLES]
        bounds = [(-3.5, 3.5), (-3.5, 3.5)]

        waypoints_2d = [start_xy]

        for i in range(len(gate_offsets)):
            pre, post = gate_offsets[i]
            print(f"[INFO] Planning from {waypoints_2d[-1]} to pre-gate {i}: {pre}")

            # ---------- Plan to pre-gate ----------
            gate_buffer_obstacles = []
            seen = set()

            for j, other_gate in enumerate(selected_gates):
                gx, gy, gyaw = other_gate[0], other_gate[1], other_gate[5]
                if j == i or (gx, gy) in seen:
                    continue
                seen.add((gx, gy))
                gate_buffer_obstacles.extend(ecu.get_gate_edge_buffers(gx, gy, gyaw))

            dynamic_obstacles = base_obstacles + gate_buffer_obstacles

            path = ecu.plan_path_rrtstar(waypoints_2d[-1], pre, dynamic_obstacles, bounds,
                                         max_iter=5000, step_size=0.1,
                                         goal_sample_rate=0.4, search_radius=1.5)
            path = ecu.shortcut_smooth(path, dynamic_obstacles, lambda p1, p2, obs: ecu.is_line_collision_free(p1, p2, obs))
            waypoints_2d.extend(path[1:])

            # ---------- Plan through gate (pre → post) ----------
            if ecu.is_line_collision_free(pre, post, base_obstacles):
                waypoints_2d.extend([post])
            else:
                path = ecu.plan_path_rrtstar(pre, post, base_obstacles, bounds,
                                             max_iter=1000, step_size=0.15,
                                             goal_sample_rate=0.3, search_radius=1.5)
                if len(path) < 2:
                    raise RuntimeError(f"Failed to pass through gate {i}")
                # path = ecu.shortcut_smooth(path, dynamic_obstacles, ecu.is_line_collision_free)
                waypoints_2d.extend(path[1:])

        # ---------- Final segment to goal ----------
        path = ecu.plan_path_rrtstar(waypoints_2d[-1], goal_xy, base_obstacles, bounds,
                                     max_iter=1500, step_size=0.15,
                                     goal_sample_rate=0.3, search_radius=1.5)
        if len(path) < 2:
            raise RuntimeError("Failed to find path to final goal")
        path = ecu.shortcut_smooth(
            path,
            base_obstacles,
            lambda p1, p2, obs: ecu.is_line_collision_free(p1, p2, obs)
        )
        waypoints_2d.extend(path[1:])

        # 5. Convert to 3D
        waypoints = np.array([(x, y, Z_FIXED) for x, y in waypoints_2d])
        self.waypoints = waypoints

        # 6. Estimate tangents for Hermite interpolation
        tangent_scale = 0.65
        tangents = np.zeros_like(waypoints)
        tangents[1:-1] = (waypoints[2:] - waypoints[:-2]) * 0.5 * tangent_scale
        tangents[0] = (waypoints[1] - waypoints[0]) * tangent_scale
        tangents[-1] = (waypoints[-1] - waypoints[-2]) * tangent_scale

        # 7. Hermite interpolation using NumPy only
        samples_per_segment = 20
        ref_x, ref_y, ref_z = [], [], []

        for i in range(len(waypoints) - 1):
            p0, p1 = waypoints[i], waypoints[i+1]
            v0, v1 = tangents[i], tangents[i+1]

            for s in np.linspace(0, 1, samples_per_segment, endpoint=False):
                h00 = 2*s**3 - 3*s**2 + 1
                h10 = s**3 - 2*s**2 + s
                h01 = -2*s**3 + 3*s**2
                h11 = s**3 - s**2
                point = h00*p0 + h10*v0 + h01*p1 + h11*v1
                ref_x.append(point[0])
                ref_y.append(point[1])
                ref_z.append(point[2])

        self.ref_x = np.array(ref_x)
        self.ref_y = np.array(ref_y)
        self.ref_z = np.array(ref_z)

        # # Unpack back to self.ref_x/y/z
        smoothed_path = ecu.ccma_path(np.stack([self.ref_x, self.ref_y], axis=1), w_ma=5, w_cc=5)
        self.ref_x, self.ref_y = smoothed_path[:, 0], smoothed_path[:, 1]

        t_scaled = np.linspace(0, len(self.ref_x) / self.CTRL_FREQ, len(self.ref_x))

        for x, y in zip(self.ref_x, self.ref_y):
            for ox, oy, r in base_obstacles:
                d = np.hypot(x - ox, y - oy)
                if d < r:
                    print(f"[WARNING] Trajectory too close to obstacle at ({ox},{oy}), dist={d:.2f}, r={r}")

        return t_scaled





    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        # [INSTRUCTIONS] 
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)

        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        # [INSTRUCTIONS] Example code for using cmdFullState interface   
        elif iteration >= 3 * self.CTRL_FREQ and iteration < 20 * self.CTRL_FREQ:
            useVelandAcc = False
            
            if useVelandAcc:
                dt = 1.0 / self.CTRL_FREQ
                step = min(iteration - 3 * self.CTRL_FREQ, len(self.ref_x) - 2)
                next_step = step + 1
                prev_step = max(step - 1, 0)

                # Position at current step
                target_pos = np.array([
                    self.ref_x[step],
                    self.ref_y[step],
                    self.ref_z[step]
                ])

                # Velocity: finite difference
                target_vel = np.array([
                    (self.ref_x[next_step] - self.ref_x[step]) / dt,
                    (self.ref_y[next_step] - self.ref_y[step]) / dt,
                    (self.ref_z[next_step] - self.ref_z[step]) / dt
                ])
                # Clamp velocity
                vel_mag = np.linalg.norm(target_vel)
                if vel_mag > 1.0:
                    target_vel = target_vel / vel_mag * 1.0

                # Acceleration: second-order finite difference
                target_acc = np.array([
                    (self.ref_x[next_step] - 2 * self.ref_x[step] + self.ref_x[prev_step]) / dt**2,
                    (self.ref_y[next_step] - 2 * self.ref_y[step] + self.ref_y[prev_step]) / dt**2,
                    (self.ref_z[next_step] - 2 * self.ref_z[step] + self.ref_z[prev_step]) / dt**2
                ])
                acc_mag = np.linalg.norm(target_acc)
                if acc_mag > 1.5:
                    target_acc = target_acc / acc_mag * 1.5

                # Optional: align yaw with direction of travel
                if np.linalg.norm(target_vel[:2]) > 1e-3:
                    target_yaw = 0
                else:
                    target_yaw = 0.

                target_rpy_rates = np.zeros(3)

                command_type = Command(1)  # cmdFullState
                
            else: # use the default controller
                step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.
                target_rpy_rates = np.zeros(3)

                command_type = Command(1)  # cmdFullState.
            
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
            
        elif iteration == 20*self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

        elif iteration == 22*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == 25*self.CTRL_FREQ-1:
            command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0

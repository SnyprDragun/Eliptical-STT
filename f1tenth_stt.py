import rclpy
import time
import threading 
import numpy as np
import matplotlib as plt
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import math

class F1TenthController(Node):
    """
    ROS 2 Node for F1Tenth vehicle control.
    Subscribes to /odom, calculates position error against a dummy target, 
    and publishes a proportional control output to /ackerman_cmd.
    """

    def __init__(self, C, dimension):
        super().__init__('tt_controller')
        self.get_logger().info("STT Controller Node started.")
        self.dimension = dimension
        self.start = 0.0
        self.end = 15.0
        self.step = 0.01
        self.t_values = np.arange(self.start, self.end + self.step, self.step)
        self.t_index = 0
        self.max_iterations = 1
        self.iteration_count = 0

        # Path Coefficients (User-provided)
        self.C1 = [2.4999999999999942, -0.8958333333333298, 0.2604166666666664, 2.4999999999999942, -1.1690821256038582, 0.3059581320450878]
        self.C2 = [-9.301776922105795, 3.038092307368601, -0.06741047005849447, -13.587862318840559, 4.193538647342993, -0.14092693236714973]
        self.C3 = [-142.94394839965668, 27.250576615735664, -1.4807933588422957, 0.026126252682958113, -206.45536823414497, 41.80834847394456, -2.5624175313412856, 0.051338311130398384]


        self.Kp_linear = 0.5
        self.Kp_angular = 1.0
        self.max_speed = 2.0
        self.max_steer = 0.6
        self.distance_tolerance = 0.05

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.t_history = []
        self.x_target_history = []
        self.y_target_history = []
        self.x_current_history = []
        self.y_current_history = []

        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.odom_subscription
        self.publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        
        self.timer_period = 0.01 
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # --- Plotting Setup ---
        self.setup_plot()
        # Start a dedicated thread for plot animation/updates to avoid blocking ROS spin
        self.plot_thread = threading.Thread(target=self.plot_thread_run)
        self.plot_thread.start()

    def real_gammas(self, t, C, degree):
        '''method to calculate tube equations'''
        real_tubes = np.zeros(self.dimension)

        for i in range(self.dimension):
            power = 0
            for j in range(degree + 1):
                real_tubes[i] += ((C[j + i * (degree + 1)]) * (t ** power))
                power += 1
        return real_tubes

    def quaternion_to_yaw(self, orientation):
        """
        Converts a quaternion (w, x, y, z) to yaw (rotation around Z-axis)
        """
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return yaw

    def odom_callback(self, msg):
        """
        Callback function for the /odom topic
        """
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(orientation)

    def control_loop(self):
        """
        Timer-driven function to execute the path tracking and control logic.
        This function contains the logic you provided.
        """
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = 'base_link'

        # Check for loop termination
        if self.iteration_count >= self.max_iterations:
            self.get_logger().info('STT control finished. Stopping node.')
            ackermann_cmd.drive.speed = 0.0
            ackermann_cmd.drive.steering_angle = 0.0
            self.publisher.publish(ackermann_cmd)
            self.control_timer.cancel()
            self.destroy_node() # Destroy the node to stop the script
            return

        if self.t_index >= len(self.t_values):
            self.t_index = 0
            self.iteration_count += 1
            if self.iteration_count >= self.max_iterations:
                return

        # Get the current point on the path
        t = self.t_values[self.t_index]
        self.t_index += 1

        # Determine the current tube and degree based on t
        tube = None
        degree = 0
        linear_velocity = 0.0
        steering_angle = 0.0
        status_message = ""
        
        if 0 <= t < 6:
            tube = self.C1
            degree = 2
        elif 6 <= t < 14:
            tube = self.C2
            degree = 2
        elif 14 <= t < 18:
            tube = self.C3
            degree = 3
        else:
            self.get_logger().info('Time out of path range. Stopping.')
            tube = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Failsafe
            linear_velocity = 0.0
            steering_angle = 0.0
            
        if tube:
            # Calculate target position
            gamma = self.real_gammas(t, tube, degree)
            target_x = gamma[0]
            target_y = gamma[1]
            
            # --- P-Controller Implementation ---
            error_x = target_x - self.current_x
            error_y = target_y - self.current_y

            distance_error = math.sqrt(error_x**2 + error_y**2)
            target_heading = math.atan2(error_y, error_x)

            error_yaw = target_heading - self.current_yaw
            error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))

            # Linear Velocity Control
            if distance_error < self.distance_tolerance:
                linear_velocity = 0.0
                status_message = "Target reached. Stopping vehicle."
            else:
                linear_velocity = self.Kp_linear * distance_error
                linear_velocity = max(-self.max_speed, min(linear_velocity, self.max_speed))

                if abs(error_yaw) > math.pi / 6:
                    linear_velocity *= 0.5 
                
                status_message = (
                    f'Current Iteration: {self.iteration_count} | '
                    f'T={t:.1f} | Dist Err={distance_error:.2f} | Vel={linear_velocity:.2f} m/s | Steering={error_yaw:.2f} rad'
                )

            # Angular (Steering) Control
            if abs(linear_velocity) > 0.0:
                steering_angle = self.Kp_angular * error_yaw
                steering_angle = max(-self.max_steer, min(steering_angle, self.max_steer))
            else:
                steering_angle = 0.0

            # Store history for plotting
            self.t_history.append(t + self.iteration_count * 18) # Adjust time for full path duration
            self.x_target_history.append(target_x)
            self.y_target_history.append(target_y)
            self.x_current_history.append(self.current_x)
            self.y_current_history.append(self.current_y)
            
        else:
            # If t is outside defined range, stop
            linear_velocity = 0.0
            steering_angle = 0.0
            status_message = 'TUBE ERROR: Time out of range.'


        # Publish and Log
        ackermann_cmd.drive.speed = linear_velocity
        ackermann_cmd.drive.steering_angle = steering_angle
        self.publisher.publish(ackermann_cmd)
        self.get_logger().info(status_message)
        
        # Call plot update (only if plotting is running)
        self.update_plot_data()

    def setup_plot(self):
        """Initializes the matplotlib figure and subplots."""
        plt.ion() # Turn on interactive mode
        self.fig = plt.figure(figsize=(10, 5))
        
        # Left subplot (X vs Y) - spanning both rows
        self.ax1 = self.fig.add_subplot(1, 2, 1) # Simple 1 row, 2 column, index 1
        self.line_path, = self.ax1.plot([], [], 'r--', label='Target Path')
        self.line_car, = self.ax1.plot([], [], 'b-', label='Current Position')
        self.ax1.set_title('Global Trajectory (X vs Y)')
        self.ax1.set_xlabel('X Position (m)')
        self.ax1.set_ylabel('Y Position (m)')
        self.ax1.set_aspect('equal', adjustable='box')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Right subplots (X vs T and Y vs T)
        # Using add_subplot(rows, cols, index) where we treat the right as two rows
        self.ax2 = self.fig.add_subplot(2, 2, 2) # Top Right
        self.line_xt, = self.ax2.plot([], [], 'r--', label='Target X')
        self.line_xc, = self.ax2.plot([], [], 'b-', label='Current X')
        self.ax2.set_title('X Trajectory over Time')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('X (m)')
        self.ax2.legend()
        self.ax2.grid(True)

        self.ax3 = self.fig.add_subplot(2, 2, 4) # Bottom Right
        self.line_yt, = self.ax3.plot([], [], 'r--', label='Target Y')
        self.line_yc, = self.ax3.plot([], [], 'b-', label='Current Y')
        self.ax3.set_title('Y Trajectory over Time')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Y (m)')
        self.ax3.legend()
        self.ax3.grid(True)
        
        # Adjust layout to prevent overlap
        self.fig.tight_layout()

    def update_plot_data(self):
        """Updates the internal data structures for the plot."""
        # This is called by the control_loop (ROS thread) to store new data
        pass # Data is already appended in control_loop

    def plot_thread_run(self):
        """
        Dedicated thread function to manage Matplotlib updates.
        This must be separate from rclpy.spin().
        """
        while rclpy.ok() and not self.control_timer.is_canceled():
            if self.t_history:
                try:
                    # Update X vs Y Plot
                    self.line_path.set_data(self.x_target_history, self.y_target_history)
                    self.line_car.set_data(self.x_current_history, self.y_current_history)
                    
                    # Auto-scale axis limits for X vs Y
                    all_x = self.x_current_history + self.x_target_history
                    all_y = self.y_current_history + self.y_target_history
                    if all_x and all_y:
                        self.ax1.set_xlim(min(all_x) - 1, max(all_x) + 1)
                        self.ax1.set_ylim(min(all_y) - 1, max(all_y) + 1)

                    # Update X vs T Plot
                    self.line_xt.set_data(self.t_history, self.x_target_history)
                    self.line_xc.set_data(self.t_history, self.x_current_history)
                    self.ax2.set_xlim(self.t_history[0], self.t_history[-1] + 1)
                    self.ax2.set_ylim(min(self.x_current_history + self.x_target_history) - 0.5, 
                                      max(self.x_current_history + self.x_target_history) + 0.5)

                    # Update Y vs T Plot
                    self.line_yt.set_data(self.t_history, self.y_target_history)
                    self.line_yc.set_data(self.t_history, self.y_current_history)
                    self.ax3.set_xlim(self.t_history[0], self.t_history[-1] + 1)
                    self.ax3.set_ylim(min(self.y_current_history + self.y_target_history) - 0.5, 
                                      max(self.y_current_history + self.y_target_history) + 0.5)
                    
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                except Exception as e:
                    # Catch exceptions during plotting update if the list is being modified
                    # simultaneously, which can happen in a multi-threaded setup.
                    self.get_logger().debug(f"Plotting error: {e}") 
                    pass
            
            time.sleep(0.05) # Update plot at 20 Hz

    def destroy_node(self):
        """Override to ensure the plot is closed when the node is destroyed."""
        self.get_logger().info("Closing plot window.")
        plt.close(self.fig)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    controller_node = F1TenthController()
    
    try:
        # rclpy.spin() blocks the main thread
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

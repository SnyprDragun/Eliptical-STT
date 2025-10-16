import rclpy
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

    def __init__(self):
        super().__init__('f1tenth_controller')
        self.get_logger().info("F1Tenth Controller Node started.")

        # --- Parameters and State ---
        self.target_x = 5.0  # Dummy Ideal Target X Position (meters)
        self.Kp_linear = 0.5 # Proportional Gain for Linear Velocity
        self.max_speed = 2.0 # Maximum allowed linear speed (m/s)

        # Initialize current state (optional, but good practice)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0


        # --- ROS 2 Setup ---

        # 1. Subscriber: /odom (nav_msgs/msg/Odometry)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.odom_subscription # prevent unused variable warning

        # 2. Publisher: /ackerman_cmd (ackermann_msgs/msg/AckermannDriveStamped)
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            '/ackerman_cmd',
            10)

    def quaternion_to_yaw(self, orientation):
        """
        Converts a quaternion (w, x, y, z) to yaw (rotation around Z-axis).
        """
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        
        # Calculate yaw (atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z))
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return yaw


    def odom_callback(self, msg):
        """
        Callback function for the /odom topic.
        1. Reads current position.
        2. Calculates error against target.
        3. Publishes a velocity command.
        """
        
        # --- 1. Get Current Position ---
        
        # Position (x, y)
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Orientation (convert quaternion to yaw for simple control)
        orientation = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(orientation)
        
        # self.get_logger().info(f'Odom received: X={self.current_x:.2f}, Yaw={self.current_yaw:.2f}')

        # --- 2. Calculate Error and Control Output (P-Controller) ---

        # Calculate position error in the X-direction
        error_x = self.target_x - self.current_x
        
        # Calculate proportional linear velocity command
        linear_velocity = self.Kp_linear * error_x
        
        # Clamp velocity to the maximum allowed speed
        linear_velocity = max(-self.max_speed, min(linear_velocity, self.max_speed))

        # Stop if we are very close to the target
        if abs(error_x) < 0.1:
            linear_velocity = 0.0
        
        # NOTE ON STEERING: For a simple controller targeting a single X point, 
        # a more complex steering logic (like pure pursuit or PID on yaw error)
        # would be needed. Here we keep a dummy, fixed steering angle.
        steering_angle = 0.0 # Keep steering straight for a simple X-target drive
        
        # --- 3. Publish Velocity Command ---

        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()
        ackermann_cmd.header.frame_id = 'base_link'  # Standard frame for commands

        # Set the drive fields
        ackermann_cmd.drive.speed = linear_velocity
        ackermann_cmd.drive.steering_angle = steering_angle
        
        self.publisher_.publish(ackermann_cmd)
        
        if abs(linear_velocity) > 0.01:
            self.get_logger().info(
                f'Target X={self.target_x:.2f}, Current X={self.current_x:.2f}, '
                f'Error X={error_x:.2f}, Cmd Vel={linear_velocity:.2f} m/s'
            )
        else:
             self.get_logger().info("Target reached. Stopping vehicle.")


def main(args=None):
    rclpy.init(args=args)
    controller_node = F1TenthController()
    
    try:
        # Keep the node running until it is manually shut down
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        pass
    
    # Destroy the node explicitly
    controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

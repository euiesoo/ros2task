import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class ActorCollision(Node):
    def __init__(self):
        super().__init__('actor_collision')

        self.declare_parameter('actor_name', "")
        self.actor_name = self.get_parameter('actor_name').get_parameter_value().string_value
        self.declare_parameter('collision_name', "")
        self.collision_name = self.get_parameter('collision_name').get_parameter_value().string_value

        self.model_states_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        while not self.set_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"service ({self.set_state_cli.srv_name}) not available, waiting again...")
    
    def model_states_callback(self, msg: ModelStates):
        try:
            names = msg.name
            idx = names.index(self.actor_name)
            actor_position = msg.pose[idx].position

            req = SetEntityState.Request()
            req.state.name = self.collision_name
            req.state.pose = Pose()
            req.state.pose.position.x = actor_position.x
            req.state.pose.position.y = actor_position.y
            req.state.pose.position.z = actor_position.z
            req.state.reference_frame = "world"
            self.set_state_cli.call_async(req) # fire-and-forget
        except ValueError:
            self.get_logger().info("Model name not found in model states.")  # Handle the case where the model name is not found

def main(args=None):
    rclpy.init(args=args)
    node = ActorCollision()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
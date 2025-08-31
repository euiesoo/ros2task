import time

import rclpy

from self_drive_sim.simulation.gazebo_env import GazeboEnv

def main():
    rclpy.init(args=None)

    env = GazeboEnv(debug_mode=True)

    time.sleep(3)

    num_episodes = 1
    num_steps = 18000

    env.get_logger().info("Debug Mode Start")
    for ep in range(num_episodes):
        env.reset()
        env.get_logger().info("Env reset done")

        init_action = (0, 0, 0)
        step_results = env.step([init_action])
        for i in range(num_steps):
            if i % 50 == 0 : # display pollution (debug)
                pol_list = []
                for i in range(env.pm.num_rooms):
                    pol_list.append(f"{env.fm.room_names[i]}: {env.pm.pollution[i]:.1f}")
                pol_str = f"Current pollutions: {', '.join(pol_list)}"
                env.get_logger().info(pol_str)

            step_results = env.step([(0, 0, 0)])
            _, _, terminated, done = step_results[0]

            if terminated:
                env.get_logger().info(f"Terminated (Hard collision detected)")
                break

            if done:
                env.get_logger().info(f"Done")
                break

        data = env.get_score_data(0)
        env.get_logger().info(data['message'])

    env.get_logger().info(f"Simulation Done")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import time

import rclpy

from self_drive_sim.simulation.gazebo_env import GazeboEnv
from self_drive_sim.agent.agent import Agent

def main():
    rclpy.init(args=None)

    env = GazeboEnv()
    agent = Agent(env.get_logger().info)

    time.sleep(3)

    num_episodes = 10
    num_steps = 18000

    env.get_logger().info("Training Start")
    for ep in range(num_episodes):
        env.reset()
        env.get_logger().info("Env reset done")
        env.get_logger().info(f"Episode {ep+1} start")
        
        agent.initialize_map(env.get_map_info(0))

        init_action = (0, 0, 0)
        step_results = env.step([init_action])
        observation = step_results[0][0]
        info = step_results[0][1]
        for i in range(num_steps):
            if i % 50 == 0 : # display pollution (debug)
                pol_list = []
                for i in range(env.pm.num_rooms):
                    pol_list.append(f"{env.fm.room_names[i]}: {env.pm.pollution[i]:.1f}")
                pol_str = f"Current pollutions: {', '.join(pol_list)}"
                env.get_logger().info(pol_str)
            
            action = agent.act(observation)

            step_results = env.step([action])
            next_observation, next_info, terminated, done = step_results[0]

            agent.learn(
                observation,
                info,
                action,
                next_observation,
                next_info,
                terminated,
                done,
                )

            observation = next_observation
            info = next_info

            if terminated:
                env.get_logger().info(f"Terminated (Hard collision detected)")
                break

            if done:
                env.get_logger().info(f"Done")
                break

        env.get_logger().info(f"Episode {ep+1} over ({env.steps * 0.1:.1f}s)")
        agent.reset()

    env.get_logger().info(f"Training Done")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
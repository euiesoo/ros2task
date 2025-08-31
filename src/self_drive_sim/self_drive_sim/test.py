import sys
import time
import json

import rclpy

from self_drive_sim.simulation.gazebo_env import GazeboEnv
from self_drive_sim.agent.agent import Agent

def main():
    rclpy.init(args=None)

    env = GazeboEnv()
    agent = Agent(env.get_logger().info)

    time.sleep(3)

    num_episodes = 3
    num_steps = 18000 # 30분 타임아웃

    data_list = []
    score_list = []

    env.get_logger().info("Test Start")
    for ep in range(num_episodes):
        env.reset()
        env.get_logger().info("Env reset done")
        env.get_logger().info(f"Episode {ep+1} start")
        
        agent.initialize_map(env.get_map_info(0))

        init_action = (0, 0, 0)
        step_results = env.step([init_action])
        observation = step_results[0][0]
        for i in range(num_steps):
            if i % 50 == 0 : # display pollution (debug)
                pol_list = []
                for i in range(env.pm.num_rooms):
                    pol_list.append(f"{env.fm.room_names[i]}: {env.pm.pollution[i]:.1f}")
                pol_str = f"Current pollutions: {', '.join(pol_list)}"
                env.get_logger().info(pol_str)
            
            action = agent.act(observation)

            step_results = env.step([action])
            observation, _, terminated, done = step_results[0]

            if terminated:
                env.get_logger().info(f"Terminated (Hard collision detected)")
                break
            
            if done:
                env.get_logger().info(f"Done")
                break

        env.get_logger().info(f"Episode {ep+1} over ({env.steps * 0.1:.1f}s)")

        data = env.get_score_data(0)
        score = data['score']
        env.get_logger().info(data['message'])
        
        data_list.append(data)
        score_list.append(score)
        agent.reset()

    env.get_logger().info(f"Test Done")
    average_score = sum(score_list) / num_episodes
    env.get_logger().info(f"Average Score: {average_score:.4f}")

    # save file
    result = {
        "random_seed": env.random_seed,
        "average_score": average_score,
        "data": data_list
    }

    with open(f"{env.result_file_name}.json", "w") as f:
        json.dump(result, f, indent=2)
        env.get_logger().info("Result file saved.")
    
    rclpy.shutdown()
    sys.exit(0) # signal to launch file

if __name__ == '__main__':
    main()
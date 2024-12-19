import gymnasium as gym

# from gymnasium import
import ale_py

gym.register_envs(ale_py)


def main():
    env = gym.make("ALE/Asteroids-v5", render_mode="human")

    observation, info = env.reset()
    done = False
    while not done:

        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        done = terminated or truncated
    env.close()


if __name__ == "__main__":
    main()

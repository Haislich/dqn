import gymnasium as gym
import ale_py
import pygame
import numpy as np

WIDTH = 800
HEIGHT = 800

gym.register_envs(ale_py)


def main():
    environment = gym.make("ALE/Atlantis-v5", render_mode="human")
    observation, info = environment.reset()

    done = False
    clock = pygame.time.Clock()

    while not done:

        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        environment.render()

    environment.close()


if __name__ == "__main__":
    main()

import gymnasium as gym
import ale_py
import pygame
import numpy as np

WIDTH = 800
HEIGHT = 800

gym.register_envs(ale_py)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    environment = gym.make("ALE/Atlantis-v5", render_mode="rgb_array")
    observation, info = environment.reset()

    done = False
    clock = pygame.time.Clock()

    while not done:

        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                done = True

        # Scale up the image to your desired dimensions
        frame = np.array(observation)
        frame = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        frame = pygame.transform.scale(frame, (WIDTH, HEIGHT))
        screen.blit(frame, (0, 0))

        pygame.display.flip()
        clock.tick(22)

    environment.close()
    pygame.quit()  # type : ignore


if __name__ == "__main__":
    main()

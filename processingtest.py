import math
import pygame

# initialize Pygame
pygame.init()

# set up screen dimensions
screen_width = 1080
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))

# set up circle dimensions
circle_color = (255, 0, 0)
circle_x = screen_width // 2
circle_y = screen_height // 2
circle_radius = 10

# main loop
while True:
    # increase circle radius
    circle_radius += 2

    # check if circle has filled the entire screen
    if circle_radius >= math.sqrt(screen_width**2 + screen_height**2):
        pygame.quit()
        exit()

    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # draw circle
    pygame.draw.circle(screen, circle_color, (circle_x, circle_y), circle_radius)

    # update screen
    pygame.display.update()
    screen.fill((255, 255, 255))

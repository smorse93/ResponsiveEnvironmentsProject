import pygame

# initialize Pygame
pygame.init()

# set up screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# set up circle dimensions
circle_radius = 25
circle_color = (255, 0, 0)
circle_x = screen_width // 2
circle_y = screen_height // 2
circle_speed_x = 5
circle_speed_y = 5

# set up clock to limit FPS
clock = pygame.time.Clock()

# main loop
while True:
    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # move circle
    circle_x += circle_speed_x
    circle_y += circle_speed_y

    # check for collision with edges of screen
    if circle_x - circle_radius < 0 or circle_x + circle_radius > screen_width:
        circle_speed_x *= -1
    if circle_y - circle_radius < 0 or circle_y + circle_radius > screen_height:
        circle_speed_y *= -1

    # draw circle and update screen
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, circle_color, (circle_x, circle_y), circle_radius)
    pygame.display.update()

    # limit FPS
    clock.tick(60)

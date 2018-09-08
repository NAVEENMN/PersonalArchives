import pygame
import numpy as np

display_width = 80
display_height = 80
latent = 12

class window():
    def __init__(self, graph):
        self.graph = graph
        pygame.init()
        self.display = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption('image space')
        white = (255, 255, 255)
        self.display.fill(white)
        self.running = True
        self.mean = 0.0
        self.sd = 1.0
        self.delta = 0.001

    def handle_keys(self):
        pressed = pygame.key.get_pressed()
        is_pressed = False
        if pressed[pygame.K_w]:
            self.mean += self.delta
            if self.mean > 2.0:
                self.mean = 0.0
            is_pressed = True
        if pressed[pygame.K_s]:
            self.mean -= self.delta
            if self.mean < -1.0:
                self.mean = 0.0
            is_pressed = True
        if pressed[pygame.K_d]:
            self.sd += self.delta
            if self.sd > 2.0:
                self.sd = 1.0
            is_pressed = True
        if pressed[pygame.K_a]:
            self.sd -= self.delta
            if self.sd < -2.0:
                self.sd = 1.0
            is_pressed = True
        if is_pressed:
            print("mean {} sd {}".format(self.mean, self.sd))
        return is_pressed


    def sample_a_image(self):
        latent_sample = np.random.normal(self.mean, self.sd, latent)
        image = self.graph.generate_a_image(latent_sample)
        image = image * 255.0
        image = np.reshape(image, [28, 28])
        #image_arr = pygame.surfarray.pixels2d(image)
        image_surface = pygame.surfarray.make_surface(image)
        image_surface = pygame.transform.scale2x(image_surface)
        return image_surface

    def update_display(self):
        image_surface = self.sample_a_image()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            is_pressed = self.handle_keys()
            if is_pressed:
                image_surface = self.sample_a_image()
            self.display.blit(image_surface, (10, 10))
            pygame.display.update()

    def close_window(self):
        pygame.quit()

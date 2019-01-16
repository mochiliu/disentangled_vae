# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def rgb_int2tuple(rgbint):
    return (rgbint // 256 // 256 % 256, rgbint // 256 % 256, rgbint % 256)

def rgb2int(rint, gint, bint):
    return int(rint*256*256 + (gint*256) + bint)

def color_drift(color, distance):
    drift_direction = np.random.uniform(low=-1,high=1,size=3)
    drift_direction = drift_direction / np.linalg.norm(drift_direction)
    newcolor = np.int32(np.abs(drift_direction*distance + color))
    newcolor[newcolor > 255] = 255
    if not np.any(newcolor):
        # rgb all 0
        nonzero_array_index = int(np.random.uniform(low=0,high=3))
        newcolor[nonzero_array_index] = 1
    return newcolor
    
def get_new_color(old_colors, mutation_distance):
    number_of_old_colors = len(old_colors)
    old_colors_rgb = np.zeros((number_of_old_colors,3))
    for old_color_index in range(number_of_old_colors):
        old_color_tuple = rgb_int2tuple(old_colors[old_color_index])
        old_colors_rgb[old_color_index, 0] = old_color_tuple[0]
        old_colors_rgb[old_color_index, 1] = old_color_tuple[1]
        old_colors_rgb[old_color_index, 2] = old_color_tuple[2]
    avg_old_color = np.mean(old_colors_rgb, axis=0)
    newcolor = color_drift(avg_old_color, mutation_distance)
    return rgb2int(newcolor[0],newcolor[1],newcolor[2])

def randomGrid(N): 
    """returns a grid of NxN random values"""
    grid = np.zeros((N,N), dtype=int)
    for i in range(N): 
        for j in range(N): 
            if np.random.uniform() < 0.1:
                # cell alive
                grid[i,j] = int(np.random.uniform(low=1, high=(256*256*256)-1))
    return grid

def addGlider(i, j, grid): 
    """adds a glider with top left cell at (i, j)"""
    color = int(np.random.uniform(low=1, high=(256*256*256)-1))
    glider = np.array([[0, 0, color], [color, 0, color], [0, color, color]]) 
    rotation_number = np.random.randint(4)
    glider = np.rot90(glider, rotation_number)
    grid[i:i+3, j:j+3] = glider 


def update(grid): 
    # copy grid since we require 8 neighbors 
    # for calculation and we go line by line 
    N = len(grid)
    chance_of_glider = 0.05
    threshold_for_adding_glider = 0.1
    
    newGrid = grid.copy() 
    
    if np.count_nonzero(grid) / (N*N) < threshold_for_adding_glider and np.random.uniform() < chance_of_glider:
        addGlider(np.random.randint(N-3), np.random.randint(N-3), newGrid)
    else:
        for i in range(N): 
            for j in range(N): 
                # compute 8-neghbor sum 
                # using toroidal boundary conditions - x and y wrap around 
                # so that the simulaton takes place on a toroidal surface. 
                eight_neighbors = np.array([ grid[i, (j-1)%N], grid[i, (j+1)%N],
                						grid[(i-1)%N, j], grid[(i+1)%N, j],
                						grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
                						grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N] ]) 
        
                nonzero_array = np.nonzero(eight_neighbors)[0]
                total = len(nonzero_array)
                # apply Conway's rules 
                if grid[i, j] > 0: 
                    #cell currently alive
                    if (total < 2) or (total > 3): 
                        newGrid[i, j] = 0
                else: 
                    #cell currently dead
                    if total == 3 or total == 6: 
                        old_colors = eight_neighbors[nonzero_array]
                        newGrid[i, j] = get_new_color(old_colors, np.random.uniform(low=0, high=10))                        
    return newGrid

def grid2img(grid, img_size):
    img = np.zeros((img_size,img_size), dtype=np.uint8)
    img[1:img_size-1, 1:img_size-1] = grid > 0
    
    # continuous boundaries
    img[0,:] = img[img_size-2,:]
    img[img_size-1,:] = img[1,:]
    img[:,0] = img[:,img_size-2]
    img[:,img_size-1] = img[:,1]
    return img

class GameManager(object):
    def __init__(self):
        self.N = 30
        self.img_size = self.N + 2
        self.grid = randomGrid(self.N)
        self.n_samples = 16*64
        self.skip_initial_iteration = 4
        
    def reset(self):
        self.grid = randomGrid(self.N)
        for index in range(self.skip_initial_iteration):
            self.grid = update(self.grid)
    
    @property
    def sample_size(self):
        return self.n_samples

#    def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
#        latents = [0, shape, scale, orientation, x, y]
#        index = np.dot(latents, self.latents_bases).astype(int)
#        return self.get_images([index])[0]
    
    def get_images(self, count):
        images = []
        self.reset()
        for index in range(count):
          img = grid2img(self.grid, self.img_size)
          img = img.reshape(self.img_size*self.img_size)
          images.append(img)
          self.grid = update(self.grid)
        return images
    
    def get_random_images(self, size):
        indices = [np.random.randint(self.n_samples) for i in range(size)]
        images = self.get_images(self.n_samples)
        random_images = []
        for random_index in indices:
            random_images.append(images[random_index])
        return random_images

if __name__ == '__main__':
    manager = GameManager()
    test = manager.get_random_images(10)
    print(test[0])
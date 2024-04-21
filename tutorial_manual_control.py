import gym
import numpy as np
import pygame, sys
from pygame.locals import *
import matplotlib.pyplot as plt
import Autonomous_CarRacing

#%%
def plot_trajectory():
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.title('trajectory')
    track = np.array(env.track)
    plt.plot(track[:,2], track[:,3])
    
    for i, r in enumerate(env.obj):
        fix = np.array(r.fixtures[0].shape.vertices)    
        plt.plot(fix[:,0], fix[:,1])
    
    plt.plot(x_list,y_list, 'r', linewidth=5)
    
    plt.subplot(122)
    plt.title('action')
    plt.plot(act_list,'.-')
    plt.show()

#%% Simulation example
'''
env = Autonomous_CarRacing.Autonomous_CarRacing(verbose=0) # Autonomous_CarRacing Class를 env로 설정
init_vel = 80 # 초기속도 설정

for _ in range(5):
    env.reset(options = {'init_vel': init_vel}) # 환경 초기화
    
    steer = 0.
    t=-1
    x_list = []
    y_list = []
    a_list = []
    act_list = []
    
    while True: 
        t+=1
        
        env.render() # show with pygame
        
        # logging car position and angle for plot
        x, y = env.car.hull.position
        a = env.car.hull.angle
        x_list.append(x)
        y_list.append(y)
        a_list.append(a)
        act_list.append(steer)
        
        # simulate step
        observation, _, done, _ = env.step(steer) # step 함수의 입력으로 steer(-1~1)값을 float으로 넣는다.
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    # plot logged trajectory
    plot_trajectory()
    
env.close()

'''
#%% Manual Control
env = Autonomous_CarRacing.Autonomous_CarRacing(verbose=0)
init_vel = 80

is_pressed = [False]*5 # [up, down, left, right, esc]

while True:
    env.reset(options = {'init_vel': init_vel})
    
    steer = 0.
    t=-1
    x_list = []
    y_list = []
    a_list = []
    act_list = []
    while True: 
        t+=1
        
        env.render() # show with pygame
        
        # logging car position and angle for plot
        x, y = env.car.hull.position
        a = env.car.hull.angle
        x_list.append(x)
        y_list.append(y)
        a_list.append(a)
        act_list.append(steer)
        
        events = pygame.event.get()
        
        for event in events:
            if event.type == QUIT:
                 print('quitting')
                 pygame.quit()
                 sys.exit()
            else:
                if event.type == KEYDOWN:
                    if event.key == K_UP:
                        is_pressed[0] = True
                    if event.key == K_DOWN:
                        is_pressed[1] = True
                    if event.key == K_RIGHT:
                        is_pressed[3] = True
                    elif event.key == K_LEFT:
                        is_pressed[2] = True
                    if event.key == K_ESCAPE:
                        is_pressed[4] = True
                if event.type == KEYUP:
                    if event.key == K_UP:
                        is_pressed[0] = False
                    if event.key == K_DOWN:
                        is_pressed[1] = False
                    if event.key == K_RIGHT:
                        is_pressed[3] = False
                    elif event.key == K_LEFT:
                        is_pressed[2] = False
                    if event.key == K_ESCAPE:
                        is_pressed[4] = False
        
        if is_pressed[2] & is_pressed[3]:
            pass
        elif is_pressed[2]:
            steer-=0.2
            steer = max(-1,steer)
        elif is_pressed[3]:
            steer+=0.2
            steer = min(1,steer)
        else:
            steer=0
                            
        # simulate step
        observation, _, done, _ = env.step(steer)
        
        if done or is_pressed[4]:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    # plot logged trajectory
    plot_trajectory()
    
    if is_pressed[4]:
        break
env.close()










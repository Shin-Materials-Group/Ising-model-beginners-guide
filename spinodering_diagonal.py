import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand


def mcmove(config, beta, N, Jx, Jy, Jd):
    for _ in range(N*N):   
        a = np.random.randint(0, N)
        b = np.random.randint(0, N)
        s = config[a, b]

     
        nb_x = config[a, (b+1)%N] + config[a, (b-1)%N]
        nb_y = config[(a+1)%N, b] + config[(a-1)%N, b]

      
        nb_diag = (config[(a+1)%N, (b+1)%N] +
                   config[(a+1)%N, (b-1)%N] +
                   config[(a-1)%N, (b+1)%N] +
                   config[(a-1)%N, (b-1)%N])

    
        cost = 2 * s * (Jx*nb_x + Jy*nb_y + Jd*nb_diag)

        
        if cost < 0:
            s *= -1
        elif rand() < np.exp(-cost * beta):
            s *= -1

        config[a, b] = s   
    return config



def output(f, config, i, n_, N):
    sp = f.add_subplot(2, 3, n_)
    plt.setp(sp.get_yticklabels(), visible=False)
    plt.setp(sp.get_xticklabels(), visible=False)

    color_map = np.zeros((N, N, 3))
    color_map[config == 1]  = [1.0, 0.5, 0.0]   # 주황색
    color_map[config == -1] = [0.0, 0.3, 0.8]   # 파란색

    plt.imshow(color_map, interpolation='nearest')
    plt.title(f'Monte Carlo step={i}')
    plt.axis('off')

  
    up_patch   = mpatches.Patch(color=[1.0, 0.5, 0.0], label='Spin Up')
    down_patch = mpatches.Patch(color=[0.0, 0.3, 0.8], label='Spin Down')
    plt.legend(handles=[up_patch, down_patch], loc="upper right", fontsize=8)



def simulate(N=64, temp=0.4, Jx=-1.0, Jy=1.0, Jd=1.25):
    config = 2*np.random.randint(2, size=(N,N)) - 1   

    f = plt.figure(figsize=(15, 10), dpi=80)
    snapshots = [0, 50, 500, 1000, 2000, 4000]
    plot_index = 1

    for i in range(max(snapshots)+1):
        mcmove(config, 1.0/temp, N, Jx, Jy, Jd)
        if i in snapshots:
            output(f, config, i, plot_index, N)   
            plot_index += 1

    plt.show()


simulate()

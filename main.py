
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------

Jx, Jy  = -1.0, 0.75


N       = 16
eqSteps = 1000
mcSteps = 1200
nt      = 100
T       = np.linspace(1.5, 3.3, nt)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def initialstate(N):
    state = 2*np.random.randint(2, size=(N,N))-1
    return state


def mcmove(config, beta, Jx, Jy):
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = config[a, b]
            nb_x = config[(a+1)%N, b] + config[(a-1)%N, b]
            nb_y = config[a, (b+1)%N] + config[a, (b-1)%N]
            cost = 2 * s * (Jx*nb_x + Jy*nb_y)
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            config[a, b] = s
    return config


def calcEnergy(config, Jx, Jy):
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i, j]
            energy += - (Jx * S * config[(i+1)%N, j]
                         + Jy * S * config[i, (j+1)%N])
    return energy



# ----------------------------------------------------------------------

E, C = np.zeros(nt), np.zeros(nt)
n1, n2 = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)



for tt in range(nt):
    E1 = E2 = 0
    config = initialstate(N)
    iT = 1.0/T[tt]; iT2 = iT*iT

 
    for i in range(eqSteps):
        mcmove(config, iT, Jx, Jy)


    for i in range(mcSteps):
        mcmove(config, iT, Jx, Jy)
        Ene = calcEnergy(config, Jx, Jy)

        E1 += Ene
        E2 += Ene*Ene


    E[tt] = n1 * E1
    C[tt] = (n1*E2 - n2*E1*E1) * iT2


# ----------------------------------------------------------------------


f = plt.figure(figsize=(8, 10))

sp = f.add_subplot(2, 1, 1)
plt.scatter(T, E, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Energy", fontsize=20)
plt.axis('tight')

sp = f.add_subplot(2, 1, 2)
plt.scatter(T, C, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Specific Heat", fontsize=20)
plt.axis('tight')

plt.show()
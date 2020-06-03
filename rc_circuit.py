"""
    RC Circuit simulation using ODE solver

    Differential equation to solve:
    q' = -q/RC + V/R

    Plot q, and q'
"""
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


def dq_dt(q, t, params):
    R, C, V = params
    return (-q/(R*C) + V/R)


def main():
    """
        5 ohm resistor, 5 farad capacitor and initial charge as 0, 5 volt DC cell
    """
    R = 5
    C = 5
    V = 5
    Q0 = 0
    params = (R, C, V)

    t = np.linspace(0, 100)
    q = odeint(dq_dt, Q0, t, args=(params, ))
    q = np.array(q).flatten()
    i = dq_dt(q, t, params)

    plt.plot(t, q)
    plt.plot(t, i)
    plt.show()


if __name__ == '__main__':
    main()



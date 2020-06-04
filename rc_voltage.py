import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def main():
    k=1.0
    R=2.0
    C=5
    tau=R*C

    #RC ckt eqn: RC(dVc/dt)=-Vc+V
    def dVc_dt(Vc,t):
        V=2
        return (-Vc+k*V)/tau

    t=np.linspace(0,50,100)
    Vc=odeint(dVc_dt,0,t)

    plt.figure(1)
    plt.plot(t,Vc,'r-')
    plt.xlim(right=50)
    plt.ylim(top=3)
    plt.xlabel('Time')
    plt.ylabel('Response(Vc)')

    plt.show()

main()

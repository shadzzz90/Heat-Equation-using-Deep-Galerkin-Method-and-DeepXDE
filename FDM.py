import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from celluloid import Camera


spatialDomain = [[0,10],[0,0],[0,0]]
temporalDomain = [0,10]
spatialResolution = 50
temporalResolution = 50



class heatEquation():

    def __init__(self, ICfunc, alpha, beta,  conductivity, specificHeat, density,heatfunc,periodicfunc, spatialDomain,
                 temporalDomain, spatialResolution,temporalResolution, type = None, lbctype='Dirichlet', rbctype = 'Dirichlet' ):

        self.spatialDomain = spatialDomain
        self.temporalDomain = temporalDomain
        self.spatialResolution = spatialResolution
        self.temporalResolution = temporalResolution
        self.type = type
        self.lbctype = lbctype
        self.rbctype = rbctype
        self.alpha = alpha
        self.beta = beta
        self.conductivity = conductivity
        self.heatfunc = heatfunc
        self.periodicfunc = periodicfunc
        self.ICfunc = ICfunc



        if self.type == '1D':

            self.deltaX = self.spatialDomain[0][1] / self.spatialResolution

            self.deltaT = self.temporalDomain[1] / self.temporalResolution

            # self.u_total = []

            self.A = np.zeros((self.spatialResolution + 1, self.spatialResolution + 1))

            self.Zi = self.deltaT/(density*specificHeat)

            self.sigma = (conductivity/(specificHeat*density)) * (self.deltaT/self.deltaX**2)

            self.generateMesh()

            self.generateA()

            self.b_ini = np.array([self.ICfunc(x) for x in self.meshSpatial])


            self.sourceTerm()




    def generateMesh(self):

        self.meshSpatial = np.array([i * self.deltaX for i in range(0, self.spatialResolution + 1)])
        self.meshTemporal = np.array([i * self.deltaT for i in range(0, self.temporalResolution + 1)])

        # print(self.meshSpatial)
        # print(self.meshTemporal)


    def boundaryCondLeft(self, u=0,t=0):


        if self.type == '1D':

            if self.lbctype == 'Dirichlet':

                return self.alpha

            if self.lbctype == 'Neuman':

                return (4*u[1]-u[2]-(2*self.alpha*self.deltaX)/self.conductivity)/3

            if self.lbctype == 'Periodic':

                return self.periodicfunc(t)




    def boundaryCondRight(self,u=0,t=0):

        if self.type == '1D':

            if self.rbctype == 'Dirichlet':

                # self.b_ini[-1] = uL
                return self.beta

            if self.rbctype == 'Neuman':

                return (-u[-3]+4*u[-2]+(2*self.beta*self.deltaX)/self.conductivity)/3

            if self.rbctype == 'Periodic':

                return self.periodicfunc(t)



    def sourceTerm(self, t=0):

        return np.array([self.heatfunc(x,t)*self.Zi for x in self.meshSpatial])



    def generateA(self):


        self.A[0][0] = 1

        for index in range(1, self.spatialResolution):
            self.A[index][index] = 1 + 2 * self.sigma
            self.A[index][index - 1] = -self.sigma
            self.A[index][index + 1] = - self.sigma

        self.A[-1][-1] = 1

    def solve(self):


        u_initial = self.b_ini[:]

        # self.u_total.append(u_init)

        fig, ax = plt.subplots()
        camera = Camera(fig)

        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Tempreature ($^0$C)')


        for i in range(0,len(self.meshTemporal)):

            u_current = np.linalg.solve(self.A,u_initial)
            ax.plot(self.meshSpatial,u_current,'r-')
            ax.text(0.5, 1.01, "Time = {} secs ".format(self.meshTemporal[i]), transform=ax.transAxes)
            camera.snap()

            u_initial = u_current + self.sourceTerm(t=0)

            if self.lbctype == 'Dirichlet' and self.rbctype == 'Dirichlet':
                u_initial[0] = self.boundaryCondLeft()
                u_initial[-1] = self.boundaryCondRight()

            if self.lbctype == 'Dirichlet' and self.rbctype == 'Neuman':
                u_initial[0] = self.boundaryCondLeft()
                u_initial[-1] = self.boundaryCondRight(u=u_initial)

            if self.lbctype == 'Neuman' and self.rbctype == 'Dirichlet':

                u_initial[0] = self.boundaryCondLeft(u=u_initial)
                u_initial[-1] = self.boundaryCondRight()

            if self.lbctype == 'Neuman' and self.rbctype == 'Neuman':

                u_initial[0] = self.boundaryCondLeft(u=u_initial)
                u_initial[-1] = self.boundaryCondRight(u=u_initial)

            if self.lbctype == 'Neuman' and self.rbctype == 'Periodic':

                u_initial[0] = self.boundaryCondLeft(u=u_initial)
                u_initial[-1] = self.boundaryCondRight(t = self.meshTemporal[i])

            if self.lbctype == 'Periodic' and self.rbctype == 'Neuman':

                u_initial[0] = self.boundaryCondLeft(t=self.meshTemporal[i])
                u_initial[-1] = self.boundaryCondRight(u=u_initial)

            if self.lbctype == 'Dirichlet' and self.rbctype == 'Periodic':
                u_initial[0] = self.boundaryCondLeft()
                u_initial[-1] = self.boundaryCondRight(t=self.meshTemporal[i])

            if self.lbctype == 'Periodic' and self.rbctype == 'Dirichlet':
                u_initial[0] = self.boundaryCondLeft(t=self.meshTemporal[i])
                u_initial[-1] = self.boundaryCondRight()


        anim = camera.animate()

        anim.save('solution.gif')


def heatfunc(x, t):

    # if t <= 15:
    #
    #     I0 = 10e10
    #     delta = 6.17e7
    #     rf = 0.62
    #
    #     q = I0*(1-rf)*delta*math.exp(-delta*x)
    #
    # else:

    q = 0

    return q


def periodicfunc(t):


    return 20 + 15 * math.sin(2 * math.pi *t / temporalDomain[1])

def ICfunc(x):

    u_init = 1-x

    return u_init



heat1D = heatEquation(ICfunc = ICfunc,alpha=100,beta= 200,conductivity=150.0,specificHeat=510.0,density=7930.0,
                      spatialDomain=spatialDomain, temporalDomain=temporalDomain, spatialResolution=spatialResolution,
                      temporalResolution=temporalResolution,type = '1D',lbctype='Dirichlet', rbctype='Dirichlet',
                      heatfunc = heatfunc, periodicfunc=periodicfunc)

heat1D.solve()
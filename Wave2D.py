import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, L=1, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.L = L
        self.h = L/N
        x = y = np.linspace(0,L,N+1)
        self.xij, self.yij = np.meshgrid(x,y,indexing = 'ij')
        return self.xij, self.yij
        #raise NotImplementedError

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return (1/self.h**2)*D
        #raise NotImplementedError

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*sp.pi
        ky = self.my*sp.pi
        return self.c*sp.sqrt(kx**2+ky**2)
        #raise NotImplementedError

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        xij, yij = self.create_mesh(N)
        dt = self.dt
        
        ue_numeric = sp.lambdify((x,y,t),self.ue(self.mx,self.my),'numpy')(xij,yij,0)
        Unm1[:] = ue_numeric
        Un[:] = Unm1[:] + 0.5*(self.c*dt)**2*(self.D2(N) @ Unm1 + Unm1 @ self.D2(N).T)
        
        self.Unm1 = Unm1
        self.Un = Un
        self.Unp1 = Unp1
        #raise NotImplementedError

     @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        xij, yij = self.create_mesh(self.N)
        ue_t0 = sp.lambdify((x,y,t),self.ue(self.mx,self.my),'numpy')(xij,yij,t0)
        return np.sqrt(self.h**2*np.sum((ue_t0-u)**2))
        #raise NotImplementedError

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0
        #raise NotImplementedError

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        xij, yij = self.create_mesh(N)
        dt = self.dt
        D = self.D2(N)
        self.initialize(N, mx, my)
        Unm1 = self.Unm1
        Un = self.Un
        Unp1 = self.Unp1

    
        plotdata = {0: self.Unm1.copy()}
        if store_data == 1:
            plotdata[1] = self.Un.copy()
        for n in range(1, Nt):
            Unp1[:] = 2*Un[:] - Unm1[:] + (c*dt)**2*(D @ Un + Un @ D.T)
            # Set boundary conditions
            self.apply_bcs()
            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1
            if n % store_data == 0:
                plotdata[n] = Unm1.copy() # Unm1 is now swapped to Un
        if store_data > 0:
            return xij, yij, plotdata
        if store_data == -1:
            #print(self.h, [self.l2_error(Unm1,(Nt-1)*dt)])
            return (self.h, [self.l2_error(Unm1,(Nt-1)*dt)])
        #raise NotImplementedError

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = -2, 2, 0, 0
        D[-1, -4:] = 0, 0, 2, -2
        return (1/self.h**2)*D
        #raise NotImplementedError

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
        #raise NotImplementedError

    def apply_bcs(self):
        return
        #raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    Dsol = Wave2D()
    Derror = Dsol.convergence_rates(m=8,cfl=1/np.sqrt(2),mx = 3,my = 3)[1]
    #Had to use more discretization levels (m=8) on this one.
    Nsol = Wave2D_Neumann()
    Nerror = Nsol.convergence_rates(cfl=1/np.sqrt(2),mx = 3,my = 3)[1]
    assert abs(Derror[-1] < 1e-12)
    assert abs(Nerror[-1] < 1e-12)


def wave2d_animation():
    import matplotlib.animation as animation
    sol = Wave2D_Neumann()
    xij, yij, data = sol(N = 40, Nt = 500, cfl = 0.5, c= 1, store_data=5)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in data.items():
        frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2);
        #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
        #                        vmax=data[0].max(), cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                repeat_delay=1000)
    ani.save('neumannwave.gif', writer='pillow', fps=5) # This animated png opens in a browser


if __name__ == '__main__':
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()

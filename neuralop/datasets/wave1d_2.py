from pathlib import Path
import torch

from ..utils import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding

import torch
import torch.nn.functional as F
from math import pi, gamma, sqrt
import numpy as np
from functorch import vmap, grad


from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import loadmat


torch.manual_seed(0)


class GRF_Mattern(object):
    def __init__(self, dim, size, length=1.0, nu=None, l=0.1, sigma=1.0, boundary="periodic", constant_eig=None, device=None):

        self.dim = dim
        self.device = device
        self.bc = boundary

        a = sqrt(2/length)
        if self.bc == "dirichlet":
            constant_eig = None
        
        
        if nu is not None:
            kappa = sqrt(2*nu)/l
            alpha = nu + 0.5*dim
            self.eta2 = size**dim * sigma*(4.0*pi)**(0.5*dim)*gamma(alpha)/(kappa**dim * gamma(nu))
        else:
            self.eta2 = size**dim * sigma*(sqrt(2.0*pi)*l)**dim
        # if sigma is None:
        #     sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2
        if self.bc == "periodic":
            const = (4.0*(pi**2))/(length**2)
        else:
            const = (pi**2)/(length**2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            k2 = k**2
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            k2 = k_x**2 + k_y**2 
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0,0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            k2 = k_x**2 + k_y**2 + k_z**2
            if nu is not None:
                # self.sqrt_eig = (size**dim)*sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
                eigs = 1.0 + (const/(kappa*length)**2*k2)
                self.sqrt_eig = self.eta2/(length**dim) * eigs**(-alpha/2.0)
            else:
                self.sqrt_eig = self.eta2/(length**dim)*torch.exp(-(l)**2*const*k2/4.0)

            if constant_eig is not None:
                self.sqrt_eig[0,0,0] = constant_eig # (size**dim)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        if self.bc == 'dirichlet':
            coeff.real[:] = 0
        if self.bc == 'neumann':
            coeff.imag[:] = 0
        coeff = self.sqrt_eig*coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u


class WaveEq1D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                #  ymin=0,
                #  ymax=1,
                #  dx=0.01,
                #  dy=0.01,
                 Nx=100,
                #  Ny=100,
                 c=1.0,
                 dt=1e-3,
                 tend=1.0,
                 device=None,
                 dtype=torch.float64,
                #  phi0='Data/data6.h5'
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        x = torch.linspace(xmin, xmax, Nx+1, device=device, dtype=dtype)
        self.x = x
        # self.y = y
        self.dx = x[1] - x[0]
        # self.dy = y[1] - y[0]
        # self.X, self.Y = torch.meshgrid(x,y,indexing='ij')
        self.c = c
        self.phi = torch.zeros_like(self.x[:Nx], device=device)
        self.psi = torch.zeros_like(self.phi, device=device)
        self.phi0 = torch.zeros_like(self.phi, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.Phi = []
        self.T = []
        self.device = device
        
        
    

    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx


    
    def wave_calc_RHS(self, phi, psi):
        phi_xx = self.Dxx(phi)
        
        psi_RHS = self.c**2 * phi_xx # it is usually c^2, but c is consistent with simflowny code
        phi_RHS = psi
        return phi_RHS, psi_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        


    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def wave_rk4(self, phi, psi, t=0):
        phi_RHS1, psi_RHS1 = self.wave_calc_RHS(phi, psi)
        t1 = t + 0.5*self.dt
        # display(phi)
        # display(phi_RHS1)
        phi1 = self.update_field(phi, phi_RHS1, step_frac=0.5)
        psi1 = self.update_field(psi, psi_RHS1, step_frac=0.5)
        
        phi_RHS2, psi_RHS2 = self.wave_calc_RHS(phi1, psi1)
        t2 = t + 0.5*self.dt
        phi2 = self.update_field(phi, phi_RHS2, step_frac=0.5)
        psi2 = self.update_field(psi, psi_RHS2, step_frac=0.5)
        
        phi_RHS3, psi_RHS3 = self.wave_calc_RHS(phi2, psi2)
        t3 = t + self.dt
        phi3 = self.update_field(phi, phi_RHS3, step_frac=1.0)
        psi3 = self.update_field(psi, psi_RHS3, step_frac=1.0)
        
        phi_RHS4, psi_RHS4 = self.wave_calc_RHS(phi3, psi3)
        
        t_new = t + self.dt
        psi_new = self.rk4_merge_RHS(psi, psi_RHS1, psi_RHS2, psi_RHS3, psi_RHS4)
        phi_new = self.rk4_merge_RHS(phi, phi_RHS1, phi_RHS2, phi_RHS3, phi_RHS4)
        
        return phi_new, psi_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        plt.plot(self.x, self.phi)
        # c = plt.pcolormesh(self.X, self.Y, self.phi, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        # fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.axis('equal')
        # plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()

        
    def wave_driver(self, phi0, save_interval=10, plot_interval=0):
        # plot results
        # t,it = get_time(time)
#         display(phi0[:self.Nx,:self.Ny].shape)
        self.phi0 = phi0[:self.Nx]
        self.phi = self.phi0
        self.t = 0
        self.it = 0
        self.T = []
        self.Phi = []
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{phi}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.Phi.append(self.phi)
            # self.Psi.append(self.psi)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
#             print(f"t:\t{self.t}")
            self.phi, self.psi, self.t = self.wave_rk4(self.phi, self.psi, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{phi}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.Phi.append(self.phi)
                # self.Psi.append(self.psi)
                self.T.append(self.t)

        return torch.stack(self.Phi)



def load_wave1d(
    n_train,
    n_tests,
    batch_size,
    test_batch_size,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
):
    """Load the Navier-Stokes dataset"""
    Nsamples = 250
    N = 128
    Nt0 = 100
    c = 1.0
    sub_t = 1
    Nt = Nt0 // sub_t + 1
    dim = 1
    l = 0.1
    L = 1.0
    sigma = 0.2 #2.0
    Nu = None # 2.0
    dt = 1.0e-4
    tend = 1.0
    save_int = int(tend/dt/Nt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    grf = GRF_Mattern(dim, N, length=L, nu=Nu, l=l, sigma=sigma, boundary="periodic", device=device)
    U0 = grf.sample(Nsamples)
    wave_eq = WaveEq1D(Nx=N, c=c, dt=dt, device=device)
    U = vmap(wave_eq.wave_driver, in_dims=(0, None))(U0, save_int)
    a = U0.cpu().float()
    u = U.cpu().float()
    a = a.unsqueeze(1).unsqueeze(1)
    u = u.unsqueeze(1)

    data = {'x' : a,
            'y' : u}




    x_train = (
        data['x'][0:n_train, :, :, :].repeat(1,1,102,1).clone()
    )
    y_train = data['y'][0:n_train, :, :].clone()


    x_test = (
        data['x'][n_train:n_train+n_tests, :, :, :].repeat(1,1,102,1).clone()
    )
    y_test = data['y'][n_train:n_train+n_tests, :, :].clone()

    del data, a, u, U, U0, grf, wave_eq




    train_db = TensorDataset(
        x_train,
        y_train,
        transform_x=PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
        transform_x=PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders= {128: test_loader}


    return train_loader, test_loaders



import numpy as np
from math import pi
import torch
import scipy.integrate as integrate

"""
simply supported - simply supported : ss-ss
fixed - fixed : fx-fx
free - free : fr-fr
fixed - simply : fx-s
fixed - free : fx-fr
"""

class cont_beam:

    # def __init__(self, E, I, rho, area, length):
    def __init__(self, def_type, **kwargs):
        super().__init__()

        self.mat_var_type = def_type
        self.L = self.import_variable(kwargs["l"])

        match def_type:
            case "sep_vars":
                self.E = self.import_variable(kwargs["E"])
                self.rho = self.import_variable(kwargs["rho"])
                self.I = self.import_variable(kwargs["I"])
                if type(kwargs["area"]) == list or type(kwargs["area"]) == tuple:
                    self.b = self.import_variable(kwargs["area"][0])
                    self.h = self.import_variable(kwargs["area"][1])
                    self. A = torch.product(self.import_variable(kwargs["area"]))
                    if torch.abs(self.I - (1/12)*self.b*self.h**3)/self.I < 0.01:
                        raise ValueError("Moment of inertia does not match values of b and h...")
                else:
                    self.A = self.import_variable(kwargs["area"])
            case "cmb_vars":
                self.EI = self.import_variable(kwargs["EI"])
                self.pA = self.import_variable(kwargs["pA"])
    
    def import_variable(self, var):
        if torch.is_tensor(var):
            return var
        else:
            return torch.tensor(var)
            
    def gen_modes(self, bc_type, n_modes, nx):

        self.bc_type = bc_type
        self.nx = nx
        x = torch.linspace(0, self.L, nx)
        self.xx = x
        self.n_modes = n_modes
        nn = torch.arange(1, n_modes+1, 1)
        match self.mat_var_type:
            case "sep_vars":
                wn_mult = (self.E * self.I / (self.rho * self.A * self.L**4))**(0.5)
            case "cmb_vars":
                wn_mult = (self.EI / (self.pA * self.L**4))**(0.5)


        match bc_type:
            case "ss-ss":
                self.bc_type_long = "simply supported - simply supported"
                beta_l = nn*pi
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] = -torch.sin(beta_l[n]*x/self.L)
            case "fx-fx":
                self.bc_type_long = "fixed - fixed"
                beta_l = (2*nn + 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (torch.cos(beta_n[n]*x) - torch.cosh(beta_n[n]*x)) - \
                                    (torch.cos(beta_l[n]) - torch.cosh(beta_l[n]))/(torch.sin(beta_l[n]) - torch.sinh(beta_l[n])) * \
                                    (torch.sin(beta_n[n]*x) - torch.sinh(beta_n[n]*x))
            case "fr-fr":
                self.bc_type_long = "free - free"
                beta_l = (2*nn + 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (torch.cos(beta_n[n]*x) + torch.cosh(beta_n[n]*x)) - \
                                    (torch.cos(beta_l[n]) - torch.cosh(beta_l[n]))/(torch.sin(beta_l[n]) - torch.sinh(beta_l[n])) * \
                                    (torch.sin(beta_n[n]*x) + torch.sinh(beta_n[n]*x))
            case "fx-ss":
                self.bc_type_long = "fixed - simply supported"
                beta_l = (4*nn + 1) * pi / 4
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
            case "fx-fr":
                self.bc_type_long = "fixed - free"
                beta_l = (2*nn - 1) * pi / 2
                beta_n = beta_l / self.L
                self.wn = (beta_l**2) * wn_mult
                self.phi_n = torch.zeros((self.nx, n_modes))
                for n in range(n_modes):
                    self.phi_n[:,n] =  (torch.cos(beta_n[n]*x) - torch.cosh(beta_n[n]*x)) - \
                                    (torch.cos(beta_l[n]) + torch.cosh(beta_l[n]))/(torch.sin(beta_l[n]) + torch.sinh(beta_l[n])) * \
                                    (torch.sin(beta_n[n]*x) - torch.sinh(beta_n[n]*x))
                    

    def free_vibration(self, time, w0, wd0):
        nt = time.shape[0]
        x = self.xx
        nx = x.shape[0]

        ww = torch.zeros((nx, nt, self.n_modes))
        wwd = torch.zeros((nx, nt, self.n_modes))
        wwdd = torch.zeros((nx, nt, self.n_modes))
        for n in range(self.n_modes):
            eta_integrand = self.pA * self.phi_n[:,n] * w0
            eta = integrate.simpson(eta_integrand, x)
            eta_dot_integrand = self.pA * self.phi_n[:,n] * wd0
            eta_dot = integrate.simpson(eta_dot_integrand, x)
            for t in range(nt):
                ww[:,t,n] = self.phi_n[:,n] * (eta*torch.cos(self.wn[n]*time[t]) + (eta_dot/self.wn[n])*torch.sin(self.wn[n]*time[t]))
                wwd[:,t,n] = self.phi_n[:,n] * (-self.wn[n]*eta*torch.sin(self.wn[n]*time[t]) + eta_dot*torch.cos(self.wn[n]*time[t]))
                wwdd[:,t,n] = self.phi_n[:,n] * (-(self.wn[n]**2)*eta*torch.cos(self.wn[n]*time[t]) - self.wn[n]*eta_dot*torch.sin(self.wn[n]*time[t]))
        self.wxt = torch.sum(ww, dim=2)
        self.wxtd = torch.sum(wwd, dim=2)
        self.wxtdd = torch.sum(wwdd, dim=2)

        return self.wxt, self.wxtd, self.wxtdd, ww
    
    def forced_vibration(self, time, forcing):
        nt = time.shape[0]
        x = self.xx
        nx = self.xx.shape[0]
        ww = torch.zeros((nx, nt, self.n_modes))
        match forcing["type"]:
            case "step_load":
                F0 = forcing["force_mag"]
                a = forcing["load_coord"]
                for n in range(1, self.n_modes+1):
                    for ti, t in enumerate(time):
                        ww[:, ti, n-1] = ((2*F0*self.L**3)/(pi**4 * self.EI)) * (1/(n**4)) * torch.sin((n*pi*x)/self.L) * torch.sin((n*pi*a)/self.L) * (1 - torch.cos(self.wn[n-1] * t))
            case "harmonic":
                F0 = forcing["force_mag"]
                a = forcing["load_coord"]
                OM = forcing["frequency"] * 2 * pi
                for n in range(1, self.n_modes+1, 2):
                    wi = (n**2 * pi**2 / self.L**2) * (self.EI / self.pA)
                    if (n+1)%4 != 0:
                        for ti, t in enumerate(time):
                            ww[:, ti, n-1] = (2*F0/(self.pA*self.L)) * (torch.sin(n*pi*x/self.L)/(wi**2 - OM**2)) * (torch.sin(OM*t) - (OM/wi)*torch.sin(wi*t))
                    else:
                        for ti, t in enumerate(time):
                            ww[:, ti, n-1] = -(2*F0/(self.pA*self.L)) * (torch.sin(n*pi*x/self.L)/(wi**2 - OM**2)) * (torch.sin(OM*t) - (OM/wi)*torch.sin(wi*t))

        
        return torch.sum(ww, dim=2), ww

    
    def init_cond_load(self, init_load):
        match init_load["type"]:
            case "point_load":
                a = init_load["load_coord"]
                b = self.L - a
                F = init_load["f0"]
                x = self.xx
                fl_id = torch.argmin(torch.abs(x - a))
                w0 = torch.zeros(x.shape[0])
                w0[:fl_id] = -((F*b*x[:fl_id])/(6*self.L*self.EI)) * (self.L**2 - b**2 - x[:fl_id]**2)
                w0[fl_id:] = -((F*b)/(6*self.L*self.EI)) * (self.L/b*(x[fl_id:]-a)**3 + (self.L**2 - b**2)*x[fl_id:] - x[fl_id:]**3)
        self.w0 = w0
        return w0



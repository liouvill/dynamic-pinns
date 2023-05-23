import torch
import torch.nn as nn
import numpy as np

def max_mag_data(data,axis=None):
    if torch.is_tensor(data):
        if axis==None:
            data_max = torch.max(torch.max(torch.abs(data)))
        else:
            data_max = torch.max(torch.abs(data),dim=axis)[0]
    else:
        data_max = np.max(np.abs(data),axis=axis)
    return data_max

def normalise(data,norm_type="var",norm_dir="all"):
    if norm_type=="var":
        if len(data.shape)>1 and norm_dir=="axis":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
        else:
            mean = data.mean()
            std = data.std()
        data_norm = (data-mean)/std
        return data_norm, (mean, std)
    elif norm_type=="range":
        if len(data.shape)>1 and norm_dir=="axis":
            dmax = max_mag_data(data,axis=0)
        elif len(data.shape)>1 and norm_dir=="all":
            dmax = max_mag_data(data,None)
        else:
            dmax = max_mag_data(data)
        data_norm = data/dmax
        return data_norm, dmax

class bbnn(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build_net()
    
    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
        
    def forward(self, x):
        x = self.net(x)
        return x
    
    def predict(self, xp, tp):
        XX = torch.cat((xp.view(-1,1), tp.view(-1,1)), dim=1)
        wp = self.forward(XX)
        return wp

    def loss_func(self, x_obs, t_obs, w_obs):
        XX = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        wp_obs = self.forward(XX)
        loss = torch.mean((wp_obs - w_obs)**2)
        return loss
    
    def train(self, x_obs, t_obs, w_obs, epochs, epsilon, opti_type, opti_params):
        match opti_type:
            case "Adam":
                learning_rate = opti_params["learning_rate"]
                betas = opti_params["betas"]
                optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas)
        loss_hist = []
        for i in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_func(x_obs, t_obs, w_obs)
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()

            if (i>1000) and (torch.mean(torch.tensor(loss_hist[:-100]))<epsilon):
                break

        return loss_hist

class beam_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.activation = nn.Tanh

        self.configure(config)

        self.build_net()

    def configure(self, **config):

        self.config = config
        self.param_type = config["phys_params"]["par_type"]

        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.set_phys_params()
        self.set_norm_params()

        self.device = torch.device("cpu")

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net

    def set_phys_params(self, params, par_type):
        self.param_type = par_type
        match par_type:
            case "constant":
                self.pA = params['pA']
                self.EI = params['EI']
                self.phys_params = torch.tensor([self.pA, self.EI])
            case "variable":
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([params['pA'], params['EI']])))
    
    def set_norm_params(self, alphas, pde_norm_Lambda):
        self.alpha_t = alphas['t']
        self.alpha_x = alphas['x']
        self.alpha_w = alphas['w']
        
        self.alpha_pA = alphas['pA']
        self.alpha_EI = alphas['EI']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * pde_norm_Lambda,
                    "dx4" : self.EI * alpha_dx4 * pde_norm_Lambda
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * pde_norm_Lambda,
                    "dx4" : alpha_dx4 * pde_norm_Lambda
                }

    def set_conditions(self, init_conds, bound_conds):
        self.init_conds = init_conds
        self.bound_conds = bound_conds

    def forward(self, x):
        x = self.net(x)
        return x
    
    def calc_residuals(self, x_pde_hat, t_pde_hat, x_obs, t_obs, w_obs):
        XX_obs = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        XX_pde = torch.cat((x_pde_hat.view(-1,1), t_pde_hat.view(-1,1)), dim=1)

        match self.param_type:
            case "constant":
                self.m_hat = self.pde_alphas["dt2"]
                self.k_hat = self.pde_alphas["dx4"]
            case "variable":
                self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                self.k_hat = self.pde_alphas["dx4"] * self.phys_params[2]

        # observation residual
        wh_obs = self.forward(XX_obs)
        R_obs = w_obs - wh_obs

        # pde residual
        wh_pde_hat = self.forward(XX_pde)
        dx = torch.autograd.grad(wh_pde_hat, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
        dx2 = torch.autograd.grad(dx, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
        dx3 = torch.autograd.grad(dx2, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
        dx4 = torch.autograd.grad(dx3, x_pde_hat, torch.ones_like(wh_pde_hat), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat

        dt = torch.autograd.grad(wh_pde_hat, t_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
        dt2 = torch.autograd.grad(dt, t_pde_hat, torch.ones_like(wh_pde_hat), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

        R_pde = self.m_hat * dt2 + self.c_hat * dt + self.k_hat * dx4

        # boundary condition residual
        R_bc_disp = wh_pde_hat[self.bound_conds['ids']]*self.alpha_w
        R_bc_stre = dx2[self.bound_conds['ids']]*self.alpha_w/(self.alpha_x**2)

        # initial condition residual
        # R_ic_disp = wh_pde_hat[self.init_conds['ids']]*self.alpha_w - self.init_conds['disp_vals']
        # R_ic_vel = dt[self.init_conds['ids']]*self.alpha_w/self.alpha_t

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde,
            "R_bc_disp" : R_bc_disp,
            "R_bc_stre" : R_bc_stre,
            # "R_ic_disp" : R_ic_disp,
            # "R_ic_vel" : R_ic_vel
        }
    
    def loss_func(self, x_pde, t_pde, x_obs, t_obs, w_obs, lambds):
        residuals = self.calc_residuals(x_pde, t_pde, x_obs, t_obs, w_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc_disp = residuals["R_bc_disp"]
        R_bc_stre = residuals["R_bc_stre"]
        # R_ic_disp = residuals["R_ic_disp"]
        # R_ic_vel = residuals["R_ic_vel"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc_disp**2) + torch.mean(R_bc_stre**2))
        # L_ic = lambds['ic'] * (torch.mean(R_ic_disp**2) + torch.mean(R_ic_vel**2))
        loss = L_obs + L_pde + L_bc

        return loss, [L_obs, L_pde, L_bc]
    
    def predict(self, xp, tp):
        XX = torch.cat((xp.view(-1,1), tp.view(-1,1)), dim=1)
        wp = self.forward(XX)
        return wp
    
class beam_pinn_Yuan(nn.Module):

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net

    def set_phys_params(self, params, par_type):
        self.param_type = par_type
        match par_type:
            case "constant":
                self.mu = params['mu']
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([params['pA'], params['EI']])))
                self.register_parameter("mu", nn.Parameter(params['mu']))
    
    def set_norm_params(self, alphas, pde_norm_Lambda):
        self.alpha_t = alphas['t']
        self.alpha_x = alphas['x']
        self.alpha_w = alphas['w']
        
        self.alpha_mu = alphas['mu']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * pde_norm_Lambda,
                    "dx4" : self.mu * alpha_dx4 * pde_norm_Lambda
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * pde_norm_Lambda,
                    "dx4" : alpha_dx4 * pde_norm_Lambda
                }

    def set_conditions(self, init_conds, bound_conds):
        self.init_conds = init_conds
        self.bound_conds = bound_conds

    def forward(self, x):
        x = self.net(x)
        return x
    
    def calc_residuals(self, x_pde_hat, t_pde_hat, x_obs, t_obs, w_obs, lambds):
        XX_obs = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        XX_pde = torch.cat((x_pde_hat.view(-1,1), t_pde_hat.view(-1,1)), dim=1)

        match self.param_type:
            case "constant":
                self.m_hat = self.pde_alphas["dt2"]
                self.k_hat = self.pde_alphas["dx4"]
            case "variable":
                self.m_hat = self.pde_alphas["dt2"]
                self.k_hat = self.pde_alphas["dx4"] * self.mu

        # observation residual
        wh_obs = self.forward(XX_obs)
        R_obs = w_obs - wh_obs

        # pde residual
        wh_pde_hat = self.forward(XX_pde)
        dx = torch.autograd.grad(wh_pde_hat, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
        dx2 = torch.autograd.grad(dx, x_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_x-hat N_w-hat

        dt = torch.autograd.grad(wh_pde_hat, t_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat

        if lambds['pde'] > 0.0:
            dx3 = torch.autograd.grad(dx2, x_pde_hat, torch.ones_like(dx2), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
            dx4 = torch.autograd.grad(dx3, x_pde_hat, torch.ones_like(dx3), retain_graph=True)[0]  # ∂^4_x-hat N_w-hat
            dt2 = torch.autograd.grad(dt, t_pde_hat, torch.ones_like(dt), retain_graph=True)[0]  # ∂^2_t-hat N_w-hat

            R_pde = self.m_hat * dt2 + self.k_hat * dx4
        else:
            R_pde = torch.zeros(XX_pde.shape[0])

        # boundary condition residual
        R_bc_disp = wh_pde_hat[self.bound_conds['ids']]*self.alpha_w
        R_bc_stre = dx2[self.bound_conds['ids']]*self.alpha_w/(self.alpha_x**2)

        # initial condition residual
        # R_ic_disp = wh_pde_hat[self.init_conds['ids']]*self.alpha_w - self.init_conds['disp_vals']
        # R_ic_vel = dt[self.init_conds['ids']]*self.alpha_w/self.alpha_t

        # flatness residual
        # R_std_o = w_obs - torch.mean(w_obs)
        R_std_p = wh_pde_hat - torch.mean(wh_pde_hat)

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde,
            "R_bc_disp" : R_bc_disp,
            "R_bc_stre" : R_bc_stre,
            # "R_ic_disp" : R_ic_disp,
            # "R_ic_vel" : R_ic_vel,
            "R_fl" : R_std_p,
        }
    
    def loss_func(self, x_pde, t_pde, x_obs, t_obs, w_obs, lambds):
        residuals = self.calc_residuals(x_pde, t_pde, x_obs, t_obs, w_obs, lambds)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc_disp = residuals["R_bc_disp"]
        R_bc_stre = residuals["R_bc_stre"]
        # R_ic_disp = residuals["R_ic_disp"]
        # R_ic_vel = residuals["R_ic_vel"]
        R_fl = residuals["R_fl"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc1 = lambds['bc1'] * torch.mean(R_bc_disp**2)
        L_bc2 = lambds['bc2'] * torch.mean(R_bc_stre**2)
        # L_ic = lambds['ic'] * (torch.mean(R_ic_disp**2) + torch.mean(R_ic_vel**2))
        L_std = torch.mean(R_fl**2)
        # L_fl = lambds['fl'] * (1/(L_std+1)**0.5)
        # L_sk = lambds['sk'] * torch.mean(R_fl**3)/L_std

        loss = L_obs + L_pde + L_bc1 + L_bc2 #+ L_fl + L_sk

        return loss, [L_obs, L_pde, L_bc1, L_bc2]#, L_fl, L_sk]
    
    def predict(self, xp, tp):
        XX = torch.cat((xp.view(-1,1), tp.view(-1,1)), dim=1)
        wp = self.forward(XX)
        return wp
        

    


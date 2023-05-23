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


class beam_bb_par(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, N_MODES):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.n_modes = N_MODES
        self.activation = nn.Tanh

        self.build()
    
    def build(self):
        self.nets = nn.ModuleList()
        for n in range(self.n_modes):
            self.nets.append(nn.Sequential(
                nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
                nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
                nn.Linear(self.n_hidden, self.n_output)
            ))
        # for n in range(self.n_modes):
        #     self.add_module(("net_mode_"+str(n+1)), bbnn)
        
    def forward(self, x):
        xn = torch.zeros((x.shape[0], self.n_modes))
        for n, net in enumerate(self.nets):
            xn[:,n] = net.forward(x).view(-1)
        return xn
    
    def predict(self, xp, tp, phi_p):
        XX = torch.cat((xp.view(-1,1), tp.view(-1,1)), dim=1)
        qn_pred = self.forward(XX)
        wn_pred = qn_pred * phi_p
        return wn_pred

    def loss_func(self, x_obs, t_obs, w_obs, phi_obs):
        XX = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        qnp_obs = self.forward(XX)
        wnp_obs = qnp_obs * phi_obs
        wp_obs = torch.sum(wnp_obs, dim=1)
        loss = torch.mean((wp_obs - w_obs)**2)
        return loss
    

class beam_bb_seq(nn.Module):
    """ Neural network for beams, with no physics information in the loss """
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, N_MODES):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.n_modes = N_MODES
        self.activation = nn.Tanh

        self.build()

    def build(self):
        self.seq_NNs = [None] * self.n_modes
        for n in range(self.n_modes):
            self.seq_NNs[n] = bbnn(self.n_input, self.n_output, self.n_hidden, self.n_layers)
    
    def predict(self, xp, tp, phi_p):
        XX = torch.cat((xp.view(-1,1), tp.view(-1,1)), dim=1)
        w_pred = torch.zeros((xp.shape[0], self.n_modes))
        for n in range(self.n_modes):
            w_pred[:,n] = self.seq_NNs[n].forward(XX).view(-1) * phi_p[:,n]
        return w_pred
    
    def ind_loss_func(self, x_obs, t_obs, w_obs, phi_obs, n):
        XX = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        qp_obs = self.seq_NNs[n].forward(XX).view(-1)
        wp_obs = qp_obs * phi_obs[:,n]
        loss = torch.mean((w_obs - wp_obs.view(-1,1)))
        return loss

    def full_loss_func(self, x_obs, t_obs, w_obs, phi_obs):
        XX = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        wp_obs = torch.zeros((x_obs.shape[0], self.n_modes))
        for n in range(self.n_modes):
            wp_obs[:,n] = self.seq_NNs[n].forward(XX).view(-1) * phi_obs[:,n]
        loss = torch.mean((torch.sum(wp_obs, dim=1) - w_obs)**2)
        return loss



# class beam_bb_par(nn.Module):
#     """ Neural network for beams, with no physics information in the loss """
#     def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, N_MODES):
#         super().__init__()
#         self.n_input = N_INPUT
#         self.n_output = N_OUTPUT
#         self.n_hidden = N_HIDDEN
#         self.n_layers = N_LAYERS
#         self.n_modes = N_MODES
#         self.activation = nn.Tanh

#         self.build()

#     def build(self):
#         self.par_NNs = [None] * self.n_modes
#         for n in range(self.n_modes):
#             self.par_NNs[n] = bbnn(self.n_input, self.n_output, self.n_hidden, self.n_layers)
    
#     def predict(self, xp, tp, phi_p):
#         XX = torch.cat((xp.view(-1,1), tp.view(-1,1)), dim=1)
#         w_pred = torch.zeros((xp.shape[0], self.n_modes))
#         for n in range(self.n_modes):
#             w_pred[:,n] = self.par_NNs[n].forward(XX).view(-1) * phi_p[:,n]
#         return w_pred

#     def loss_func(self, x_obs, t_obs, w_obs, phi_obs):
#         XX = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
#         wp_obs = torch.zeros((x_obs.shape[0], self.n_modes))
#         for n in range(self.n_modes):
#             wp_obs[:,n] = self.par_NNs[n].forward(XX).view(-1) * phi_obs[:,n]
#         loss = torch.mean((torch.sum(wp_obs, dim=1) - w_obs)**2)
#         return loss


class beam_pinn(nn.Module):

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
                self.pA = params['pA']
                self.c = params['c']
                self.EI = params['EI']
                # self.phys_params = torch.tensor([self.pA, self.EI])
                self.phys_params = torch.tensor([self.pA, self.c, self.EI])
            case "variable":
                # self.register_parameter("phys_params", nn.Parameter(torch.tensor([params['pA'], params['EI']])))
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([params['pA'], params['c'], params['EI']])))
    
    def set_norm_params(self, alphas, pde_norm_Lambda):
        self.alpha_t = alphas['t']
        self.alpha_x = alphas['x']
        self.alpha_w = alphas['w']
        
        self.alpha_pA = alphas['pA']
        self.alpha_EI = alphas['EI']
        self.alpha_c = alphas['c']

        alpha_dt2 = 1.0 / (self.alpha_t**2)
        alpha_dt1 = 1.0 / self.alpha_t
        alpha_dx4 = 1.0 / (self.alpha_x**4)

        match self.param_type:
            case "constant":
                self.pde_alphas = {
                    "dt2" : self.pA * alpha_dt2 * pde_norm_Lambda,
                    "dt1" : self.pA * self.c * alpha_dt1 * pde_norm_Lambda,
                    "dx4" : self.EI * alpha_dx4 * pde_norm_Lambda
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * pde_norm_Lambda,
                    "dt1" : alpha_dt1 * pde_norm_Lambda,
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
                self.c_hat = self.pde_alphas["dt1"]
                self.k_hat = self.pde_alphas["dx4"]
            case "variable":
                self.m_hat = self.pde_alphas["dt2"] * self.phys_params[0]
                self.c_hat = self.pde_alphas["dt1"] * self.phys_params[0] * self.phys_params[1]
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
        R_bc_disp = wh_pde_hat[self.bound_conds['ids']]
        R_bc_stre = dx2[self.bound_conds['ids']]

        # initial condition residual
        R_ic_disp = wh_pde_hat[self.init_conds['ids']]*self.alpha_w - self.init_conds['disp_vals']
        R_ic_vel = dt[self.init_conds['ids']]*self.alpha_w/self.alpha_t

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde,
            "R_bc_disp" : R_bc_disp,
            "R_bc_stre" : R_bc_stre,
            "R_ic_disp" : R_ic_disp,
            "R_ic_vel" : R_ic_vel
        }
    
    def loss_func(self, x_pde, t_pde, x_obs, t_obs, w_obs, lambds):
        residuals = self.calc_residuals(x_pde, t_pde, x_obs, t_obs, w_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]
        R_bc_disp = residuals["R_bc_disp"]
        R_bc_stre = residuals["R_bc_stre"]
        R_ic_disp = residuals["R_ic_disp"]
        R_ic_vel = residuals["R_ic_vel"]

        L_obs = lambds['obs'] * torch.mean(R_obs**2)
        L_pde = lambds['pde'] * torch.mean(R_pde**2)
        L_bc = lambds['bc'] * (torch.mean(R_bc_disp**2) + torch.mean(R_bc_stre**2))
        L_ic = lambds['ic'] * (torch.mean(R_ic_disp**2) + torch.mean(R_ic_vel**2))
        loss = L_obs + L_pde + L_bc + L_ic

        return loss, [L_obs, L_pde, L_bc, L_ic]
    
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
        

class beam_mode_pinn(nn.Module):
    
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hidden
        self.n_layers = n_layers
        self.activation = nn.Tanh
        self.params = {
            'E' : 0.0,
            'rho' : 0.0,
            'I' : 0.0,
            'A' : 0.0,
            'l' : 0.0
        }

        self.build()
    
    def build(self):
        self.fcs = nn.Sequential(*[
                        nn.Linear(self.n_input, self.n_hidden),
                        self.activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(self.n_hidden, self.n_hidden),
                            self.activation()]) for _ in range(self.n_layers-1)])
        self.fce = nn.Linear(self.n_hidden, self.n_output)

    def set_phys_params(self, E, rho, I, A, l):
        var_params = []
        for param, key in zip([E, rho, I, A, l], ['E', 'rho', 'I', 'A', 'l']):
            if param['type'] == 'constant':
                self.params[key] = param['value']
            else:
                var_params.append(param['value'])
        if len(var_params) > 0:
            self.register_parameter("params", nn.Parameter(torch.tensor(var_params)))

    def set_norm_params(self, alphas, pde_norm_type):
        self.norm_ = 1
        self.alpha_c = alphas["c"]
        self.alpha_k = alphas["k"]
        self.alpha_t = alphas["t"]

        # self.alpha_x = alphas["x"]
        self.alpha_x = 1.0
        match pde_norm_type:
            case "trad":
                self.alpha_d0 = self.alpha_x
                self.alpha_d1 = self.alpha_x / self.alpha_t
                self.alpha_d2 = self.alpha_x / (self.alpha_t**2)
            case "up_time":
                self.alpha_d0 = self.alpha_x * self.alpha_t
                self.alpha_d1 = self.alpha_x
                self.alpha_d2 = self.alpha_x / self.alpha_t
            case "up_time2":
                self.alpha_d0 = self.alpha_x * self.alpha_t**2
                self.alpha_d1 = self.alpha_x * self.alpha_t
                self.alpha_d2 = self.alpha_x
        
    def forward(self, t, x):
        X = torch.cat((t.view(-1,1), x.view(-1,1)))
        X = self.fcs(X)
        X = self.fch(X)
        X = self.fce(X)
        return X

    def ode_residual(self, t_pde_hat, x_pde_hat, t_obs, x_obs, q_obs):

        match self.param_type:
            case "constant":
                self.m_hat = self.alpha_d2
                self.c_hat = self.alpha_d1 * self.c
                self.k_hat = self.alpha_d0 * self.k

            case "variable":
                self.m_hat = self.alpha_d2
                self.c_hat = self.alpha_d1 * self.params[0] * self.alpha_c
                self.k_hat = self.alpha_d0 * self.params[1] * self.alpha_k

        # observation loss
        qp_obs = self.forward(t_obs, x_obs)  # N_y-hat or N_y (in Ω_a)
        R_obs = qp_obs - q_obs

        # pde loss
        qp_pde_hat = self.forward(t_pde_hat, x_pde_hat)   # N_y-hat (in Ω_ode)
        dx = torch.autograd.grad(qp_pde_hat, t_pde_hat, torch.ones_like(qp_pde_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, t_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat
        R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * qp_pde_hat

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde
        }

    def loss_func(self, t_pde, t_obs, x_obs, lambds):
        residuals = self.ode_residual(t_pde, t_obs, x_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]

        L_obs = lambds[0] * torch.mean(R_obs**2)
        L_pde = lambds[1] * torch.mean(R_pde**2)
        loss = L_obs + L_pde
        return loss, [L_obs, L_pde]


class beam_parall_pinn(nn.Module):

    def __init__(self, n_input, n_output, n_hiddens, n_layers, n_modes):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.params = {
            'E' : 0.0,
            'rho' : 0.0,
            'I' : 0.0,
            'A' : 0.0,
            'l' : 0.0
        }

        self.build()

    def build(self):
        seq_NNs = [None] * self.n_modes
        for n in range(self.n_modes):
            seq_NNs[n] = beam_mode_pinn(
                self.n_input,
                self.n_output,
                self.n_hiddens[n],
                self.n_layers[n]
                )
        self.ANNs = seq_NNs

    def set_phys_params(self, E, rho, I, A, l):
        var_params = []
        for param, key in zip([E, rho, I, A, l], ['E', 'rho', 'I', 'A', 'l']):
            if param['type'] == 'constant':
                self.params[key] = param['value']
            else:
                var_params.append(param['value'])
        if len(var_params) > 0:
            self.register_parameter("params", nn.Parameter(torch.tensor(var_params)))


    # def loss_func(self, t_pde_hat, x_pde_hat, t_obs_hat, x_obs_hat, q_obs_hat):
    #     pde_losses = torch.zeros(self.n_modes)

    #     for n in range(self.n_modes):
    #         pdelosses[n] = self.ANNs[n].




    


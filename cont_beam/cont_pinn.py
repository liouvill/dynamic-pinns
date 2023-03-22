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
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
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

    def loss_func(self, x_obs, t_obs, w_obs, phi_obs):
        XX = torch.cat((x_obs.view(-1,1), t_obs.view(-1,1)), dim=1)
        wp_obs = torch.zeros((x_obs.shape[0], self.n_modes))
        for n in range(self.n_modes):
            wp_obs[:,n] = self.seq_NNs[n].forward(XX).view(-1) * phi_obs[:,n]
        loss = torch.mean((torch.sum(wp_obs, dim=1) - w_obs)**2)
        return loss


class beam_pinn(nn.Module):

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

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
                    "dt2" : self.pA * self.c * alpha_dt2 * pde_norm_Lambda,
                    "dt1" : self.EI * alpha_dt1 * pde_norm_Lambda,
                    "dx4" : self.pA * alpha_dx4 * pde_norm_Lambda
                }

            case "variable":
                self.pde_alphas = {
                    "dt2" : alpha_dt2 * pde_norm_Lambda,
                    "dt1" : alpha_dt1 * pde_norm_Lambda,
                    "dx4" : alpha_dx4 * pde_norm_Lambda
                }

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
    def pde_residual(self, x_pde_hat, t_pde_hat, x_obs, t_obs, w_obs):
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

        # observation loss
        wh_obs = self.forward(XX_obs)
        R_obs = w_obs - wh_obs

        # pde loss
        wh_pde_hat = self.forward(XX_pde)
        dx = torch.autograd.grad(wh_pde_hat, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂_x-hat N_w-hat
        dx2 = torch.autograd.grad(dx, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂^2_x-hat N_w-hat
        dx3 = torch.autograd.grad(dx2, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂^3_x-hat N_w-hat
        dx4 = torch.autograd.grad(dx3, x_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂^4_x-hat N_w-hat

        dt = torch.autograd.grad(wh_pde_hat, t_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_w-hat
        dt2 = torch.autograd.grad(dt, t_pde_hat, torch.ones_like(wh_pde_hat), create_graph=True)[0]  # ∂^2_t-hat N_w-hat

        R_pde = self.m_hat * dt2 + self.c_hat * dt + self.k_hat * dx4

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde
        }
    
    def loss_func(self, x_pde, t_pde, x_obs, t_obs, w_obs, lambds):
        residuals = self.pde_residual(x_pde, t_pde, x_obs, t_obs, w_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]

        L_obs = lambds[0] * torch.mean(R_obs**2)
        L_pde = lambds[1] * torch.mean(R_pde**2)
        loss = L_obs + L_pde

        return loss, [L_obs, L_pde]
    
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




    


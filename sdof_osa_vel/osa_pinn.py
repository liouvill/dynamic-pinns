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
        else:
            dmax = max_mag_data(data)
        data_norm = data/dmax
        return data_norm, dmax
    

class osa_pinn_sdof(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.activation = nn.Tanh

        self.build_net()
        if config["phys_params"]["par_type"] == "variable":
            self.build_ed_net()

        self.configure(**config)

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def build_ed_net(self):
        self.ed_net = nn.Sequential(
            nn.Sequential(*[nn.Linear(1, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.ed_net
    
    def forward(self, x0, v0, t, f=None, G=0.0, D=1.0):
        # y_ = self.net(x)
        # y = self.G + self.D * y_
        if f is None:
            x = torch.cat((x0.view(-1,1), v0.view(-1,1), t.view(-1,1)), dim=1)
        else:
            x = torch.cat((x0.view(-1,1), v0.view(-1,1), t.view(-1,1), f.view(-1,1)), dim=1)
        y = G + D * self.net(x)
        return y
    
    def configure(self, **config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.forcing = config["forcing"]
        self.param_type = config["phys_params"]["par_type"]

        self.nct = config["nct"]  # number of time collocation points

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"linear"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.phys_params = torch.tensor([self.c, self.k])
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"cubic"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.k3 = config["phys_params"]['k3']
                self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"linear"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"]])))
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"cubic"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"], config["phys_params"]["k3"]])))
        match config["forcing"]:
            case dict():
                self.force = torch.tensor(config["forcing"]["F_tild"])

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_v = config["alphas"]["v"]
        if config["forcing"] != None:
            self.alpha_F = config["alphas"]["F"]
        
        match config:
            case {"nonlinearity":"linear","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_cc = self.alpha_x/(self.alpha_t*self.alpha_v)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "cc" : alpha_cc
                }
            case {"nonlinearity":"linear","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_cc = self.alpha_x/(self.alpha_t*self.alpha_v)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "cc" : alpha_cc,
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_cc = self.alpha_x/(self.alpha_t*self.alpha_v)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "cc" : alpha_cc
                }
            case {"nonlinearity":"cubic","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_cc = self.alpha_x/(self.alpha_t*self.alpha_v)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "cc" : alpha_cc,
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, v_data, f_data=None):

        n_obs = x_data.shape[0]-1

        # Observation set (uses displacement one data point ahead)
        self.x_obs = x_data[:-1].view(-1,1)  # initial displacement input
        self.v_obs = v_data[:-1].view(-1,1)  # initial velocity input
        # self.t_obs = self.T_hat*torch.ones((n_obs,1))  # time at end of horizon (window)
        self.t_obs = torch.zeros((n_obs,1))
        for i in range(n_obs):
            self.t_obs[i] = t_data[i+1] - t_data[i]  # time at end of horizon (window)
        if f_data is not None:
            self.f_obs = f_data[:-1].view(-1,1)  # force input
        self.yy_obs = torch.cat((x_data[1:].view(-1,1), v_data[1:].view(-1,1)), dim=1).requires_grad_()  # displacement at end of window (output)

        # Collocation set (sets a copy of the x0, v0 for a vector of time over the time horizon)
        x_col = torch.zeros((n_obs*self.nct,1))
        v_col = torch.zeros((n_obs*self.nct,1))
        t_col = torch.zeros((n_obs*self.nct,1))
        f_col = torch.zeros((n_obs*self.nct,1))
        D_col = torch.ones((n_obs*self.nct,1))
        G_col = torch.zeros((n_obs*self.nct,1))
        t_pred = torch.zeros((n_obs*self.nct,1))

        for i in range(n_obs):
            # t_wind = torch.linspace(0, t_data[i+1]-t_data[i], self.nct)
            x_col[self.nct*i:self.nct*(i+1),0] = x_data[i].item()*torch.ones(self.nct)
            v_col[self.nct*i:self.nct*(i+1),0] = v_data[i].item()*torch.ones(self.nct)
            t_col[self.nct*i:self.nct*(i+1),0] = torch.linspace(0, t_data[i+1].item()-t_data[i].item(), self.nct)
            if f_data is not None:
                f_col[self.nct*i:self.nct*(i+1),0] = f_data[i].item()*torch.ones(self.nct)

            D_col[self.nct*i,0] = 0.0
            G_col[self.nct*i,0] = x_data[i]

            # generates a vector of the time for the predicted output, by simply adding the total window onto the current time in the data
            t_pred[self.nct*i:self.nct*(i+1),0] = t_data[i] + torch.linspace(0, t_data[i+1].item()-t_data[i].item(), self.nct)

        self.x_col = x_col.requires_grad_()
        self.v_col = v_col.requires_grad_()
        self.t_col = t_col.requires_grad_()
        if f_data is not None:
            self.f_col = f_col.requires_grad_()
        self.G_col = G_col
        self.D_col = D_col
        
        self.ic_ids = torch.argwhere(t_col[:,0]==torch.tensor(0.0))

        if self.config["phys_params"]["par_type"] == "variable":
            self.t_ed_col = torch.linspace(torch.min(t_data),torch.max(t_data),self.config["phys_params"]["ed_nct"]).reshape(-1,1).requires_grad_()
            self.t_ed_obs = t_data.clone().requires_grad_()
            self.y_ed_obs = x_data.clone().requires_grad_() 

        return t_pred
    
    def calc_residuals(self):

        # generate prediction at observation points
        if self.config["forcing"] is None:
            xh_obs_hat = self.forward(self.x_obs, self.v_obs, self.t_obs)
        else:
            xh_obs_hat = self.forward(self.x_obs, self.v_obs, self.t_obs, self.f_obs)
        R_obs = torch.sqrt(torch.sum((xh_obs_hat - self.yy_obs)**2,dim=1))

        # generate prediction over prediction horizon
        if self.config["forcing"] is None:
            xh_coll_hat = self.forward(self.x_col, self.v_col, self.t_col)
        else:
            xh_coll_hat = self.forward(self.x_col, self.v_col, self.t_col, self.f_col)
        # retrieve derivatives
        dx = torch.autograd.grad(xh_coll_hat[:,0], self.t_col, torch.ones_like(xh_coll_hat[:,0]), create_graph=True)[0]  # ∂_t-hat N_w-hat
        dx2 = torch.autograd.grad(dx, self.t_col, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_w-hat
        dv = torch.autograd.grad(xh_coll_hat[:,1], self.t_col, torch.ones_like(xh_coll_hat[:,1]), create_graph=True)[0]  # ∂_t-hat N_v-hat

        # generate predictions for equation discovery domain
        match self.param_type:
            case "variable":
                xh_ed_obs = self.ed_net(self.t_ed_obs)
                R_ed_b = xh_ed_obs - self.y_ed_obs
                xh_ed_coll = self.ed_net(self.t_ed_col)
                dt_ed = torch.autograd.grad(xh_ed_coll, self.t_ed_col, torch.ones_like(xh_ed_coll), create_graph=True)[0]
                dt2_ed = torch.autograd.grad(dt_ed, self.t_ed_col, torch.ones_like(xh_ed_coll), retain_graph=True)[0]

        # initial condition residual
        R_ic1 = self.alpha_v * self.v_col[self.ic_ids,0] - (self.alpha_x/self.alpha_t) * dx[self.ic_ids,0]
        R_ic2 = self.alpha_x * self.x_col[self.ic_ids,0] - self.alpha_x * xh_coll_hat[self.ic_ids,0]
        R_ic3 = self.alpha_v * self.v_col[self.ic_ids,0] - self.alpha_v * xh_coll_hat[self.ic_ids,1]
        R_ic = torch.cat((R_ic1,R_ic2,R_ic3), dim=1)

        # continuity condition residual
        R_cc1 = self.alpha_v * xh_coll_hat[:,1] - (self.alpha_x/self.alpha_t) * dx[:,0]
        R_cc2 = (self.alpha_v/self.alpha_t) * dv[:,0] - (self.alpha_x/(self.alpha_t**2)) * dx2[:,0]
        R_cc = torch.cat((R_cc1.view(-1,1),R_cc2.view(-1,1)),dim=1)

        # retrieve ode loss parameters
        self.m_hat = self.ode_alphas["d2"]
        match self.param_type:
            case "constant":
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.k3
            case "variable":
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3
        match self.config["forcing"]:
            case dict():
                self.eta = self.ode_alphas["ff"]

        # calculate ode residual
        match self.param_type:
            case "constant":
                match self.config:
                    case {"nonlinearity":"linear","forcing":None}:
                        R_ode = self.m_hat * dx2[:,0] + self.c_hat * dx[:,0] + self.k_hat * xh_coll_hat[:,0]
                    case {"nonlinearity":"cubic","forcing":None}:
                        R_ode = self.m_hat * dx2[:,0] + self.c_hat * dx[:,0] + self.k_hat * xh_coll_hat[:,0] + self.k3_hat * xh_coll_hat[:,0]**3
                    case {"nonlinearity":"linear","forcing":{}}:
                        R_ode = self.m_hat * dx2[:,0] + self.c_hat * dx[:,0] + self.k_hat * xh_coll_hat[:,0] - self.eta * self.f_col[:,0]
                    case {"nonlinearity":"cubic","forcing":{}}:
                        R_ode = self.m_hat * dx2[:,0] + self.c_hat * dx[:,0] + self.k_hat * xh_coll_hat[:,0] + self.k3_hat * xh_coll_hat[:,0]**3 - self.eta * self.f_col[:,0]

                return {
                    "R_obs" : R_obs,
                    "R_ic" : R_ic,
                    "R_cc" : R_cc,
                    "R_ode" : R_ode
                }
            case "variable":
                match self.config:
                    case {"nonlinearity":"linear","forcing":None}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll
                    case {"nonlinearity":"cubic","forcing":None}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll + self.k3_hat * xh_ed_coll**3
                    case {"nonlinearity":"linear","forcing":{}}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat - self.eta * self.force
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll - self.eta*self.force
                    case {"nonlinearity":"cubic","forcing":{}}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3 - self.eta * self.force
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll + self.k3_hat * xh_ed_coll**3 - self.eta * self.force

                return {
                    "R_obs" : R_obs,
                    "R_ic" : R_ic,
                    "R_ode" : R_ode,
                    "R_ed_b" : R_ed_b,
                    "R_ed_p" : R_ed_p
                }

    def loss_func(self, lambdas):
        residuals = self.calc_residuals()
        R_obs = residuals["R_obs"]
        R_ic = residuals["R_ic"]
        R_cc = residuals["R_cc"]
        R_ode = residuals["R_ode"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        if R_ic.shape[1]>1:
            L_ic = lambdas['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)
        else:
            L_ic = lambdas['ic'] * torch.mean(R_ic**2,dim=0)
        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2,dim=0),dim=0)
        L_ode = lambdas['ode'] * torch.mean(R_ode**2)

        match self.param_type:
            case "variable":
                R_ed_b = residuals["R_ed_b"]
                L_ed_b = lambdas['ed_b'] * torch.mean(R_ed_b**2)
                R_ed_p = residuals["R_ed_p"]
                L_ed_p = lambdas['ed_p'] * torch.mean(R_ed_p**2)
                loss = L_obs + L_ic + L_ode + L_ed_b + L_ed_p

                return loss, [L_obs, L_ic, L_ode, L_ed_b, L_ed_p]
            case "constant":
                loss = L_obs + L_ic + L_cc + L_ode
                return loss, [L_obs, L_ic, L_cc, L_ode]
    
    def predict(self):
        if self.config['forcing'] is None:
            xp = self.forward(self.x_col, self.v_col, self.t_col)#, self.G_col, self.D_col)
        else:
            xp = self.forward(self.x_col, self.v_col, self.t_col, self.f_col)
        match self.param_type:
            case "variable":
                xp_ed = self.ed_net(self.t_ed_col)
                return xp, xp_ed, self.t_ed_col
            case "constant":
                return xp
    
    def predict_new(self, x0, v0, t, f0=None):
        if f0 is None:
            X = torch.tensor([x0, v0, t])
            xp = self.net(X)
        else:
            X = torch.tensor([x0, v0, t, f0])
            xp = self.net(X)
        return xp

class osa_pinn_sdof_acc(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.activation = nn.Tanh

        self.build_net()
        if config["phys_params"]["par_type"] == "variable":
            self.build_ed_net()

        self.configure(**config)

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def build_ed_net(self):
        self.ed_net = nn.Sequential(
            nn.Sequential(*[nn.Linear(1, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.ed_net
    
    def forward(self, x0, a0, t, f=None, G=0.0, D=1.0):
        # y_ = self.net(x)
        # y = self.G + self.D * y_
        if f is None:
            x = torch.cat((x0.view(-1,1), a0.view(-1,1), t.view(-1,1)), dim=1)
        else:
            x = torch.cat((x0.view(-1,1), a0.view(-1,1), t.view(-1,1), f.view(-1,1)), dim=1)
        y = G + D * self.net(x)
        return y
    
    def configure(self, **config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.forcing = config["forcing"]
        self.param_type = config["phys_params"]["par_type"]

        self.T = config["T"]  # time interval (not normalised)
        self.T_hat = config["T_hat"]  # time interval (normalised)
        self.nct = config["nct"]  # number of time collocation points
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)  # time collocation points in each window

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"linear"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.phys_params = torch.tensor([self.c, self.k])
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"cubic"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.k3 = config["phys_params"]['k3']
                self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"linear"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"]])))
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"cubic"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"], config["phys_params"]["k3"]])))
        match config["forcing"]:
            case dict():
                self.force = torch.tensor(config["forcing"]["F_tild"])

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_a = config["alphas"]["a"]
        if config["forcing"] != None:
            self.alpha_F = config["alphas"]["F"]
        
        match config:
            case {"nonlinearity":"linear","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"linear","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }

    def set_colls_and_obs(self, t_data, x_data, a_data, f_data=None):

        n_obs = x_data.shape[0]-1

        # Observation set (uses displacement one data point ahead)
        self.x_obs = x_data[:-1].view(-1,1)  # initial displacement input
        self.a_obs = a_data[:-1].view(-1,1)  # initial velocity input
        self.t_obs = self.T_hat*torch.ones((n_obs,1))  # time at end of horizon (window)
        if f_data is not None:
            self.f_obs = f_data[:-1].view(-1,1)  # force input
        self.yy_obs = x_data[1:].view(-1,1).requires_grad_()  # displacement at end of window (output)

        # Collocation set (sets a copy of the x0, v0 for a vector of time over the time horizon)
        x_col = torch.zeros((n_obs*self.nct,1))
        a_col = torch.zeros((n_obs*self.nct,1))
        t_col = torch.zeros((n_obs*self.nct,1))
        f_col = torch.zeros((n_obs*self.nct,1))
        D_col = torch.ones((n_obs*self.nct,1))
        G_col = torch.zeros((n_obs*self.nct,1))
        t_pred = torch.zeros((n_obs*self.nct,1))

        for i in range(n_obs):
            x_col[self.nct*i:self.nct*(i+1),0] = x_data[i].item()*torch.ones(self.nct)
            a_col[self.nct*i:self.nct*(i+1),0] = a_data[i].item()*torch.ones(self.nct)
            t_col[self.nct*i:self.nct*(i+1),0] = self.t_wind.clone()
            if f_data is not None:
                f_col[self.nct*i:self.nct*(i+1),0] = f_data[i].item()*torch.ones(self.nct)

            D_col[self.nct*i,0] = 0.0
            G_col[self.nct*i,0] = x_data[i]

            # generates a vector of the time for the predicted output, by simply adding the total window onto the current time in the data
            t_pred[self.nct*i:self.nct*(i+1),0] = t_data[i] + self.t_wind

        self.x_col = x_col.requires_grad_()
        self.a_col = a_col.requires_grad_()
        self.t_col = t_col.requires_grad_()
        if f_data is not None:
            self.f_col = f_col.requires_grad_()
        self.G_col = G_col
        self.D_col = D_col
        
        self.ic_ids = torch.argwhere(t_col[:,0]==torch.tensor(0.0))

        if self.config["phys_params"]["par_type"] == "variable":
            self.t_ed_col = torch.linspace(torch.min(t_data),torch.max(t_data),self.config["phys_params"]["ed_nct"]).reshape(-1,1).requires_grad_()
            self.t_ed_obs = t_data.clone().requires_grad_()
            self.y_ed_obs = x_data.clone().requires_grad_() 

        return t_pred
    
    def calc_residuals(self):

        # generate prediction at observation points
        if self.config["forcing"] is None:
            xh_obs_hat = self.forward(self.x_obs, self.a_obs, self.t_obs)
        else:
            xh_obs_hat = self.forward(self.x_obs, self.a_obs, self.t_obs, self.f_obs)
        R_obs = xh_obs_hat - self.yy_obs

        # generate prediction over prediction horizon
        if self.config["forcing"] is None:
            xh_coll_hat = self.forward(self.x_col, self.a_col, self.t_col)
        else:
            xh_coll_hat = self.forward(self.x_col, self.a_col, self.t_col, self.f_col)
        # retrieve derivatives
        dx = torch.autograd.grad(xh_coll_hat, self.t_col, torch.ones_like(xh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, self.t_col, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat

        # generate predictions for equation discovery domain
        match self.param_type:
            case "variable":
                xh_ed_obs = self.ed_net(self.t_ed_obs)
                R_ed_b = xh_ed_obs - self.y_ed_obs
                xh_ed_coll = self.ed_net(self.t_ed_col)
                dt_ed = torch.autograd.grad(xh_ed_coll, self.t_ed_col, torch.ones_like(xh_ed_coll), create_graph=True)[0]
                dt2_ed = torch.autograd.grad(dt_ed, self.t_ed_col, torch.ones_like(xh_ed_coll), retain_graph=True)[0]

        # initial condition residual
        R_ic1 = self.alpha_a * self.a_col[self.ic_ids] - (self.alpha_x/(self.alpha_t**2)) * dx2[self.ic_ids]
        R_ic2 = self.x_col[self.ic_ids] - xh_coll_hat[self.ic_ids]
        R_ic = torch.cat((R_ic1,R_ic2), dim=1)

        # retrieve ode loss parameters
        self.m_hat = self.ode_alphas["d2"]
        match self.param_type:
            case "constant":
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.k3
            case "variable":
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3
        match self.config["forcing"]:
            case dict():
                self.eta = self.ode_alphas["ff"]

        # calculate ode residual
        match self.param_type:
            case "constant":
                match self.config:
                    case {"nonlinearity":"linear","forcing":None}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat
                    case {"nonlinearity":"cubic","forcing":None}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3
                    case {"nonlinearity":"linear","forcing":{}}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat - self.eta * self.f_col
                    case {"nonlinearity":"cubic","forcing":{}}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3 - self.eta * self.f_col

                return {
                    "R_obs" : R_obs,
                    "R_ic" : R_ic,
                    "R_ode" : R_ode
                }
            case "variable":
                match self.config:
                    case {"nonlinearity":"linear","forcing":None}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll
                    case {"nonlinearity":"cubic","forcing":None}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll + self.k3_hat * xh_ed_coll**3
                    case {"nonlinearity":"linear","forcing":{}}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat - self.eta * self.force
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll - self.eta*self.force
                    case {"nonlinearity":"cubic","forcing":{}}:
                        R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3 - self.eta * self.force
                        R_ed_p = self.m_hat * dt2_ed + self.c_hat * dt_ed + self.k_hat * xh_ed_coll + self.k3_hat * xh_ed_coll**3 - self.eta * self.force

                return {
                    "R_obs" : R_obs,
                    "R_ic" : R_ic,
                    "R_ode" : R_ode,
                    "R_ed_b" : R_ed_b,
                    "R_ed_p" : R_ed_p
                }

    def loss_func(self, lambdas):
        residuals = self.calc_residuals()
        R_obs = residuals["R_obs"]
        R_ic = residuals["R_ic"]
        R_ode = residuals["R_ode"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ic = lambdas['ic'] * torch.sum(torch.mean(R_ic**2,dim=0),dim=0)[0]
        L_ode = lambdas['ode'] * torch.mean(R_ode**2)

        match self.param_type:
            case "variable":
                R_ed_b = residuals["R_ed_b"]
                L_ed_b = lambdas['ed_b'] * torch.mean(R_ed_b**2)
                R_ed_p = residuals["R_ed_p"]
                L_ed_p = lambdas['ed_p'] * torch.mean(R_ed_p**2)
                loss = L_obs + L_ic + L_ode + L_ed_b + L_ed_p

                return loss, [L_obs, L_ic, L_ode, L_ed_b, L_ed_p]
            case "constant":
                loss = L_obs + L_ic + L_ode
                return loss, [L_obs, L_ic, L_ode]
    
    def predict(self):
        if self.config['forcing'] is None:
            xp = self.forward(self.x_col, self.a_col, self.t_col)#, self.G_col, self.D_col)
        else:
            xp = self.forward(self.x_col, self.a_col, self.t_col, self.f_col)
        match self.param_type:
            case "variable":
                xp_ed = self.ed_net(self.t_ed_col)
                return xp, xp_ed, self.t_ed_col
            case "constant":
                return xp


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

    def predict(self, tp):
        yp = self.forward(tp)
        return yp

    def loss_func(self, x_obs, y_obs):
        yp_obs = self.forward(x_obs)
        loss = torch.mean((yp_obs - y_obs)**2)
        return loss

class ParamClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'phys_params'):
            params = module.phys_params.data
            params = params.clamp(0,1)
            module.phys_params.data = params
            
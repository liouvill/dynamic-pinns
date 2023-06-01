import torch
import toybox as tb
import numpy as np
from math import pi
import scipy

def add_noise(x, noise_std, seed=160151156156):
    torch.manual_seed(seed)
    std = noise_std * torch.amax(x)
    x = x + torch.normal(0.0, std, (x.shape[0],1))
    return x

def mdof_solution(time, **config): #params, init_conds, forcing=None, nonlinearity="linear"):
    nt = time.shape[0]

    params = config["params"]
    init_conds = config["init_conds"]
    forcing = config["forcing"]
    n_dofs = config["n_dofs"]

    x0 = init_conds["x0"]
    v0 = init_conds["v0"]

    mdof_system = tb.system(config["params"]["M"], config["params"]["C"], config["params"]["K"])

    match config:
        case {"nonlinearity":"linear"}:
            mdof_system.N = None
        #TODO: add nonlinearities to simulator
        case {"nonlinearity":"cubic","m_norm":True}:
            def cub_stiff(_, t, y, ydot):
                return np.dot(y**3, np.array(params["k3"]))
            mdof_system.N = cub_stiff
        case {"nonlinearity":"cubic","m_norm":False}:
            def cub_stiff(_, t, y, ydot):
                return np.dot(y**3, params["k3"])
            mdof_system.N = cub_stiff

    match config:
        case {"forcing":None}:
            mdof_system.excitation = [None]
        case {"forcing":dict()}:
            mdof_system.excitation = forcing["F"].T
    
    w0 = np.concatenate((np.array(x0),np.array(v0)), axis=0)
    data = mdof_system.simulate((nt, time[1]), w0=w0, normalise=False)
    x = torch.zeros((nt, n_dofs))
    v = torch.zeros((nt, n_dofs))
    for n in range(n_dofs):
        x[:,n] = torch.tensor(data[("y"+str(n+1))]).to(torch.float32).squeeze()
        v[:,n] = torch.tensor(data[("ydot"+str(n+1))]).to(torch.float32).squeeze()
    return x.squeeze(), v.squeeze()

def generate_excitation(time, **exc_config):

    F0 = exc_config["F0"]
    n_modes = exc_config["n_dofs"]
    time = time.reshape(-1,1)
    F = np.zeros((time.shape[0], n_modes))

    match exc_config["type"]:
        case "sinusoid":
            for n in range(n_modes):
                w = exc_config["w"][n]
                F[:,n] = (F0 * np.sin(w * time)).reshape(-1)
        case "white_gaussian":
            u = exc_config["offset"]
            sig = F0
            for n in range(n_modes):
                match exc_config:
                    case {"seed" : int() as seed}:
                        np.random.seed(seed+n)
                    case _:
                        np.random.seed(43810+n)
                F[:,n] = np.random.normal(u, sig, size=(time.shape[0],1))
        case "sine_sweep":
            for n in range(n_modes):
                f0 = exc_config["w"][n][0] / (2*pi)
                f1 = exc_config["w"][n][1] / (2*pi)
                F[:,n] = F0*scipy.signal.chirp(time, f0, time[-1], f1, method=exc_config["scale"])
        case "rand_phase_ms":
            for n in range(n_modes):
                freqs = exc_config["freqs"][n].reshape(-1,1)
                Sx = exc_config["Sx"][n].reshape(-1,1)
                match exc_config:
                    case {"seed" : int() as seed}:
                        np.random.seed(seed+n)
                    case _:
                        np.random.seed(43810+n)
                phases = np.random.rand(freqs.shape[0], 1)
                F_mat = np.sin(time @ freqs.T + phases.T)
                F[:,n] = (F_mat @ Sx).reshape(-1)
    return F
        
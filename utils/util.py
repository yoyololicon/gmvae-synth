import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


def get_instance(module, name, config, *args):
    func_args = config[name]['args'] if 'args' in config[name] else None

    if func_args:
        return getattr(module, config[name]['type'])(*args, **func_args)
    else:
        return getattr(module, config[name]['type'])(*args)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(x, fname, if_sort_key=False, n_indent=None):
    with open(fname, 'w') as outfile:
        json.dump(x, outfile, sort_keys=if_sort_key, indent=n_indent)


def log_gauss(q_z, mu, logdet_cov, precision):
    z_dim = q_z.shape[-1]
    q_z = q_z.unsqueeze(1)  # (batch, 1, z_dim)
    dist = q_z - mu  # (batch, k, z_dim)
    llh = -0.5 * (z_dim * np.log(2 * np.pi) + logdet_cov + torch.einsum('bkz,kzl,bkl->bk', dist, precision, dist))
    return llh


def approx_q_y(q_z, mu_lookup, logs_lookup, rho_lookup, k=10):
    """
    refer to eq.13 in the paper
    """
    q_z_shape = list(q_z.size())  # (b, z_dim)
    mu_lookup_shape = [mu_lookup.num_embeddings, mu_lookup.embedding_dim]  # (k, z_dim)
    logs_lookup_shape = [logs_lookup.num_embeddings, logs_lookup.embedding_dim]  # (k, 1)
    rho_lookup_shape = [rho_lookup.num_embeddings, rho_lookup.embedding_dim]  # (k, 1)

    if not mu_lookup_shape[0] == k:
        raise ValueError("mu_lookup_shape (%s) does not match the given k (%s)" % (
            mu_lookup_shape, k))

    if not q_z_shape[1] == mu_lookup_shape[1]:
        raise ValueError("q_z_shape (%s) does not match mu_lookup_shape (%s) in dimension of z" % (
            q_z_shape, mu_lookup_shape))

    mu, logs, rho = mu_lookup.weight, logs_lookup.weight.squeeze(), rho_lookup.weight.squeeze().tanh() * 0.99999
    z_dim = q_z_shape[1]
    logdet_cov = rho_logdet_cov(logs, rho, z_dim)

    # get inverse of AR(1) covariance matrix
    # https://math.stackexchange.com/questions/975069/the-inverse-of-ar-structure-correlation-matrix-kac-murdock-szeg-%CC%88o-matrix
    precision_matrix = rho_precision(logs, rho, z_dim)
    log_q_y_logit = log_gauss(q_z, mu, logdet_cov, precision_matrix)
    q_y = torch.nn.functional.softmax(log_q_y_logit, dim=1)
    return log_q_y_logit, q_y


def rho_L(logs, rho, z_dim):
    L = F.pad(rho.unsqueeze(1) ** torch.arange(z_dim, dtype=logs.dtype, device=logs.device),
              (z_dim - 1, 0)).unfold(1, z_dim, 1).flip(-1)
    sigma = torch.exp(0.5 * logs)[:, None, None] * L * F.pad(
        torch.sqrt(1 - rho * rho).unsqueeze(1).expand(-1, z_dim - 1), (1, 0), value=1.).unsqueeze(1)
    return sigma


def rho_L_inv(logs, rho, z_dim):
    L_inv = torch.eye(z_dim, dtype=logs.dtype, device=logs.device) + \
            torch.diag_embed(-rho.unsqueeze(1).expand(-1, z_dim - 1), offset=-1)
    sigma = torch.exp(-0.5 * logs)[:, None, None] * L_inv * F.pad(
        torch.rsqrt(1 - rho * rho).unsqueeze(1).expand(-1, z_dim - 1), (1, 0), value=1.).unsqueeze(2)
    return sigma


def rho_cov(logs, rho, z_dim):
    tmp = torch.arange(z_dim, dtype=logs.dtype, device=logs.device)
    tmp = torch.cat((tmp, tmp.flip(0)[:-1]))
    return logs.exp()[:, None, None] * (rho.unsqueeze(1) ** tmp).unfold(1, z_dim, 1).flip(-1)


def rho_logdet_cov(logs, rho, z_dim):
    return z_dim * logs + (z_dim - 1) * torch.log(1 - rho * rho)


def rho_precision(logs, rho, z_dim):
    precision_matrix = (torch.diag_embed(-rho.unsqueeze(1).expand(-1, z_dim - 1), offset=-1)
                        + torch.diag_embed(-rho.unsqueeze(1).expand(-1, z_dim - 1), offset=1)
                        + torch.diag_embed(F.pad((1 + rho * rho).unsqueeze(1).expand(-1, z_dim - 2), (1, 1), value=1.))) \
                       * (torch.exp(-logs) / (1 - rho * rho))[:, None, None]
    return precision_matrix

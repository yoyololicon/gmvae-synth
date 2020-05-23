import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from base.base_loss import BaseLoss  # why do I need to explicity import this time?
from utils import log_gauss, approx_q_y, rho_cov, rho_logdet_cov, rho_precision


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(x_predict, x, reduction="none"):
    loss = F.mse_loss(x_predict, x, reduction=reduction)
    if len(loss.size()) > 2:
        loss = torch.sum(loss, dim=-1)
    return torch.sum(loss, dim=1)


def bce_loss(x_predict, x, reduction="none"):
    loss = F.binary_cross_entropy_with_logits(x_predict, x, reduction=reduction)
    return torch.sum(loss, dim=1)


def ce_loss(x_predict, x, reduction="none", is_train=True, label_idx=None):
    if not label_idx:
        loss = torch.zeros(1)
    else:
        if is_train:
            x_predict = x_predict[label_idx]
            x = x[label_idx]
        loss = F.cross_entropy(x_predict, x, reduction=reduction)
    return loss


def pitch_ce_loss(x_predict, x, reduction="none"):
    loss = F.cross_entropy(x_predict, x, reduction=reduction)
    return loss


def kl_gauss(q_mu, q_logvar, mu=None, logvar=None):
    """
    KL divergence between two diagonal gaussians
    """
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    return -0.5 * (1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar))


def kl_gauss_full(q_mu, q_cov, q_logdet_cov, mu=None, cov_inv=None, logdet_cov=None, overbatch=False):
    """
    KL divergence between two gaussians
    """
    z_dim = q_mu.shape[-1]
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if cov_inv is None:
        cov_inv = torch.eye(z_dim).unsqueeze(0).expand(mu.shape[0], -1)
        logdet_cov = torch.zeros_like(mu[:, 0])
    if logdet_cov is None:
        logdet_cov = torch.slogdet(cov_inv)[1]

    if not overbatch:
        dist = mu - q_mu.unsqueeze(1)
        return 0.5 * (torch.einsum('...ii->...', torch.einsum('klm,bmn->bkln'), cov_inv, q_cov) +
                      torch.einsum('bki,kij,bkj->bk', dist, cov_inv, dist) +
                      logdet_cov - q_logdet_cov.unsqueeze(1) - z_dim)
    else:
        dist = mu - q_mu
        return 0.5 * (torch.einsum('...ii->...', cov_inv @ q_cov) +
                      torch.einsum('bi,bij,bj->b', dist, cov_inv, dist) +
                      logdet_cov - q_logdet_cov - z_dim)


def kl_class(log_q_y_logit, q_y, k=10):
    q_y_shape = list(q_y.size())

    if not q_y_shape[1] == k:
        raise ValueError("q_y_shape (%s) does not match the given k (%s)" % (
            q_y_shape, k))

    h_y = torch.sum(q_y * torch.nn.functional.log_softmax(log_q_y_logit, dim=1), dim=1)

    return h_y - np.log(1 / k), h_y


def kl_latent(q_mu, q_logs, q_rho, q_y, mu_lookup, logs_lookup, rho_lookup):
    """
    q_z (b, z)
    q_y (b, k)
    mu_lookup (k, z)
    logvar_lookup (k, z)
    """
    mu_lookup_shape = [mu_lookup.num_embeddings, mu_lookup.embedding_dim]  # (k, z_dim)
    logs_lookup_shape = [logs_lookup.num_embeddings, logs_lookup.embedding_dim]  # (k, 1)
    rho_lookup_shape = [rho_lookup.num_embeddings, rho_lookup.embedding_dim]  # (k, 1)
    q_mu_shape = list(q_mu.size())
    q_logs_shape = list(q_logs.size())
    q_rho_shape = list(q_rho.size())
    q_y_shape = list(q_y.size())

    if not q_y_shape[0] == q_mu_shape[0]:
        raise ValueError("q_y_shape (%s) and q_mu_shape (%s) do not match in batch size" % (
            q_y_shape, q_mu_shape))
    if not q_y_shape[1] == mu_lookup_shape[0]:
        raise ValueError("q_y_shape (%s) and mu_lookup_shape (%s) do not match in number of class" % (
            q_y_shape, mu_lookup_shape))

    batch_size, n_class = q_y_shape
    z_dim = q_mu.shape[-1]
    q_cov = rho_cov(q_logs, q_rho, z_dim)
    q_logdet_cov = rho_logdet_cov(q_logs, q_rho, z_dim)

    mu = mu_lookup.weight
    logs = logs_lookup.weight.squeeze()
    rho = rho_lookup.weight.squeeze().tanh() * 0.99999
    cov_inv = rho_precision(logs, rho, z_dim)
    logdet_cov = rho_logdet_cov(logs, rho, z_dim)
    kl_sum = torch.einsum('bk,bk->b', kl_gauss_full(q_mu, q_cov, q_logdet_cov, mu, cov_inv, logdet_cov), q_y)

    return kl_sum  # sum over classes


def kl_pitch_emb(pitch_mu_lookup, pitch_logs_lookup, pitch_rho_lookup, pitch_mu, pitch_logs, pitch_rho, y_pitch):
    mu = pitch_mu_lookup(y_pitch)
    logs = pitch_logs_lookup(y_pitch)
    rho = pitch_rho_lookup(y_pitch)

    z_dim = pitch_mu.shape[-1]
    pitch_cov = rho_cov(pitch_logs, pitch_rho, z_dim)
    pitch_logdet_cov = rho_logdet_cov(pitch_logs, pitch_rho, z_dim)

    cov_inv = rho_precision(logs, rho, z_dim)
    logdet_cov = rho_logdet_cov(logs, rho, z_dim)

    return kl_gauss_full(pitch_mu, pitch_cov, pitch_logdet_cov, mu, cov_inv, logdet_cov, overbatch=True)


class KLpitch(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(KLpitch, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, pitch_mu_lookup, pitch_logs_lookup, pitch_rho_lookup, pitch_mu, pitch_logs, pitch_rho,
                 y_pitch):
        if epoch >= self.effect_epoch:
            return self.weight * kl_pitch_emb(pitch_mu_lookup, pitch_logs_lookup, pitch_rho_lookup, pitch_mu,
                                              pitch_logs, pitch_rho,
                                              y_pitch)
        else:
            return torch.zeros(1)


class MSEloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(MSEloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x):
        if epoch >= self.effect_epoch:
            return self.weight * mse_loss(x_predict, x)
        else:
            return torch.zeros(1)


class BCEloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(BCEloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x):
        if epoch >= self.effect_epoch:
            return self.weight * bce_loss(x_predict, x)
        else:
            return torch.zeros(1)


class CEloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(CEloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x, is_train=True, label_idx=None):
        if epoch >= self.effect_epoch:
            return self.weight * ce_loss(x_predict, x, is_train=is_train, label_idx=label_idx)
        else:
            return torch.zeros(1)


class PDloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(PDloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x):
        if epoch >= self.effect_epoch:
            return self.weight * pitch_ce_loss(x_predict, x)
        else:
            return torch.zeros(1)


class KLlatent(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(KLlatent, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, q_mu, q_logs, q_rho, q_y, mu_lookup, logs_lookup, rho_lookup):
        if epoch >= self.effect_epoch:
            return self.weight * kl_latent(q_mu, q_logs, q_rho, q_y, mu_lookup, logs_lookup, rho_lookup)
        else:
            return torch.zeros(1)


class KLclass(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(KLclass, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, log_q_y_logit, q_y, k=10):
        if epoch >= self.effect_epoch:
            kl, h_y = kl_class(log_q_y_logit, q_y, k)
            return self.weight * kl, h_y
        else:
            return torch.zeros(1), torch.zeros(1)


if __name__ == '__main__':
    n_class = 10
    f, t = [80, 15]
    batch_size = 256
    latent_dim = 16
    x = torch.randn(batch_size, f, t)
    y = torch.randn(batch_size, f, t)
    dummy_mu_lookup = nn.Embedding(n_class, latent_dim)
    dummy_logs_lookup = nn.Embedding(n_class, 1)
    dummy_rho_lookup = nn.Embedding(n_class, 1)
    q_z = torch.randn(batch_size, latent_dim)

    q_y = approx_q_y(q_z, dummy_mu_lookup, dummy_logs_lookup, dummy_rho_lookup, k=n_class)
    neg_kld_y = -1 * kl_class(q_y, k=n_class)
    neg_kld_z = -1 * kl_latent(q_z, q_y, dummy_mu_lookup, dummy_logs_lookup)
    reconloss = mse_loss(x, y)
    print(reconloss.size(), neg_kld_y.size(), neg_kld_z.size())

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from model.loss import approx_q_y


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

    mu, logs, rho = mu_lookup.weight, logs_lookup.weight.squeeze(), rho_lookup.squeeze().weight.tanh() * 0.99999
    z_dim = q_z_shape[1]
    logdet_cov = z_dim * logs + (z_dim - 1) * torch.log(1 - rho * rho)

    # get inverse of AR(1) covariance matrix
    # https://math.stackexchange.com/questions/975069/the-inverse-of-ar-structure-correlation-matrix-kac-murdock-szeg-%CC%88o-matrix
    precision_matrix = (torch.diag_embed(-rho.unsqueeze(1).expand(-1, z_dim - 1), offset=-1)
                        + torch.diag_embed(-rho.unsqueeze(1).expand(-1, z_dim - 1), offset=1)
                        + torch.diag_embed(F.pad((1 + rho * rho).unsqueeze(1).expand(-1, z_dim - 2), (1, 1), value=1.))) \
                       * (torch.exp(-logs) / (1 - rho * rho))[:, None, None]

    log_q_y_logit = log_gauss(q_z, mu, logdet_cov, precision_matrix)
    q_y = torch.nn.functional.softmax(log_q_y_logit, dim=1)
    return log_q_y_logit, q_y


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
        # print(super(BaseModel, self))


class BaseGMVAE(BaseModel):
    def __init__(self, input_size, latent_dim, n_class=10, is_featExtract=False):
        super(BaseGMVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.n_class = n_class
        self.is_featExtract = is_featExtract
        self._build_mu_lookup()
        self._build_logs_rho_lookup()

    def _encode(self, x):
        """
        implementation should end with
        1. self._infer_latent()
        2. self._infer_class()
        and their outputs combined
        """
        raise NotImplementedError

    def _decode(self, z):
        raise NotImplementedError

    def _infer_latent(self, mu, logs, rho, weight=1):
        if self.is_featExtract:
            """
            only when NOT is_train;
            return mu as the representative latent vector
            """
            return mu, logs, rho, mu

        z_dim = mu.shape[-1]
        L = F.pad(rho.unsqueeze(1) ** torch.arange(z_dim), (z_dim - 1, 0)).unfold(1, z_dim, 1).flip(-1)
        sigma = torch.exp(0.5 * logs)[:, None, None] * L * F.pad(
            torch.sqrt(1 - rho * rho).unsqueeze(1).expand(-1, z_dim - 1), (1, 0), value=1.)
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=mu.size())  # default require_grad=False

        z = mu + weight * torch.einsum('bij,bj->bi', sigma, eps)  # reparameterization trick

        return mu, logs, rho, z

    def _build_mu_lookup(self):
        """
        follow Xavier initialization as in the paper
        """
        mu_lookup = nn.Embedding(self.n_class, self.latent_dim)
        nn.init.xavier_uniform_(mu_lookup.weight)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logs_rho_lookup(self, pow_exp=0, logvar_trainable=False):
        """
        follow Table 7 in the paper
        """
        logs_lookup = nn.Embedding(self.n_class, 1)
        rho_lookup = nn.Embedding(self.n_class, 1)
        # init_sigma = np.exp(-1)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logs_lookup.weight, init_logvar)
        nn.init.constant_(rho_lookup.weight, -5)
        logs_lookup.weight.requires_grad = logvar_trainable
        rho_lookup.weight.requires_grad = logvar_trainable
        self.logs_lookup = logs_lookup
        self.rho_lookup = rho_lookup
        # self.logvar_bound = np.log(np.exp(-1) ** 2)

    def _bound_logvar_lookup(self):
        self.logvar_lookup.weight.data[torch.le(self.logvar_lookup.weight, self.logvar_bound)] = self.logvar_bound

    def _infer_class(self, q_z):
        log_q_y_logit, q_y = approx_q_y(q_z, self.mu_lookup, self.logvar_lookup, k=self.n_class)
        val, ind = torch.max(q_y, dim=1)
        return log_q_y_logit, q_y, ind

    def forward(self, x):
        raise NotImplementedError
        # mu, logvar, z, q_y, ind = self._encode(x)
        # x_predict = x_self._decode(z)
        # return [mu, logvar, z], [q_y, ind], x_predict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, BaseGMVAE


class CnnGMVAE(BaseGMVAE):
    def __init__(self, input_size=(128, 1, 20), latent_dim=16, n_class=10,
                 pow_exp=0, logvar_trainable=False,
                 is_pitch_condition=False, is_pitch_discriminate=False,
                 is_featExtract=False):
        super(CnnGMVAE, self).__init__(input_size, latent_dim, n_class, is_featExtract)
        self.emb_dim = int(latent_dim)
        self.n_channel = input_size[0]
        self.context_size = self.input_size[1]
        self.is_pitch_condition = is_pitch_condition
        self.is_pitch_discriminate = is_pitch_discriminate
        decoder_input_dim = latent_dim if not is_pitch_condition else latent_dim + int(latent_dim)
        self._build_logs_rho_lookup(pow_exp=pow_exp, logvar_trainable=logvar_trainable)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.n_channel, 512, 3, 1),  # padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1),  # padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.flat_size, self.encoder_output_size = self.infer_flat_size(self.encoder)

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.lin_mu = nn.Linear(512, self.latent_dim)
        self.lin_logs_rho = nn.Linear(512, 2)
        self.decoder_fc = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.flat_size),
            nn.BatchNorm1d(self.flat_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 3, 1),  # padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, self.n_channel, 3, 1),  # padding=1),
            nn.Tanh()
        )

        if is_pitch_condition:
            self._build_pitch_lookup()
            self.pitch_encoder = nn.Sequential(
                nn.Conv1d(self.n_channel, 512, 3, 1),  # padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, 3, 1),  # padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
            self.pitch_flat_size, _ = self.infer_flat_size(self.pitch_encoder)
            self.pitch_encoder_fc = nn.Sequential(
                # nn.Linear(self.flat_size, 128),
                nn.Linear(self.pitch_flat_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.pitch_lin_mu = nn.Linear(512, int(self.latent_dim))
            self.pitch_lin_logs_rho = nn.Linear(512, 2)

        if is_pitch_discriminate:
            self.pitch_classifier = nn.Sequential(
                nn.Linear(int(self.latent_dim), 82),
            )

    def _build_pitch_lookup(self):
        pitch_mu_lookup = nn.Embedding(82, self.emb_dim)
        nn.init.xavier_uniform_(pitch_mu_lookup.weight)
        pitch_logs_lookup = nn.Embedding(82, 1)
        pitch_rho_lookup = nn.Embedding(82, 1)
        init_sigma = np.exp(-2)
        # init_sigma = np.exp(-1)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(pitch_logs_lookup.weight, init_logvar)
        pitch_logs_lookup.weight.requires_grad = False
        nn.init.constant_(pitch_rho_lookup.weight, 0)
        pitch_rho_lookup.weight.requires_grad = True
        # pitch_logvar_lookup.weight.requires_grad = True

        self.pitch_mu_lookup = pitch_mu_lookup
        self.pitch_logs_lookup = pitch_logs_lookup
        self.pitch_rho_lookup = pitch_rho_lookup

    def infer_flat_size(self, encoder):
        # encoder_output = self.encoder(torch.ones(1, *self.input_size))
        encoder_output = encoder(torch.ones(1, *self.input_size))
        encoder_output_size = encoder_output.size()[1:]
        return int(np.prod(encoder_output_size)), encoder_output_size

    def _encode(self, x):
        h = self.encoder(x)
        h2 = self.encoder_fc(h.view(-1, self.flat_size))
        mu = self.lin_mu(h2)
        logs, rho = self.lin_logs_rho(h2).chunk(2, -1)
        logs, rho = logs.squeeze(), rho.squeeze().tanh() * 0.99999

        mu, logs, rho, z = self._infer_latent(mu, logs, rho)
        log_q_y_logit, q_y, ind = self._infer_class(z)

        if self.is_pitch_condition:
            h = self.pitch_encoder(x)
            # h2 = self.pitch_encoder_fc(h.view(-1, self.flat_size))
            h2 = self.pitch_encoder_fc(h.view(-1, self.pitch_flat_size))
            pitch_mu = self.pitch_lin_mu(h2)
            pitch_logs, pitch_rho = self.pitch_lin_logs_rho(h2).chunk(2, -1)
            pitch_logs, pitch_rho = pitch_logs.squeeze(), pitch_rho.squeeze().tanh() * 0.99999
            pitch_mu, pitch_logs, pitch_rho, pitch_z = self._infer_latent(pitch_mu, pitch_logs, pitch_rho)
            return mu, logs, rho, z, log_q_y_logit, q_y, ind, pitch_mu, pitch_logs, pitch_rho, pitch_z
        else:
            return mu, logs, rho, z, log_q_y_logit, q_y, ind

    def _decode(self, z):
        h = self.decoder_fc(z)
        y = self.decoder(h.view(-1, *self.encoder_output_size))
        return y

    def forward(self, x):
        if self.is_pitch_condition:
            mu, logs, rho, z, log_q_y_logit, q_y, ind, pitch_mu, pitch_logs, pitch_rho, pitch_z = self._encode(x)
            x_predict = self._decode(torch.cat([z, pitch_z], dim=1))
        else:
            mu, logs, rho, z, log_q_y_logit, q_y, ind = self._encode(x)
            x_predict = self._decode(z)
            pitch_mu = pitch_logs = pitch_rho = pitch_z = None

        if self.is_pitch_discriminate:
            assert self.is_pitch_condition
            pitch_logit = self.pitch_classifier(pitch_z)
        else:
            pitch_logit = None

        return x_predict, mu, logs, rho, z, log_q_y_logit, q_y, ind, pitch_mu, pitch_logs, pitch_rho, pitch_z, pitch_logit

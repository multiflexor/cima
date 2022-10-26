import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_agents, n_actions, name, checkpoint_dir):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, name, checkpoint_dir):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.pi = nn.Linear(64, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class AutoEncoderLinear(nn.Module):
    def __init__(self, input_dims, output_dims, n_actions, name, lr, checkpoint_dir):
        super(AutoEncoderLinear, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        self.dropout_rate = 0.2

        self.encoder = nn.Sequential(
            nn.Linear(input_dims + n_actions, 256),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, output_dims),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class VAE(nn.Module):
    def __init__(self, input_dims, output_dims, n_actions, name, lr, checkpoint_dir):
        super(VAE, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, name)
        # self.dropout_rate = 0.2

        self.encoder = nn.Sequential(
            nn.Linear(input_dims + n_actions, 256),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64)
        )

        # distribution parameters
        self.fc_mu = nn.Linear(64, 32)
        self.fc_var = nn.Linear(64, 32)

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(128, output_dims),
            nn.Tanh()
        )

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(T.Tensor([0.0]))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward_kl(self, x):
        encoded_obs = self.encoder(x)
        mu, log_var = self.fc_mu(encoded_obs), self.fc_var(encoded_obs)

        # std = T.exp(log_var / 2)
        # q = T.distributions.Normal(mu, std)
        # z = q.rsample()
        # kl = self.kl_divergence(z, mu, std)

        kl_vanilla_vae = self.kld_vanilla_vae(mu, log_var)

        return kl_vanilla_vae

    def forward(self, x):
        # encode input to get the mu and variance parameters
        encoded_obs = self.encoder(x)
        mu, log_var = self.fc_mu(encoded_obs), self.fc_var(encoded_obs)

        # sample z from q
        std = T.exp(log_var / 2)
        q = T.distributions.Normal(mu, std)
        z = q.rsample()

        # decode and get reconstruction loss
        predicted_obs = self.decoder(z)

        # kl-divergence (and Intrinsic Reward)
        # kl = self.kl_divergence(z, mu, std)
        kl_vanilla_vae = self.kld_vanilla_vae(mu, log_var)

        return predicted_obs, kl_vanilla_vae

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    @staticmethod
    def gaussian_likelihood(x_hat, log_scale, x):
        scale = T.exp(log_scale)
        mean = x_hat
        dist = T.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        return log_pxz.sum(dim=1)

    @staticmethod
    def kl_divergence(z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = T.distributions.Normal(T.zeros_like(mu), T.ones_like(std))
        q = T.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    @staticmethod
    def kld_vanilla_vae(mu, log_var):
        kl = T.mean(-0.5 * T.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kl

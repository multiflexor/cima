import torch as T
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork, AutoEncoderLinear, VAE


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, scenario,
                 checkpoint_dir, alpha=0.0005, beta=0.0005, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agent_name = 'agent_%s' % agent_idx
        self.scenario = scenario
        self.actor = ActorNetwork(alpha, actor_dims, n_actions,
                                  name=self.agent_name + '_actor', checkpoint_dir=checkpoint_dir)
        self.critic = CriticNetwork(beta, critic_dims, n_agents, n_actions,
                                    name=self.agent_name + '_critic', checkpoint_dir=checkpoint_dir)
        self.target_actor = ActorNetwork(alpha, actor_dims, n_actions,
                                         name=self.agent_name + '_target_actor', checkpoint_dir=checkpoint_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, n_agents, n_actions,
                                           name=self.agent_name + '_target_critic', checkpoint_dir=checkpoint_dir)

        self.update_network_parameters(tau=1)

        if self.scenario == "MADDPG_AE":
            self.env_model = AutoEncoderLinear(actor_dims * n_agents, actor_dims, n_actions * n_agents,
                                               name=self.agent_name + '_env_model',
                                               lr=0.001, checkpoint_dir=checkpoint_dir)
            self.im_reward_multiplier = 20
        elif self.scenario == "MADDPG_VAE":
            self.env_model = VAE(actor_dims * n_agents, actor_dims, n_actions * n_agents,
                                 name=self.agent_name + '_env_model',
                                 lr=0.001, checkpoint_dir=checkpoint_dir)
            self.im_reward_multiplier = 20

    def choose_action(self, observation, noise_rate):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise * noise_rate

        return action.detach().cpu().numpy()[0]

    def get_intrinsic_reward(self, observation, observation_, action):
        if self.scenario == "MADDPG_AE":
            action_ohe = np.zeros((self.n_agents, self.n_actions))
            for i, act in enumerate(action):
                action_ohe[i, act] = 1

            state = np.concatenate(observation)
            state = np.concatenate((state, action_ohe.flatten()))
            state = T.tensor([state], dtype=T.float).to(self.actor.device)
            model_out = self.env_model.forward(state)
            im_reward = F.mse_loss(model_out,
                                   T.from_numpy(np.expand_dims(observation_, 0)).to(
                                       self.env_model.device)).cpu().detach().numpy() * self.im_reward_multiplier
            im_reward = im_reward.tolist()

        elif self.scenario == "MADDPG_VAE":
            action_ohe = np.zeros((self.n_agents, self.n_actions))
            for i, act in enumerate(action):
                action_ohe[i, act] = 1

            state = np.concatenate(observation)
            state = np.concatenate((state, action_ohe.flatten()))
            state = T.tensor([state], dtype=T.float).to(self.actor.device)
            model_out = self.env_model.forward_kl(state)
            im_reward = model_out.cpu().detach().numpy() * self.im_reward_multiplier
            # im_reward = abs(im_reward.tolist()[0])
            im_reward = im_reward.tolist()

        else:
            im_reward = 0

        return im_reward

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

        if self.scenario in ["MADDPG_AE", "MADDPG_VAE"]:
            self.env_model.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

        if self.scenario in ["MADDPG_AE", "MADDPG_VAE"]:
            self.env_model.load_checkpoint()

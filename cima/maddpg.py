import torch as T
import torch.nn.functional as F
import numpy as np
from agent import Agent
from networks import AutoEncoderLinear


class MADDPG:
    def __init__(self, n_agents, obs_shape, n_actions, scenario,
                 alpha, beta, checkpoint_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.scenario = scenario
        checkpoint_dir += scenario
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(obs_shape, obs_shape * n_agents, n_actions, n_agents, agent_idx,
                                     scenario, alpha=alpha, beta=beta, checkpoint_dir=checkpoint_dir))

        if self.scenario == "MADDPG_AE_common":
            self.env_model = AutoEncoderLinear(obs_shape * n_agents, obs_shape * n_agents,
                                               n_actions * n_agents, name='env_model',
                                               lr=0.001, checkpoint_dir=checkpoint_dir)
            self.im_reward_multiplier = 20

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

        if self.scenario == "MADDPG_AE_common":
            self.env_model.save_checkpoint()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

        if self.scenario == "MADDPG_AE_common":
            self.env_model.load_checkpoint()

    def choose_action(self, raw_obs, noise_rate):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], noise_rate)
            actions.append(action)
        return actions

    def get_intrinsic_rewards(self, obs, obs_, actions):
        im_rewards = []
        for agent_idx, agent in enumerate(self.agents):
            im_reward = agent.get_intrinsic_reward(obs, obs_[agent_idx], actions)
            im_rewards.append(im_reward)
        return im_rewards

    def get_intrinsic_rewards_common(self, obs, obs_, actions):
        if self.scenario == "MADDPG_AE_common":
            action_ohe = np.zeros((self.n_agents, self.n_actions))
            for i, act in enumerate(actions):
                action_ohe[i, act] = 1

            state = np.concatenate(obs)
            state = np.concatenate((state, action_ohe.flatten()))
            state = T.tensor([state], dtype=T.float).to(self.device)
            state_ = np.concatenate(obs_)
            state_ = T.tensor([state_], dtype=T.float).to(self.device)

            model_out = self.env_model.forward(state)
            im_reward = F.mse_loss(model_out, state_).cpu().detach().numpy() * self.im_reward_multiplier
        else:
            im_reward = 0

        return im_reward

    def learn(self, memory):
        losses = {}

        if not memory.ready():
            return losses

        actor_states, states, actions, actions_taken, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()

            losses["critic"] = losses.get("critic", []) + [critic_loss.cpu().detach().numpy().tolist()]
            losses["actor"] = losses.get("actor", []) + [actor_loss.cpu().detach().numpy().tolist()]

        if self.scenario == "MADDPG_AE":
            device = self.agents[0].env_model.device

            obs = T.tensor(np.concatenate(actor_states, axis=1), dtype=T.float).to(device)
            obs_ = T.tensor(actor_new_states, dtype=T.float).to(device)
            actions_taken = T.tensor(np.concatenate(actions_taken, axis=1), dtype=T.float).to(device)

            for agent_idx, agent in enumerate(self.agents):
                predicted_obs = agent.env_model.forward(T.cat([obs, actions_taken], dim=1))
                env_model_loss = F.mse_loss(predicted_obs, obs_[agent_idx])
                agent.env_model.optimizer.zero_grad()
                env_model_loss.backward(retain_graph=True)
                agent.env_model.optimizer.step()

                losses["env_model_AE"] = losses.get("env_model_AE", []) + [env_model_loss.cpu().detach().numpy().tolist()]

        elif self.scenario == "MADDPG_AE_common":
            device = self.device

            obs = T.tensor(np.concatenate(actor_states, axis=1), dtype=T.float).to(device)
            obs_ = T.tensor(np.concatenate(actor_new_states, axis=1), dtype=T.float).to(device)
            actions_taken = T.tensor(np.concatenate(actions_taken, axis=1), dtype=T.float).to(device)

            predicted_obs = self.env_model.forward(T.cat([obs, actions_taken], dim=1))
            env_model_loss = F.mse_loss(predicted_obs, obs_)
            self.env_model.optimizer.zero_grad()
            env_model_loss.backward(retain_graph=True)
            self.env_model.optimizer.step()

            losses["env_model_AE_common"] = losses.get("env_model_AE_common", []) + [env_model_loss.cpu().detach().numpy().tolist()]

        elif self.scenario == "MADDPG_VAE":
            device = self.agents[0].env_model.device

            obs = T.tensor(np.concatenate(actor_states, axis=1), dtype=T.float).to(device)
            obs_ = T.tensor(actor_new_states, dtype=T.float).to(device)
            actions_taken = T.tensor(np.concatenate(actions_taken, axis=1), dtype=T.float).to(device)

            for agent_idx, agent in enumerate(self.agents):
                predicted_obs, kl = agent.env_model.forward(T.cat([obs, actions_taken], dim=1))
                # reconstruction_loss = agent.env_model.gaussian_likelihood(predicted_obs,
                #                                                           agent.env_model.log_scale,
                #                                                           obs_[agent_idx])

                # MSE rec loss
                reconstruction_loss = F.mse_loss(predicted_obs, obs_[agent_idx])

                # evidence lower bound (elbo loss)
                # env_model_loss = (kl - reconstruction_loss).mean()
                env_model_loss = kl + reconstruction_loss

                agent.env_model.optimizer.zero_grad()
                env_model_loss.backward(retain_graph=True)
                agent.env_model.optimizer.step()

                kl_loss_mean = kl.mean().cpu().detach().numpy().tolist()
                rec_loss_mean = reconstruction_loss.mean().cpu().detach().numpy().tolist()
                env_model_loss_mean = env_model_loss.cpu().detach().numpy().tolist()
                losses["env_model_VAE_kl"] = losses.get("env_model_VAE_kl", []) + [kl_loss_mean]
                losses["env_model_VAE_rec"] = losses.get("env_model_VAE_rec", []) + [rec_loss_mean]
                losses["env_model_VAE"] = losses.get("env_model_VAE", []) + [env_model_loss_mean]

        return losses

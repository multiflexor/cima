import argparse
import numpy as np
import random
import cv2
import os
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from smac.env import StarCraft2Env
import datetime
import time
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='CIMA training loop')
parser.add_argument('--scenario', type=str,
                    help='Available scenarios: MADDPG, MADDPG_GRID_SN, MADDPG_AE, MADDPG_AE_common, MADDPG_VAE')
parser.add_argument('--map_name', type=str,
                    help='Available maps: 2m_vs_2zg_IM, 2m_vs_10zg_IM')
parser.add_argument('--n_steps', type=int, help='Training steps')


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':
    args = parser.parse_args()
    # map_name = "2m_vs_2zg_IM"
    map_name = args.map_name

    # scenarios: MADDPG, MADDPG_GRID_SN, MADDPG_AE, MADDPG_AE_common, MADDPG_VAE
    # scenario = "MADDPG"
    scenario = args.map_name

    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    obs_shape = env_info["obs_shape"]
    n_actions = env_info["n_actions"]

    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(obs_shape)
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    maddpg_agents = MADDPG(n_agents, obs_shape, n_actions, scenario, alpha=0.001, beta=0.001,
                           checkpoint_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(100_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=2048)

    PRINT_INTERVAL = 1000
    # N_STEPS = 1_500_000
    N_STEPS = args.n_steps
    learn_every = 100
    TEST_EPISODES = 100

    MAX_STEPS = env_info["episode_limit"]
    score_history = []
    ep_len_history = []
    evaluate = False
    load_models = False
    best_score = 0

    noise_rate = 0.99
    noise_rate_min = 0.01
    noise_decay_rate = noise_rate / N_STEPS

    heat_map = None
    hm_size = 10

    time_date = datetime.datetime.now().strftime('%Y_%m_%d')
    time_day = datetime.datetime.now().strftime('%H%M%S')
    train_info_folder = "/train_info/"

    short_map_name = map_name.split("_vs_")[0] + map_name.split("_vs_")[1].split("_")[0]
    short_map_name += f"_{scenario}"

    folder_name = train_info_folder + time_date + "/" + short_map_name + "_" + time_day + "/"

    if not os.path.isdir(os.getcwd() + train_info_folder + time_date):
        os.mkdir(os.getcwd() + train_info_folder + time_date)
    if not os.path.isdir(os.getcwd() + folder_name):
        os.mkdir(os.getcwd() + folder_name)

    writer = SummaryWriter(os.getcwd() + folder_name + "logs")

    out = cv2.VideoWriter(f"{os.getcwd() + folder_name}{map_name}.mp4", -1, 100.0, (320, 320))
    save_every = 100
    start = time.time()

    map_x, map_y = 32, 32
    state_novelty = [np.zeros((map_x, map_y), dtype=np.uint32) for _ in range(n_agents)]

    if evaluate or load_models:
        maddpg_agents.load_checkpoint()

    done = True
    step = 0
    episode_step = 0
    test_episode_number = 0
    obs = None
    episode_reward = []
    episode_im_reward = [[] for i in range(n_agents)]
    prev_agents_positions = []

    baseline_2v2 = [2, 4] * 10
    baseline_2v10 = [5] * 6 + [3] * 7

    while step < N_STEPS + (TEST_EPISODES * MAX_STEPS):
        if done:
            if step >= N_STEPS:
                if evaluate:
                    test_episode_number += 1
                evaluate = True

                if test_episode_number >= TEST_EPISODES:
                    break

            env.reset()
            if heat_map is None:
                heat_map = np.zeros((env.map_x * hm_size,
                                     env.map_y * hm_size))
            obs = env.get_obs()
            done = False
            episode_step = 0
            episode_reward = []
            episode_im_reward = [[] for i in range(n_agents)]

        noise_rate = max(noise_rate_min, noise_rate - noise_decay_rate)

        # HEATMAP START
        heat_map *= 0.9998
        positions = np.array(
            [[env.agents[i].health > 0, env.agents[i].pos.x * hm_size, env.agents[i].pos.y * hm_size] for i in
             env.agents.keys()], dtype=np.uint8)
        heat_map[env.map_y * hm_size - positions[:, 2], positions[:, 1]] = 1

        heatmap_image = np.zeros((env.map_x * hm_size,
                                  env.map_y * hm_size, 3), dtype=np.uint8)
        heatmap_image[:, :, 2] = (heat_map * 255).astype(np.uint8)
        for alive, *pos in positions:
            if alive:
                cv2.circle(heatmap_image, (pos[0], env.map_y * hm_size - pos[1]), 2,
                           (255, 255, 255), -1)

        if not step % save_every:
            out.write(heatmap_image)

        cv2.imshow("Heatmap", heatmap_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # HEATMAP END

        if not evaluate:
            actions_probabilities = maddpg_agents.choose_action(obs, noise_rate)
            actions = [random.choices(np.arange(n_actions),
                                      weights=actions_probabilities[i] * env.get_avail_actions()[i])[0]
                       for i in range(n_agents)]
        else:
            actions_probabilities = maddpg_agents.choose_action(obs, 0)
            actions = [(actions_probabilities[i] * env.get_avail_actions()[i]).argmax()
                       for i in range(n_agents)]

        # BASELINE MODELLING
        # avail_attack_actions = [[x for x in range(6, len(env.get_avail_agent_actions(i)))
        #                          if env.get_avail_agent_actions(i)[x]] if env.agents[i].health > 0 else [0]
        #                         for i in env.agents.keys()]

        # baseline_2v2
        # actions = [baseline_2v2[episode_step] if not len(avail_attack_actions[i])
        #            else avail_attack_actions[i][0] for i in range(n_agents)]

        # baseline_2v10
        # actions = [baseline_2v10[episode_step] if episode_step < len(baseline_2v10)
        #            else avail_attack_actions[i][0] for i in range(n_agents)]

        try:
            reward, done, info = env.step(actions)
        except AssertionError as e:
            print(e)
            print(f"actions: {actions}")
            print(f"avail_actions: {env.get_avail_actions()}")
            continue

        obs_ = env.get_obs()

        if scenario == "MADDPG_GRID_SN":
            agents_state_novelties = []
            agents_positions = [[env.agents[i].health > 0,
                                 list(map(int, [env.agents[i].pos.x, env.agents[i].pos.y]))]
                                for i in env.agents.keys()]
            for idx, (alive, pos) in enumerate(agents_positions):
                if alive:
                    state_novelty[idx][pos[0], pos[1]] += 1

                agents_state_novelties.append([alive, state_novelty[idx][pos[0], pos[1]]])

            if not len(prev_agents_positions):
                prev_agents_positions = agents_positions.copy()

            intrinsic_rewards = []
            for agent_idx in range(n_agents):
                if prev_agents_positions[agent_idx][1] == agents_positions[agent_idx][1]:
                    intrinsic_rewards.append(0)
                else:
                    alive, agents_sn = agents_state_novelties[agent_idx]
                    sn_min, sn_max = state_novelty[agent_idx].min(), state_novelty[agent_idx].max()
                    im_reward = (1 - (agents_sn - sn_min) / sn_max) ** 2 if alive else 0
                    intrinsic_rewards.append(im_reward)
        elif scenario == "MADDPG_AE":
            intrinsic_rewards = maddpg_agents.get_intrinsic_rewards(obs, obs_, actions)
        elif scenario == "MADDPG_AE_common":
            intrinsic_rewards = maddpg_agents.get_intrinsic_rewards_common(obs, obs_, actions)
        elif scenario == "MADDPG_VAE":
            intrinsic_rewards = maddpg_agents.get_intrinsic_rewards(obs, obs_, actions)
        else:
            intrinsic_rewards = [0 for _ in range(n_agents)]

        episode_reward.append(reward)
        if scenario == "MADDPG_AE_common":
            [episode_im_reward[i].append(intrinsic_rewards) for i in range(n_agents)]
        else:
            [episode_im_reward[i].append(intrinsic_rewards[i]) for i in range(n_agents)]

        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)

        if episode_step >= MAX_STEPS:
            done = True

        memory.store_transition(obs, state, actions_probabilities, actions, reward, intrinsic_rewards, obs_, state_,
                                done)

        if step % learn_every == 0 and not evaluate:
            losses = maddpg_agents.learn(memory)
            for k, v in losses.items():
                loss_dict = {f"{k}_{idx}": x for idx, x in enumerate(v)}
                writer.add_scalars(f'Losses/{k}', loss_dict, step)

        obs = obs_

        writer.add_scalar('Stats/noise_rate', noise_rate, step)

        if done:
            sum_reward = sum(episode_reward)
            score_history.append(sum_reward)
            ep_len_history.append(episode_step)
            healths_dict = {f"agent_{i}": env.agents[i].health for i in env.agents.keys()}
            healths_dict["enemy_total"] = sum([env.enemies[i].health for i in env.enemies.keys()])

            im_rewards_dict = {f"agent_{i}": sum(episode_im_reward[i]) for i in range(n_agents)}
            writer.add_scalar('Rewards/external_reward', sum_reward, step)
            writer.add_scalars('Rewards/IM_rewards', im_rewards_dict, step)
            writer.add_scalars('Stats/end_health_points', healths_dict, step)
            writer.add_scalar('Stats/episode_length', episode_step, step)

            # if scenario == "MADDPG_GRID_SN":
            #     sn_img = np.zeros((env.map_x, env.map_y, 3), dtype=np.uint8)
            #     sn_img[:, :, 0] = (state_novelty / state_novelty.max() * 255).astype(np.uint8)
            #     sn_img = cv2.rotate(cv2.resize(sn_img, dsize=(env.map_x * hm_size, env.map_y * hm_size),
            #                                    interpolation=cv2.INTER_CUBIC), cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     writer.add_image('Heatmaps/state_novelty_grid', sn_img, step, dataformats='HWC')

            avg_score = np.mean(score_history[-20:])
            if not evaluate:
                if avg_score > best_score:
                    maddpg_agents.save_checkpoint()
                    best_score = avg_score

        if step % PRINT_INTERVAL == 0 and step > 0:
            avg_score = np.mean(score_history[-20:])
            avg_ep_len = np.mean(ep_len_history[-20:])
            print('step', step, 'noise_rate', round(noise_rate, 5),
                  'average score {:.2f}'.format(avg_score),
                  'average episode length {:.2f}'.format(avg_ep_len))

        episode_step += 1
        step += 1

    print("End:", (time.time() - start) / 60)
    out.release()
    cv2.destroyAllWindows()

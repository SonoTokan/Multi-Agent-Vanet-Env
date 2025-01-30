from collections import defaultdict
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class VANETRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for VANET. See parent class for details."""

    def __init__(self, config):
        super(VANETRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        tmp_timestep = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    self.collect(step)
                )

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if getattr(self.all_args, "use_CADP", False):
                if (total_num_steps - tmp_timestep) - 500000 > 0:
                    tmp_timestep = total_num_steps
                    self.save_timestep(total_num_steps)

                if total_num_steps > self.all_args.cadp_breakpoint:
                    self.trainer.use_cadp_loss = True

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.map_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                if getattr(self.all_args, "use_CADP", False):
                    self.policy.actor.use_att_v = True
                    self.eval(total_num_steps, "student_")
                    self.policy.actor.use_att_v = False
                    self.eval(total_num_steps, "teacher_")
                    pass
                else:
                    self.eval(total_num_steps)

    def run_eval(self):
        if getattr(self.all_args, "use_CADP", False):
            self.policy.actor.use_att_v = True
            self.eval_restore(self.episode_length, "student_")
            self.policy.actor.use_att_v = False
            self.eval_restore(self.episode_length, "teacher_")
            pass
        else:
            self.eval_restore(self.episode_length)

    def warmup(self):
        # reset env
        obs, share_obs, dones, infos = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

        # 确认active mask shape是否正确（对比前面obs）
        dones_env = np.all(dones, axis=1)

        idle_mask = np.array(
            [
                [info[agent_id]["idle"] for agent_id in range(self.num_agents)]
                for info in infos
            ]
        )

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        # active_masks[(dones == True) | (idle_mask == True)] = np.zeros(
        #     ((dones == True).sum(), 1), dtype=np.float32
        # )

        active_masks[(dones == True) | (idle_mask == True)] = 0

        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )
        # 执行这一步可能会导致问题？
        # self.buffer.active_masks[0] = active_masks.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = (
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        idle_mask = np.array(
            [
                [info[agent_id]["idle"] for agent_id in range(self.num_agents)]
                for info in infos
            ]
        )

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        # 是否正确被mask
        # active_masks[(dones == True) | (idle_mask == True)] = np.zeros(
        #     ((dones == True).sum(), 1), dtype=np.float32
        # )

        active_masks[(dones == True) | (idle_mask == True)] = 0

        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        bad_masks = np.array(
            [
                [
                    [0.0] if info[agent_id]["bad_transition"] else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

        ava_rews = (self.buffer.rewards * self.buffer.active_masks[:-1]).sum() / (
            self.buffer.active_masks[:-1].sum() + 1e-6
        )

        train_infos["availble_average_step_rewards"] = ava_rews

        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval_restore(self, total_num_steps, log_prefix="", log_interval=1024):
        if self.model_dir is not None:
            self.restore()

        eval_avg_qoe = 0
        eval_avg_ee = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        # avg for every single agent
        qoes = []
        ees = []
        hit_ratios = []

        eval_obs, eval_share_obs, dones, infos = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for time_step in range(total_num_steps):
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = (
                self.eval_envs.step(eval_actions)
            )

            # 假设只有一个env，如果需要n_eval_rollout_threads需要再调整
            # 只需计算in range car的qoe
            vehs_qoe = defaultdict(list)
            rsus_mean_qoe = defaultdict(list)
            rsus_avg_hit_ratio = {}

            for rsu in self.eval_envs.envs[0].rsus:
                for veh_id in rsu.range_connections:
                    if veh_id is not None:
                        veh = self.eval_envs.envs[0].vehicles[veh_id]
                        vehs_qoe[rsu.id].append(veh.job.qoe)
                rsus_mean_qoe[rsu.id] = np.nanmean(vehs_qoe[rsu.id])
                rsus_avg_hit_ratio[rsu.id] = np.nanmean(rsu.hit_ratios)

            ee = np.mean([float(rsu.ee) for rsu in self.eval_envs.envs[0].rsus])

            qoes.append(list(rsus_mean_qoe.values()))
            ees.append(ee)
            hit_ratios.append(list(rsus_avg_hit_ratio.values()))

            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            if time_step % log_interval == 0:
                avg_rew = np.mean(one_episode_rewards, axis=0)

                eval_env_infos = {log_prefix + "eval_average_step_rewards": avg_rew}
                self.log_env(eval_env_infos, time_step)

                avg_qoe = np.nanmean(qoes)
                avg_ee = np.mean(ees)
                avg_hit_ratio = np.nanmean(hit_ratios)

                print(
                    (
                        log_prefix
                        + "avg qoe is {}, avg ee is {}, avg caching hit ratio is {}."
                    ).format(avg_qoe, avg_ee, avg_hit_ratio)
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            log_prefix + "avg qoe": avg_qoe,
                            log_prefix + "avg ee ": avg_ee,
                            log_prefix + "avg hit ratio ": avg_hit_ratio,
                        },
                        step=time_step,
                    )
                else:
                    self.writter.add_scalars(
                        log_prefix + "avg",
                        {log_prefix + "avg": avg_qoe},
                        time_step,
                    )

    @torch.no_grad()
    def eval(self, total_num_steps, log_prefix=""):

        eval_avg_qoe = 0
        eval_avg_ee = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        qoes = []
        ees = []

        eval_obs, eval_share_obs, dones, infos = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = (
                self.eval_envs.step(eval_actions)
            )

            # qoe = np.mean([float(v.job.qoe) for v in self.env.vehicles.values()])
            # ee = np.mean([float(rsu.ee) for rsu in self.env.rsus])

            # qoes.append(qoe)
            # ees.append(ee)

            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {
                    log_prefix + "eval_average_episode_rewards": eval_episode_rewards
                }
                self.log_env(eval_env_infos, total_num_steps)

                # avg_qoe = np.mean(qoes)
                # avg_ee = np.mean(ees)

                # print((log_prefix + "avg qoe is {}, avg ee is {}.").format(avg_qoe, avg_ee))
                # if self.use_wandb:
                #     wandb.log(
                #         {log_prefix + "eval_win_rate": eval_win_rate},
                #         step=total_num_steps,
                #     )
                # else:
                #     self.writter.add_scalars(
                #         log_prefix + "eval_win_rate",
                #         {log_prefix + "eval_win_rate": eval_win_rate},
                #         total_num_steps,
                #     )
                break

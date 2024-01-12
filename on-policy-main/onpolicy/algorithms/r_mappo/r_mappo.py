import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.popart import PopArt
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.distance import D_dict
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import pickle
import copy
class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        if self.args.dynamic_clip_tag:
            self.clip_param = np.ones(self.args.n_agents)*args.clip_param
            self.clip_constant = np.log(self.clip_param + 1).sum()

            self.all_updates = int(self.args.num_env_steps) // self.args.episode_length // self.args.n_rollout_threads
            self.curr_delta_update = 0
            self.min_clip_constant = np.log(self.args.min_clip_params + 1) * self.args.n_agents
            self.curr_constant = self.clip_constant
        else:
            self.clip_param = args.clip_param




        self.value_clip_param = args.clip_param

        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta



        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None



        if self.args.penalty_method:

            self.beta_kl = np.ones(self.args.n_agents) * self.args.beta_kl

            self.dtar_kl = self.args.dtar_kl
            self.kl_para1 = self.args.kl_para1
            self.kl_para2 = self.args.kl_para2
            self.kl_lower = self.dtar_kl / self.kl_para1
            self.kl_upper = self.dtar_kl * self.kl_para1

            if self.args.inner_refine:
                self.args.dtar_sqrt_kl = np.sqrt(self.args.dtar_kl)



            self.beta_sqrt_kl = np.ones(self.args.n_agents) * self.args.beta_sqrt_kl


            self.dtar_sqrt_kl = self.args.dtar_sqrt_kl
            self.sqrt_kl_para1 = self.args.sqrt_kl_para1
            self.sqrt_kl_para2 = self.args.sqrt_kl_para2
            self.sqrt_kl_lower = self.dtar_sqrt_kl / self.sqrt_kl_para1
            self.sqrt_kl_upper = self.dtar_sqrt_kl * self.sqrt_kl_para1

            self.para_upper_bound = self.args.para_upper_bound
            self.para_lower_bound = self.args.para_lower_bound
        self.term_kl = None
        self.term_sqrt_kl = None
        self.p_loss_part1 = None
        self.p_loss_part2 = None
        self.d_coeff = None
        self.d_term = None


        self.term1_grad_norm = None
        self.term2_grad_norm = None
        if self.args.overflow_save:
            self.overflow = np.zeros(self.args.n_agents)


        self.term_dist = None



    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.value_clip_param,
                                                                                        self.value_clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.value_clip_param,
                                                                                        self.value_clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True,aga_update_tag = False,update_index = -1,curr_update_num=None):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        torch.autograd.set_detect_anomaly(True)

        base_ret,q_ret,sp_ret,penalty_ret = sample
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = base_ret




        # print('share_obs_batch = {}'.format(share_obs_batch.shape))
        # print('obs_batch = {}'.format(share_obs_batch.shape))

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        # print('active_masks = {}'.format(active_masks_batch.shape))
        # Reshape to do in a single forward pass for all steps
        # print('aga_update_tag = {}, update_index = {}'.format(aga_update_tag,update_index))
        if self.args.idv_para and aga_update_tag and self.args.aga_tag:
            # print('case1')
            values_all, action_log_probs, dist_entropy = self.policy.evaluate_actions_single(share_obs_batch,
                                                                                  obs_batch,
                                                                                  rnn_states_batch,
                                                                                  rnn_states_critic_batch,
                                                                                  actions_batch,
                                                                                  masks_batch,
                                                                                  available_actions_batch,
                                                                                  active_masks_batch,update_index=update_index)
        else:
            prob_merge_tag = not self.args.dynamic_clip_tag
            values_all, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                                  obs_batch,
                                                                                  rnn_states_batch,
                                                                                  rnn_states_critic_batch,
                                                                                  actions_batch,
                                                                                  masks_batch,
                                                                                  available_actions_batch,
                                                                                  active_masks_batch,prob_merge=prob_merge_tag)
        # actor update
        #imp_weights = (episode_length * agent_num, 1)
        # imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # print('imp_weights = {}'.format(imp_weights.shape))
        self.es_tag = False
        if self.args.penalty_method:
            eps_kl = 1e-9
            eps_sqrt = 1e-12
            if self.args.env_name == 'mujoco':
                with torch.no_grad():
                    old_dist = self.policy.get_dist(obs_batch, rnn_states_batch, masks_batch,target=True)
                new_dist = self.policy.get_dist(obs_batch, rnn_states_batch, masks_batch)
                kl = []
                for idv_old,idv_new in zip(old_dist,new_dist):
                    idv_kl = torch.distributions.kl_divergence(idv_old,idv_new)
                    kl.append(idv_kl)
                kl = torch.stack(kl,dim = 1)
                # print('n_actions = {} kl = {}, idv_kl = {}'.format(self.args.n_actions,kl.shape,idv_kl.shape))
            else:
                if self.args.env_name == 'StarCraft2':
                    probs = self.policy.get_probs(obs_batch, rnn_states_batch, masks_batch,available_actions=available_actions_batch)
                else:
                    probs = self.policy.get_probs(obs_batch, rnn_states_batch, masks_batch)


                old_probs_batch = penalty_ret[0]

                old_probs_batch = check(old_probs_batch).to(**self.tpdv)

                # print('n_actions = {} old_probs_batch = {}, probs = {}'.format(self.args.n_actions,old_probs_batch.shape,probs.shape))
                kl = old_probs_batch * (torch.log(old_probs_batch + eps_kl) - torch.log(probs + eps_kl) )
            # print('prev kl = {}'.format(kl.shape))
            # print('obs_batch = {}'.format(obs_batch.shape))
            # kl and obs_batch is with shape (episode_length * n_agents, n_actions)
            kl = torch.sum(kl,dim = -1,keepdim=True).reshape([-1,1])
            # print('after kl = {}'.format(kl.shape))
            imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
            term1 = imp_weights * adv_targ

            sqrt_kl = torch.sqrt(torch.max(kl + eps_sqrt,eps_sqrt * torch.ones_like(kl)))


            term1 = term1.reshape([-1,self.args.n_agents])
            sqrt_kl = sqrt_kl.reshape([-1,self.args.n_agents])
            kl = kl.reshape([-1,self.args.n_agents])
            policy_active_masks_batch = active_masks_batch.reshape([-1,self.args.n_agents])
            if self._use_policy_active_masks:
                term1 = (-term1 * policy_active_masks_batch).sum(dim = 0) / active_masks_batch.sum(dim = 0)
                term_sqrt_kl = (sqrt_kl * policy_active_masks_batch).sum(dim = 0) / active_masks_batch.sum(dim = 0)
                term_kl  = (kl * policy_active_masks_batch).sum(dim = 0) / active_masks_batch.sum(dim = 0)
                # term1 = (-torch.sum(term1,dim=-1,keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                # print('kl = {} sqrt_kl = {} active_masks = {}'.format(kl.shape,sqrt_kl.shape,active_masks_batch.shape))
                # term_sqrt_kl = (torch.sum(sqrt_kl,dim=-1,keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                # term_kl = (torch.sum(kl, dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                term1 = (-term1).mean(dim=0)
                term_sqrt_kl = sqrt_kl.mean(dim=0)
                term_kl = kl.mean(dim=0)
                # term1 = -torch.sum(term1, dim=-1, keepdim=True).mean()
                # term_sqrt_kl = torch.sum(sqrt_kl, dim=-1, keepdim=True).mean()
                # term_kl = torch.sum(kl, dim=-1, keepdim=True).mean()
            if self.args.idv_beta:
                self.term_sqrt_kl = term_sqrt_kl
                self.term_kl = term_kl
            else:
                self.term_sqrt_kl = torch.ones_like(term_sqrt_kl) * (term_sqrt_kl.mean())
                self.term_kl = torch.ones_like(term_kl) * (term_kl.mean())

            if self.args.dpo_policy_div_agent_num:
                term1 /= self.args.n_agents

            if self.args.dpo_check_kl_baseline:
                imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = imp_weights * adv_targ
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                if self._use_policy_active_masks:
                    policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                                     dim=-1,
                                                     keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

                policy_loss = policy_action_loss
            else:
                sqrt_coeff = torch.tensor(self.beta_sqrt_kl).to(**self.tpdv).detach()
                kl_coeff = torch.tensor(self.beta_kl).to(**self.tpdv).detach()
                policy_loss = term1 + sqrt_coeff * term_sqrt_kl + kl_coeff * term_kl
                policy_loss = policy_loss.mean()
            self.es_tag = False

            clip_rate = 0
        else:
            if self.args.dynamic_clip_tag:
                if self.args.idv_para and aga_update_tag and self.args.aga_tag:
                    imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
                    surr1 = imp_weights * adv_targ
                    surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param[update_index], 1.0 + self.clip_param[update_index]) * adv_targ
                else:
                    surr1 = []
                    surr2 = []
                    agent_adv_targ = adv_targ.reshape([-1, self.args.n_agents, 1])
                    if self.args.env_name == 'mujoco':
                        old_action_log_probs_batch = old_action_log_probs_batch.mean(axis = -1)
                    agent_old_action_log_probs = old_action_log_probs_batch.reshape([-1, self.args.n_agents, 1])
                    # print('init_agent_old_action_log_probs = {},agent_old_action_log_probs = {}'.format(old_action_log_probs_batch.shape,agent_old_action_log_probs.shape))
                    # print('old_batch = {}'.format(old_action_log_probs_batch))
                    imp_weights = []
                    for i in range(self.args.n_agents):
                        # print('action_log_probs[{}] = {},  agent_old_action_log_probs[:, {}] = {}'.format(i,action_log_probs[i].shape,i,agent_old_action_log_probs[:, i].shape ))

                        agent_imp_weights = torch.exp(action_log_probs[i] - agent_old_action_log_probs[:, i])
                        agent_surr1 = agent_imp_weights * agent_adv_targ[:, i]
                        agent_surr2 = torch.clamp(agent_imp_weights, 1.0 - self.clip_param[i],
                                                  1.0 + self.clip_param[i]) * agent_adv_targ[:, i]
                        surr1.append(agent_surr1)
                        surr2.append(agent_surr2)
                        imp_weights.append(agent_imp_weights)
                    surr1 = torch.stack(surr1, dim=1).reshape([-1, 1])
                    surr2 = torch.stack(surr2, dim=1).reshape([-1, 1])
                    imp_weights = torch.stack(imp_weights, dim=1).reshape([-1, 1])
            else:
                imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = imp_weights * adv_targ
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
            clip_check = (surr1 != surr2)
            clip_check_sum = clip_check.sum()
            clip_check_total = torch.ones_like(clip_check).sum()
            # print('clip_check = {}'.format(clip_check))
            # print('clip_check_sum = {}'.format(clip_check_sum))
            # print('clip_check_total = {}'.format(clip_check_total))
            clip_rate = float(clip_check_sum) / float(clip_check_total)
            print('clip_rate = {}'.format(clip_rate))

            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                                 dim=-1,
                                                 keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss



        auto_loss = 0


        self.actor_zero_grad(aga_update_tag,update_index)

        # print('policy_loss = {}'.format(policy_loss))
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor[update_index].parameters(), self.max_grad_norm)
                else:
                    actor_grad_norm = 0
                    for i in range(self.args.n_agents):
                        idv_actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor[i].parameters(),
                                                                   self.max_grad_norm)
                        actor_grad_norm += idv_actor_grad_norm
                    actor_grad_norm /= self.args.n_agents

            else:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    actor_grad_norm = get_gard_norm(self.policy.actor[update_index].parameters())
                else:
                    actor_grad_norm = 0
                    for i in range(self.args.n_agents):
                        idv_actor_grad_norm = get_gard_norm(self.policy.actor[i].parameters())
                        actor_grad_norm += idv_actor_grad_norm
                    actor_grad_norm /= self.args.n_agents

            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())




        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.actor_optimizer[update_index].step()
            else:
                for i in range(self.args.n_agents):
                    self.policy.actor_optimizer[i].step()
        else:
            self.policy.actor_optimizer.step()

        # critic update
        if self.args.use_q:
            act_idx = torch.from_numpy(actions_batch).to(**self.tpdv)
            act_idx = act_idx.long()
            # act_idx =
            # print('act_idx.dtype = {} act_idx = {} values = {}'.format(act_idx.dtype,act_idx.shape,values.shape))
            values = torch.gather(values_all,index = act_idx,dim = -1)
        else:
            values = values_all

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.critic_zero_grad(aga_update_tag,update_index)

        # print('value_loss = {}'.format(value_loss.dtype))

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic[update_index].parameters(),
                                                               self.max_grad_norm)
                else:
                    critic_grad_norm = 0
                    for i in range(self.args.n_agents):
                        idv_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic[i].parameters(),
                                                                       self.max_grad_norm)
                        critic_grad_norm += idv_critic_grad_norm
                    critic_grad_norm /= self.args.n_agents

            else:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            if self.args.idv_para:
                if aga_update_tag and self.args.aga_tag:
                    critic_grad_norm = get_gard_norm(self.policy.critic[update_index].parameters())
                else:
                    critic_grad_norm = 0
                    for i in range(self.args.n_agents):
                        idv_critic_grad_norm = get_gard_norm(self.policy.critic[i].parameters())
                        critic_grad_norm += idv_critic_grad_norm
                    critic_grad_norm /= self.args.n_agents

            else:
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())


        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.critic_optimizer[update_index].step()
            else:
                for i in range(self.args.n_agents):
                    self.policy.critic_optimizer[i].step()
        else:
            self.policy.critic_optimizer.step()

        # if self.args.dynamic_clip_tag and self.args.use_q and curr_update_num < self.args.clip_update_num:
        #     # all_probs = self.policy.get_probs(obs_batch,rnn_states_batch, masks_batch,available_actions_batch)
        #     old_A = old_values_batch - baseline_batch
        #     self.update_policy_clip(old_probs_batch,old_A)

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights,clip_rate,auto_loss

    def train(self, buffer, update_actor=True,time_steps=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        epoch_num = 0
        if self.args.new_period and buffer.aga_update_tag and self.args.aga_tag:
            epoch_num = self.ppo_epoch * self.args.period
        else:
            epoch_num = self.ppo_epoch





        if self.args.penalty_method and self.term_kl is not None:
            print('term_sqrt_kl = {} term_kl = {} old_beta_sqrt_kl = {}, old_beta_kl = {}'.format(self.term_sqrt_kl,
                                                                                                  self.term_kl,
                                                                                                  self.beta_sqrt_kl,
                                                                                                  self.beta_kl))
            print('prev beta_kl = {}'.format(self.beta_kl))

            if self.args.penalty_beta_type == 'adaptive':
                for i in range(self.args.n_agents):
                    if self.term_kl[i] < self.kl_lower:
                        self.beta_kl[i] /= self.kl_para2
                        # self.beta_kl = np.maximum(self.para_lower_bound,self.beta_kl)
                    elif self.term_kl[i] > self.kl_upper:
                        self.beta_kl[i] *= self.kl_para2
                    # self.beta_kl = np.minimum(self.para_upper_bound, self.beta_kl)
                print('after beta_kl = {}'.format(self.beta_kl))

            if self.args.penalty_beta_sqrt_type == 'adaptive':
                for i in range(self.args.n_agents):
                    if self.term_sqrt_kl[i] < self.sqrt_kl_lower:
                        self.beta_sqrt_kl[i] /= self.sqrt_kl_para2
                        # self.beta_sqrt_kl = np.maximum(self.para_lower_bound, self.beta_sqrt_kl)

                    elif self.term_sqrt_kl[i] > self.sqrt_kl_upper:
                        self.beta_sqrt_kl[i] *= self.sqrt_kl_para2
                    # self.beta_sqrt_kl = np.minimum(self.para_upper_bound, self.beta_sqrt_kl)

            elif self.args.penalty_beta_type == 'adaptive_rule2':
                if self.args.use_q:
                    old_pi = buffer.old_probs_all[:-1]

                    # (39,1,3,5) (episode_length,rollout_thread,agents, actions)
                    old_pi = old_pi.reshape(-1, self.args.n_actions)
                    old_q = buffer.old_values_all[:-1]
                    # print('old_q_shape = {}'.format(old_q.shape))
                    old_q = old_q.reshape(-1, self.args.n_actions)
                    old_baseline = buffer.baseline[:-1]
                    old_baseline = old_baseline.reshape(-1, 1)
                    old_A = old_q - old_baseline
                    E_A_square = old_A * old_A * old_pi
                    E_A_square = 2 * np.sqrt( np.sum(E_A_square) )
                elif self.args.sp_use_q:
                    old_pi = buffer.sp_prob_all[:-1]

                    # (39,1,3,5) (episode_length,rollout_thread,agents, actions)
                    old_pi = old_pi.reshape(-1, self.args.sp_num)
                    old_q = buffer.sp_value_all[:-1]
                    # print('old_q_shape = {}'.format(old_q.shape))
                    old_q = old_q.reshape(-1, self.args.sp_num)
                    old_baseline = np.mean(old_q, axis=-1, keepdims=True)
                    old_A = old_q - old_baseline
                    E_A_square = old_A * old_A
                    E_A_square = 2 * np.sqrt(np.mean(E_A_square))
                self.beta_sqrt_kl = 0.9 * E_A_square

                print('before_ramge_clip: new_beta_sqrt_kl = {}'.format(self.beta_sqrt_kl))
            for i in range(self.args.n_agents):
                if self.beta_kl[i] < self.para_lower_bound:
                    self.beta_kl[i] = self.para_lower_bound
                if self.beta_kl[i] > self.para_upper_bound:
                    self.beta_kl[i] = self.para_upper_bound
                if self.beta_sqrt_kl[i] < self.para_lower_bound:
                    self.beta_sqrt_kl[i] = self.para_lower_bound
                if self.beta_sqrt_kl[i] > self.para_upper_bound:
                    self.beta_sqrt_kl[i] = self.para_upper_bound

            print('after_ramge_clip: new_beta_sqrt_kl = {}'.format(self.beta_sqrt_kl))

            if self.args.no_sqrt_kl:
                for i in range(self.args.n_agents):
                    self.beta_sqrt_kl[i] = 0
            if self.args.no_kl:
                for i in range(self.args.n_agents):
                    self.beta_kl[i] = 0

            print('new_beta_sqrt_kl = {}, new_beta_kl = {}'.format(self.beta_sqrt_kl, self.beta_kl))

        if self.args.dynamic_clip_tag:
            if self.args.use_q:
                if self.args.all_state_clip:

                    value,probs = self.get_matrix_state_table()
                    old_q.append(value.detach().cpu().numpy())
                    old_pi.append(probs.detach().cpu().numpy())
                    old_q = np.array(old_q)
                    old_pi = np.array(old_pi)
                    old_baseline = np.sum(old_q * old_pi,axis = -1,keepdims= True)
                    old_A = old_q - old_baseline

                else:
                    old_pi = buffer.old_probs_all[:-1]

                    # (39,1,3,5) (episode_length,rollout_thread,agents, actions)
                    old_pi = old_pi.reshape(-1, self.args.n_actions)
                    old_q = buffer.old_values_all[:-1]
                    # print('old_q_shape = {}'.format(old_q.shape))
                    old_q = old_q.reshape(-1, self.args.n_actions)
                    old_baseline = buffer.baseline[:-1]
                    old_baseline = old_baseline.reshape(-1, 1)
                    old_A = old_q - old_baseline
            elif self.args.sp_clip:
                old_pi = buffer.sp_prob_all[:-1]

                # (39,1,3,5) (episode_length,rollout_thread,agents, actions)
                old_pi = old_pi.reshape(-1, self.args.sp_num)
                old_q = buffer.sp_value_all[:-1]
                # print('old_q_shape = {}'.format(old_q.shape))
                old_q = old_q.reshape(-1, self.args.sp_num)
                old_baseline = np.mean(old_q,axis=-1,keepdims=True)
                old_A = old_q - old_baseline

            print('all_state_clip = {}'.format(self.args.all_state_clip))
            print('old_pi_shape = {}'.format(old_pi.shape))
            print('buffer.obs = {}'.format(buffer.obs.shape))
            # (40,10,3,30) (epsiode_limit + 1, rollout_num,agents, state_dim)
            if self.args.true_rho_s:
                sample_state = buffer.obs[:-1,:,0,:] #(39,10,30)
                sample_state = np.mean(sample_state,axis = 1)
                # print('state_cnt = {}'.format(sample_state))
                episode_length = sample_state.shape[0]
                gamma_list = np.ones([episode_length,1])
                for i in range(1,episode_length):
                    gamma_list[i] = self.args.gamma * gamma_list[i - 1]
                rho_s = np.sum(gamma_list * sample_state,axis = 0)
                print('rho_s = {}'.format(rho_s))
            else:
                rho_s = None
            if self.args.dcmode == 1:
                if self.args.delta_decay:
                    decay_constant = (1 - (2*self.curr_delta_update)/self.all_updates )*self.clip_constant
                    print('self.min_clip_constant = {}, decay_constant = {}'.format(self.min_clip_constant,decay_constant))
                    self.curr_constant = np.maximum(self.min_clip_constant,  decay_constant)
                else:
                    self.curr_constant = self.clip_constant

                if self.args.delta_reset:
                    idv_delta = self.curr_constant / self.args.n_agents
                    idv_eps = np.exp(idv_delta) - 1
                    self.clip_param = np.ones(self.args.n_agents) * idv_eps
                print('clip update {}/{} init clip params = {}'.format(self.curr_delta_update,self.all_updates,self.clip_param))

                all_deltas = []
                for i in range(self.args.clip_update_num):
                    clip_iter, clip_solve_loss = self.update_policy_clip_ver_1(old_pi, old_A)
                    print('clip step {}: clip_params = {}'.format(i,self.clip_param))
                    all_deltas.append(self.clip_delta)
                all_deltas = np.array(all_deltas)
                final_delta = np.mean(all_deltas,axis = 0)
                final_eps = np.exp(final_delta) - 1
                self.clip_param = final_eps
                self.curr_delta_update += 1

            elif self.args.dcmode == 2:
                clip_iter, clip_solve_loss = self.update_policy_clip_ver_2(old_pi, old_A,rho_s)


            print('solved_clip_params = {}'.format( self.clip_param))
            if self.args.weighted_clip and time_steps is not None:
                ratios = np.minimum(float(time_steps)/float(self.args.weighted_clip_step),1.0)
                self.clip_param = ratios * self.clip_param + (1.0 - ratios) * self.args.weighted_clip_init
                print('weighted_clip_params = {}, time_steps = {} ratios = {}'.format(self.clip_param,time_steps,ratios))

        if self.args.penalty_method and self.args.env_name == 'mujoco' and self.args.correct_kl:
            self.policy.hard_update_policy()


        if self.args.use_q:
            if self._use_popart:
                advantages = self.value_normalizer.denormalize(buffer.value_curr[:-1]) - self.value_normalizer.denormalize(buffer.baseline[:-1])
            else:
                advantages = buffer.value_curr[:-1] - buffer.baseline[:-1]
        else:
            if self._use_popart:
                advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]



        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['clip_rate'] = 0
        train_info['kl_div'] = 0
        train_info['p_loss_part1'] = 0
        train_info['p_loss_part2'] = 0
        train_info['d_coeff'] = 0
        train_info['d_term'] = 0
        train_info['auto_loss'] = 0
        train_info['term1_grad_norm'] = 0
        train_info['term2_grad_norm'] = 0
        train_info['grad_ratio'] = 0




        if self.args.optim_reset and self.args.aga_tag and buffer.aga_update_tag:
            self.policy.optim_reset(buffer.update_index)
        curr_update_num = 0
        for epoch_cnt in range(epoch_num):
            self.es_tag = False
            self.es_kl = []
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights,clip_rate,auto_loss \
                    = self.ppo_update(sample, update_actor,aga_update_tag = buffer.aga_update_tag,update_index = buffer.update_index,curr_update_num = epoch_cnt)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['clip_rate'] += clip_rate
                train_info['auto_loss'] += auto_loss

                if self.term_kl is not None:
                    train_info['kl_div'] += self.term_kl.mean()
                curr_update_num += 1
                if self.args.early_stop:
                    self.es_kl = torch.mean(torch.tensor(self.es_kl))
                    self.es_tag = self.es_kl > self.args.es_judge
                    if self.es_tag:
                        print('es_kl = {}, es_judge = {}'.format(self.es_kl,self.args.es_judge))
                        print('early stop after {} epoch, total updates {}'.format(epoch_cnt,curr_update_num))
                        train_info['early_stop_epoch'] = curr_update_num
                        train_info['early_stop_kl'] = self.es_kl
                        break
            if self.es_tag:
                break
        if not self.es_tag and self.args.early_stop:
            train_info['early_stop_epoch'] = curr_update_num
            train_info['early_stop_kl'] = self.es_kl

        if self.args.target_dec:
            if self.args.soft_target:
                self.policy.soft_update_policy()
            else:
                if buffer.aga_update_tag and self.args.aga_tag:
                    self.policy.hard_update_policy()

        if self.args.sp_use_q:
            if self.args.sp_update_policy == 'hard':
                self.policy.hard_update_policy()
            else:
                self.policy.soft_update_policy()

        if self.args.use_q or self.args.sp_use_q:
            self.policy.soft_update_critic()


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if k in ['early_stop_epoch','early_stop_kl','grad_ratio']:
                continue
            train_info[k] /= num_updates
        out_info_key = [ 'min_clip', 'max_clip', 'mean_clip', 'clip_rate']
        if self.args.dynamic_clip_tag and self.args.use_q:
            train_info['clip_iteration'] = clip_iter
            train_info['clip_solve_loss'] = clip_solve_loss
            out_info_key.append('clip_iteration')
            out_info_key.append('clip_solve_loss')

        train_info['min_clip'] = np.min(self.clip_param)
        train_info['max_clip'] = np.max(self.clip_param)
        train_info['mean_clip'] = np.mean(self.clip_param)


        out_info = ''
        for key in out_info_key:
            out_info += '{}: {}    '.format(key, train_info[key])
        print(out_info)
        return train_info

    def prep_training(self):
        if self.args.idv_para:
            for i in range(self.args.n_agents):
                self.policy.actor[i].train()
                self.policy.critic[i].train()
        else:
            self.policy.actor.train()
            self.policy.critic.train()

    def prep_rollout(self):
        if self.args.idv_para:
            for i in range(self.args.n_agents):
                self.policy.actor[i].eval()
                self.policy.critic[i].eval()
        else:
            self.policy.actor.eval()
            self.policy.critic.eval()

    def actor_zero_grad(self,aga_update_tag = False,update_index = -1):
        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.actor_optimizer[update_index].zero_grad()
            else:
                for i in range(self.args.n_agents):
                    self.policy.actor_optimizer[i].zero_grad()
        else:
            self.policy.actor_optimizer.zero_grad()
    def critic_zero_grad(self,aga_update_tag = False,update_index = -1):
        if self.args.idv_para:
            if aga_update_tag and self.args.aga_tag:
                self.policy.critic_optimizer[update_index].zero_grad()
            else:
                for i in range(self.args.n_agents):
                    self.policy.critic_optimizer[i].zero_grad()
        else:
            self.policy.critic_optimizer.zero_grad()
#!/usr/bin/env python
import sys
import os
import pickle
# import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from gym.spaces import Discrete
"""Train script for SMAC."""
def get_element(array,index):
    ret = array
    # print('array_shape = {}'.format(np.shape(array)))
    # print('index = {}'.format(index))
    for x in index:
        ret = ret[x]
        # print('x = {}, ret_shape = {}'.format(x,np.shape(ret)))
    return ret
def make_not_exist_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def make_mat_game_from_file(filename,episode_longer=False):
    with open(filename,'rb') as f:
        matrix_para = pickle.load(f)
        r_mat = matrix_para['reward']

        trans_mat = matrix_para['trans_mat']
        end_state = np.zeros(np.shape(r_mat)[0])
        state_num = np.shape(r_mat)[0]
        if episode_longer:
            max_episode_length = int(state_num *1.34)
        else:
            max_episode_length = state_num
        init_state = 0
        env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length,evaluate_mat=True)
        return env
class MatrixGame:
    def __init__(self,r_mat,trans_mat,init_state,end_state,max_episode_length,evaluate_mat=False):
        r_shape = np.shape(r_mat)
        self.r_mat = r_mat
        self.trans_mat = trans_mat
        self.state_num = r_shape[0]

        self.agent_num = len(r_shape) - 1
        self.action_num = r_shape[1]
        self.share_observation_space = []
        self.observation_space = []
        self.action_space = []
        for i in range(self.agent_num):
            self.action_space.append(Discrete(self.action_num ))
            self.observation_space.append(self.state_num)
            self.share_observation_space.append(self.state_num)
        self.init_state = init_state
        self.now_state  = init_state
        self.step_count = 0
        self.end_state = end_state
        self.max_episode_length = max_episode_length
        self.state_action_count = np.zeros_like(r_mat).reshape([self.state_num,-1])
        self.long_step_count = 0
        self.evaluate_mat = evaluate_mat
        self.traverse = 0
        self.abs_traverse = 0
        self.relative_traverse = 0
    def eval_traverse(self):
        # print('state_action_count = {}'.format(self.state_action_count))
        print('long_step_count = {}'.format(self.long_step_count))
        covered_count = (self.state_action_count > 0).sum()
        all_state_action = self.state_action_count.shape[0] * self.state_action_count.shape[1]
        traverse = covered_count / all_state_action
        max_traverse = min(self.long_step_count / all_state_action, 1)
        relative_traverse = covered_count / self.long_step_count
        print('abs_traverse = {} max_traverse = {} relative_traverse = {}'.format(traverse, max_traverse,
                                                                                  relative_traverse))
        freq_mat = self.state_action_count / self.long_step_count
        freq_mat = freq_mat.reshape(self.r_mat.shape)
        static_return = (freq_mat * self.r_mat).sum()
        print('static_return = {}'.format(static_return))
        self.traverse = traverse
        self.abs_traverse = max_traverse
        self.relative_traverse = relative_traverse
        return self.traverse, self.abs_traverse,self.relative_traverse
    def reset(self):
        self.now_state = self.init_state
        self.step_count = 0
        obs = self.get_obs()
        obs = np.expand_dims(obs, axis=0)
        share_obs = obs
        available_actions = self.get_avail_agent_actions_all()
        available_actions = np.expand_dims(available_actions, axis=0)
        return obs,share_obs,available_actions
    def reset_evaluate(self):
        self.long_step_count = 0
        self.state_action_count = np.zeros_like(self.state_action_count)
    def get_obs(self):
        obs = []
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        for i in range(self.agent_num):
            obs.append(state)
        return obs
    def get_ac_idx(self,action):
        idx = 0
        for a in action:
            idx = self.action_num * idx + a
            # print('idx = {} a = {}'.format(idx,a))
        return idx
    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state
    def step(self,action,evaluate=False):
        # print('step = {} action  = {}'.format(self.step_count))
        sa_index = []
        sa_index.append(self.now_state)
        # sa_index += action
        action = np.array(action).reshape(-1)
        for a in action:
            sa_index.append(a)
        # print('sa_index = {},action = {}'.format(sa_index, action))
        if not evaluate:
            ac_idx = self.get_ac_idx(action)
            self.state_action_count[self.now_state,ac_idx] += 1
            self.long_step_count += 1

        r = get_element(self.r_mat,sa_index)
        next_s_prob = get_element(self.trans_mat,sa_index)
        # print('sa_index = {} next_s_prob = {}'.format(sa_index,next_s_prob))
        next_state = np.random.choice(range(self.state_num),size = 1, p = next_s_prob)[0]
        self.now_state = next_state
        self.step_count += 1

        done = self.end_state[self.now_state]
        if self.step_count >= self.max_episode_length:
            done = 1
        done_ret = np.ones([1,self.agent_num],dtype=bool) *done
        reward_ret = np.ones([1,self.agent_num,1]) * r
        obs = self.get_obs()
        obs = np.expand_dims(obs,axis = 0)
        share_obs = obs
        available_actions = self.get_avail_agent_actions_all()
        available_actions = np.expand_dims(available_actions,axis = 0)
        info = [[{} for i in range(self.agent_num)]]
        return obs,share_obs,reward_ret,done_ret,info,available_actions
    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.action_num
        env_info["n_agents"] = self.agent_num
        env_info["state_shape"] = self.state_num
        env_info["obs_shape"] = self.state_num
        env_info["episode_limit"] = self.max_episode_length
        return env_info
    def get_avail_agent_actions_all(self):
        return np.ones([self.agent_num,self.action_num])
    def get_avail_agent_actions(self,agent_id):
        return np.ones(self.action_num)
    def get_model_info(self,state,action):
        sa_index = []
        sa_index.append(state)
        action = np.array(action)
        # print('action = {}'.format(action))
        for a in action:
            sa_index.append(a)
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('action = {} sa_index = {} self.trans_mat = {} next_s_prob = {}'.format(action,sa_index,self.trans_mat.shape, next_s_prob.shape  ))
        return r,next_s_prob
    def close(self):
        return

def parse_args(args, parser):

    parser.add_argument('--map_name', type=str, default='random_matrix_game_3',
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]
    all_args.matrix_test = True
    all_args.env_name = 'matrix_game'
    return all_args


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results")
    name_list = [all_args.env_name,all_args.map_name,all_args.algorithm_name,all_args.group_name,all_args.experiment_name]
    for name in name_list:
        run_dir /= name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_mat_game_from_file('a_mat_model/{}.pkl'.format(all_args.map_name),all_args.episode_longer)
    eval_envs =  make_mat_game_from_file('a_mat_model/{}.pkl'.format(all_args.map_name),all_args.episode_longer) if all_args.use_eval else None
    env_info = envs.get_env_info()
    print('env_info = {}'.format(env_info))
    num_agents = env_info["n_agents"]
    all_args.n_agents = num_agents
    all_args.num_agents = num_agents
    all_args.n_actions = env_info["n_actions"]
    all_args.obs_shape = env_info["obs_shape"]
    all_args.shape_shape = env_info["state_shape"]
    all_args.num_actions = env_info["n_actions"]
    all_args.episode_length = env_info['episode_limit']

    if all_args.check_optimal_V_bound:
        with open('a_mat_model/{}_optimal_V_table.pkl'.format(all_args.map_name),'rb') as f:
            optimal_V_table = pickle.load(f)
        with open('a_mat_model/{}_optimal_pi_table.pkl'.format(all_args.map_name),'rb') as f:
            optimal_pi_table = pickle.load(f)
        all_args.optimal_V_table = optimal_V_table
        all_args.optimal_pi_table = optimal_pi_table
        all_args.calc_V_envs = envs

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from onpolicy.runner.shared.matrix_runner import MatrixRunner as Runner


    runner = Runner(config)
    with open(str(runner.log_dir) + '/arg.txt','w') as f:
        f.write('{}'.format(config))
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

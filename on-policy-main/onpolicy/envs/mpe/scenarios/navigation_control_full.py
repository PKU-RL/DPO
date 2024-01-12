import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment

class Scenario(BaseScenario):
    def make_world(self,args):
        world = World()
        # set any world properties first
        world.dim_c = 2

        self.collision_penal = args.collision_penal
        self.vision = args.vision
        num_agents = args.num_agents
        num_landmarks = args.num_landmarks
        self.n_agents = args.num_agents

        print('self.n_agents = {}'.format(self.n_agents))
        self.rewards = np.zeros(self.n_agents)
        self.state_buff = []
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)

        world.dists = []
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.dists = []

        world.steps = 0

    def benchmark_data(self, agent, world):
        if agent.name == 'agent 0':
            entity_info = []
            state_shape = 0
            for l in world.landmarks:
                entity_info.append(l.state.p_pos)
                state_shape += len(l.state.p_pos)
            agent_info = []
            for a in world.landmarks:
                agent_info.append(a.state.p_pos)
                agent_info.append(a.state.p_vel)
                state_shape += (len(a.state.p_pos) + len(a.state.p_pos))
            state = np.concatenate(entity_info+agent_info)
            ret = {}
            ret['state_shape'] =state_shape
            ret['s'] = state
        return  ret

    def is_obs(self,entity1,entity2):
        delt_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delt_pos)))
        return True if dist < self.vision else False
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        if agent.name == 'agent 0':
            rew = 0
            world.dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
                                    for a in world.agents])
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.min_dists = self._bipartite_min_dists(world.dists)
            # the reward is normalized by the number of agents
            rew = -np.mean(self.min_dists)


            collision_rew = 0
            for b in world.agents:
                for a in world.agents:
                    if self.is_collision(a, b):
                        collision_rew -= self.collision_penal
            collision_rew /= (2 * self.n_agents)
            rew += collision_rew

            rew = np.clip(rew, -15, 15)
            self.rewards = np.full(self.n_agents, rew)
            world.min_dists = self.min_dists
        # print('reward = {} , return = {}'.format(self.rewards,self.rewards.mean()))
        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def done(self, agent, world):
        condition1 = world.steps >= world.max_steps_episode
        return condition1

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            # if self.is_obs(agent,entity):
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            # else:
            #     entity_pos.append(np.zeros_like(entity.state.p_pos - agent.state.p_pos))
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            # if self.is_obs(agent,other):
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # else:
            #     other_pos.append(np.zeros_like(other.state.p_pos - agent.state.p_pos))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

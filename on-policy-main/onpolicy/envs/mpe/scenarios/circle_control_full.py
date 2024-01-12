import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None]*len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas


def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle<0:
        angle += 2*np.pi
    return angle


class Scenario(BaseScenario):
    def make_world(self,args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.collision_penal = args.collision_penal
        self.vision = args.vision
        num_agents = args.num_agents
        num_landmarks = 1
        self.n_agents = args.num_agents


        self.arena_size = 1

        self.rewards = np.zeros(self.n_agents)

        self.target_radius = 0.5  # fixing the target radius for now
        self.ideal_theta_separation = (2 * np.pi) / self.n_agents  # ideal theta difference between two agents
        self.dist_thres = 0.05
        self.theta_thres = 0.1


        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
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

        world.landmarks[0].state.p_pos = np.random.uniform(-0.25, +0.25, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)


        world.steps = 0
        world.dists = []



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

            landmark_pose = world.landmarks[0].state.p_pos
            relative_poses = [agent.state.p_pos - landmark_pose for agent in world.agents]
            thetas = get_thetas(relative_poses)

            theta_min = min(thetas)
            self.expected_positions = [landmark_pose + self.target_radius * np.array(
                [np.cos(theta_min + i * self.ideal_theta_separation),
                 np.sin(theta_min + i * self.ideal_theta_separation)])
                              for i in range(self.n_agents)]


            world.dists = np.array([[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_positions]
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

        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if self.is_obs(agent,entity):
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(np.zeros_like(entity.state.p_pos - agent.state.p_pos))
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
            if self.is_obs(agent,other):
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append(np.zeros_like(other.state.p_pos - agent.state.p_pos))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def done(self, agent, world):
        condition1 = world.steps >= world.max_steps_episode
        return condition1

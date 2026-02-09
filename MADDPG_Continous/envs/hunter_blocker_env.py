import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agents_dynamics import CustomWorld


class Custom_raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_hunters=2,
        num_blockers=1,
        num_targets=1,
        num_obstacles=0,
        max_cycles=50,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        world_size=2.5,
        capture_distance=0.3,
        capture_steps=5,
        hunter_speed=1.2,
        blocker_speed=1.0,
        target_speed=1.0,
        hunter_view_range=0.6,
        blocker_view_range=1.0,
        target_view_range=0.8,
    ):
        EzPickle.__init__(
            self,
            num_hunters=num_hunters,
            num_blockers=num_blockers,
            num_targets=num_targets,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario(
            capture_distance=capture_distance,
            capture_steps=capture_steps,
        )
        world = scenario.make_world(
            num_hunters,
            num_blockers,
            num_targets,
            num_obstacles,
            world_size=world_size,
            hunter_speed=hunter_speed,
            blocker_speed=blocker_speed,
            target_speed=target_speed,
            hunter_view_range=hunter_view_range,
            blocker_view_range=blocker_view_range,
            target_view_range=target_view_range,
        )
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.world_size = world_size
        self.metadata["name"] = "hunter_blocker_env"

        self.max_force = 1.0
        self.capture_threshold = capture_distance
        self.capture_steps = capture_steps

        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                if self.continuous_actions:
                    space_dim = self.world.dim_p
                else:
                    space_dim = self.world.dim_p * 2 + 1
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c
            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim

            if self.continuous_actions:
                self.action_spaces[agent.name] = gymnasium.spaces.Box(
                    low=-1.0, high=1.0, shape=(space_dim,), dtype=np.float32
                )
            else:
                self.action_spaces[agent.name] = gymnasium.spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = gymnasium.spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = gymnasium.spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            mdim = self.world.dim_p if self.continuous_actions else self.world.dim_p * 2 + 1
            if agent.movable:
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name], time=None)

        self.world.step()
        self.scenario.update_capture(self.world)

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

        if self.world.capture:
            for agent_name in self.terminations:
                self.terminations[agent_name] = True

    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                agent.action.u[0] = action[0][0]
                agent.action.u[1] = action[0][1]
            else:
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0

        agent.action.u = np.clip(agent.action.u, -self.max_force, self.max_force)


class Scenario(BaseScenario):
    def __init__(self, capture_distance=0.3, capture_steps=5):
        self.capture_distance = capture_distance
        self.capture_steps = capture_steps

    def make_world(
        self,
        num_hunters=2,
        num_blockers=1,
        num_targets=1,
        num_obstacles=0,
        world_size=2.5,
        hunter_speed=1.2,
        blocker_speed=1.0,
        target_speed=1.0,
        hunter_view_range=0.6,
        blocker_view_range=1.0,
        target_view_range=0.8,
    ):
        world = CustomWorld()
        world.world_size = world_size
        world.dim_c = 0
        world.dim_p = 2
        world.dt = 0.1
        world.damping = 0.2
        world.capture = False
        world.capture_counter = 0
        world.capture_distance = self.capture_distance
        world.capture_steps = self.capture_steps

        num_agents = num_hunters + num_blockers + num_targets
        world.agents = [Agent() for _ in range(num_agents)]

        for i, agent in enumerate(world.agents):
            if i < num_hunters:
                role = "hunter"
                agent.role = role
                agent.adversary = True
                agent.max_speed = hunter_speed
                agent.view_range = hunter_view_range
                agent.name = f"hunter_{i}"
                agent.size = world_size * 0.06
                agent.initial_mass = 1.2
            elif i < num_hunters + num_blockers:
                role = "blocker"
                agent.role = role
                agent.adversary = True
                agent.max_speed = blocker_speed
                agent.view_range = blocker_view_range
                agent.name = f"blocker_{i - num_hunters}"
                agent.size = world_size * 0.08
                agent.initial_mass = 1.4
            else:
                role = "target"
                agent.role = role
                agent.adversary = False
                agent.max_speed = target_speed
                agent.view_range = target_view_range
                agent.name = f"target_{i - num_hunters - num_blockers}"
                agent.size = world_size * 0.05
                agent.initial_mass = 0.9

            agent.collide = True
            agent.silent = True
            agent.accel = None

        world.landmarks = [Landmark() for _ in range(num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = world_size * 0.08
            landmark.boundary = False

        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            if agent.role == "target":
                agent.color = np.array([0.35, 0.35, 0.85])
            elif agent.role == "hunter":
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = np.array([0.85, 0.65, 0.35])
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(
                -world.world_size * 0.9,
                +world.world_size * 0.9,
                world.dim_p,
            )
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(
                    -world.world_size * 0.8,
                    +world.world_size * 0.8,
                    world.dim_p,
                )
                landmark.state.p_vel = np.zeros(world.dim_p)

        world.capture = False
        world.capture_counter = 0

    def update_capture(self, world):
        if world.capture:
            return
        targets = self.targets(world)
        hunters = self.hunters(world)
        if not targets or not hunters:
            world.capture = False
            world.capture_counter = 0
            return
        target = targets[0]
        min_dist = np.inf
        for hunter in hunters:
            dist = np.linalg.norm(target.state.p_pos - hunter.state.p_pos)
            min_dist = min(min_dist, dist)
        if min_dist <= world.capture_distance:
            world.capture_counter += 1
        else:
            world.capture_counter = 0
        if world.capture_counter >= world.capture_steps:
            world.capture = True

    def hunters(self, world):
        return [agent for agent in world.agents if agent.role == "hunter"]

    def blockers(self, world):
        return [agent for agent in world.agents if agent.role == "blocker"]

    def trackers(self, world):
        return [agent for agent in world.agents if agent.role in {"hunter", "blocker"}]

    def targets(self, world):
        return [agent for agent in world.agents if agent.role == "target"]

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        if agent.role == "target":
            return self.target_reward(agent, world)
        if agent.role == "hunter":
            return self.hunter_reward(agent, world)
        return self.blocker_reward(agent, world)

    def global_reward(self, world):
        return 0.0

    def target_reward(self, agent, world):
        trackers = self.trackers(world)
        if not trackers:
            return 0.0
        distances = [
            np.linalg.norm(agent.state.p_pos - tracker.state.p_pos)
            for tracker in trackers
        ]
        min_dist = min(distances)
        reward = min_dist / world.world_size
        if world.capture:
            reward -= 10.0
        reward -= self.boundary_penalty(agent, world)
        return reward

    def hunter_reward(self, agent, world):
        targets = self.targets(world)
        if not targets:
            return 0.0
        target = targets[0]
        dist = np.linalg.norm(target.state.p_pos - agent.state.p_pos)
        reward = -dist / world.world_size
        if world.capture:
            reward += 10.0
        reward -= self.boundary_penalty(agent, world)
        return reward

    def blocker_reward(self, agent, world):
        targets = self.targets(world)
        if not targets:
            return 0.0
        target = targets[0]
        dist = np.linalg.norm(target.state.p_pos - agent.state.p_pos)
        reward = -0.5 * dist / world.world_size
        if world.capture:
            reward += 5.0
        reward -= self.boundary_penalty(agent, world)
        return reward

    def boundary_penalty(self, agent, world):
        def bound(x):
            boundary_start = world.world_size * 0.96
            full_boundary = world.world_size
            if x < boundary_start:
                return 0
            if x < full_boundary:
                return (x - boundary_start) * 10
            return min(np.exp(2 * x - 2 * full_boundary), 10)

        penalty = 0.0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            penalty += bound(x)
        return penalty

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                relative_entity_pos = (entity.state.p_pos - agent.state.p_pos) / world.world_size
                entity_pos.append(relative_entity_pos)

        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            dist = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            if dist <= agent.view_range:
                relative_pos = (other.state.p_pos - agent.state.p_pos) / world.world_size
                other_pos.append(relative_pos)
                norm_vel = other.state.p_vel / other.max_speed
                other_vel.append(norm_vel)
            else:
                other_pos.append(np.zeros(world.dim_p))
                other_vel.append(np.zeros(world.dim_p))

        norm_self_vel = agent.state.p_vel / max(agent.max_speed, 1e-6)
        norm_self_pos = agent.state.p_pos / world.world_size
        return np.concatenate(
            [norm_self_vel]
            + [norm_self_pos]
            + entity_pos
            + other_pos
            + other_vel
        )


env = make_env(Custom_raw_env)
parallel_env = parallel_wrapper_fn(env)

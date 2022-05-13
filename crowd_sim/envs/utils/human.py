from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.type = 'human'

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        full_state = self.get_full_state()
        state = JointState(full_state, ob)
        action = self.policy.predict(state)
        return action

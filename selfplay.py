from stable_baselines3 import A2C


class SelfAgent:
    def __init__(self, directory):
        self.model = A2C.load(directory)

    def get_action(self, state):
        action, _states = self.model.predict(state)
        return action

class Reward:
    def calculate_reward(self):
        raise NotImplementedError

class LengthReward(Reward):
    def calculate_reward(self, curr_coverage, prev_coverage):
        terminate = False
        if curr_coverage > prev_coverage:
            reward = (curr_coverage - prev_coverage) * 20
            if curr_coverage > 0.9:
                reward = 5
            if curr_coverage > 0.95:
                terminate = True
        else:
            reward = -1
        return reward, terminate
        
            
def get_reward_and_termination(curr_coverage, prev_coverage, reward_type='right_length_reward:-1'):
    reward_calculator = LengthReward()
    reward, terminate = reward_calculator.calculate_reward(curr_coverage, prev_coverage)
    return reward, terminate
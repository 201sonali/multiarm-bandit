import numpy as np
import argparse

bandit_arms = 10
steps = 10000
runs = 300
e = 0.1
a = 0.1

# average rewards and ratio of optimal actions for sample average (sa) and constant step size (css)
rewards_sa = np.zeros(steps)
actions_sa = np.zeros((runs, steps))
rewards_css = np.zeros(steps)
actions_css = np.zeros(steps)

for run in range(runs):
    # Separate true action values (q*) for each method
    true_action_values_sa = np.zeros(bandit_arms)
    true_action_values_css = np.zeros(bandit_arms)

    action_counts_sa = np.zeros(bandit_arms)
    action_counts_css = np.zeros(bandit_arms)
    q_values_sa = np.zeros(bandit_arms)
    q_values_css = np.zeros(bandit_arms)

    for step in range(steps):
        # true action for q*(a)~n(0,0.01)
        true_action_values_sa += np.random.normal(0, 0.01, bandit_arms)
        true_action_values_css += np.random.normal(0, 0.01, bandit_arms)

        # if e-greedy is true (10%) --> random
        if np.random.rand() < e:
            action_sa = np.random.choice(bandit_arms)
            action_css = np.random.choice(bandit_arms)
        # if e-greedy is not true (90%) --> go with the max
        else:
            action_sa = np.argmax(q_values_sa)
            action_css = np.argmax(q_values_css)

        # what reward you get
        reward_sa = np.random.normal(true_action_values_sa[action_sa], 1)
        reward_css = np.random.normal(true_action_values_css[action_css], 1)

        # update for sample average
        if action_sa == np.argmax(true_action_values_sa):
            actions_sa[run, step] = 1
        action_counts_sa[action_sa] += 1
        q_values_sa[action_sa] += (1 / action_counts_sa[action_sa]) * (reward_sa - q_values_sa[action_sa])
        rewards_sa[step] += reward_sa

        # update for constant step size
        if action_css == np.argmax(true_action_values_css):
            actions_css[step] += 1
        action_counts_css[action_css] += 1
        q_values_css[action_css] += a * (reward_css - q_values_css[action_css])
        rewards_css[step] += reward_css

avg_actions_sa = np.mean(actions_sa, axis=0)
avg_actions_css = actions_css / runs
rewards_sa /= runs
rewards_css /= runs

# Initialize actions to 1 at time = 0
avg_actions_sa[0] = 1
avg_actions_css[0] = 1

# output results
parser = argparse.ArgumentParser()
parser.add_argument("output_filename")
args = parser.parse_args()

data_array = np.vstack((rewards_sa, avg_actions_sa, rewards_css, avg_actions_css))
np.savetxt(args.output_filename, data_array)
from agent.dqn_agent import DQNAgent


def test_agent_updates_q_table_after_learning() -> None:
    agent = DQNAgent(action_size=6, seed=2)
    observation = (1, 1, 1, 1, 1, 1, 1, 1)
    next_observation = (1, 1, 2, 1, 1, 1, 1, 1)

    before = agent.q_table[observation][2]
    loss = agent.learn(observation, 2, 1.5, next_observation, False)
    after = agent.q_table[observation][2]

    assert after != before
    assert loss >= 0.0


def test_agent_act_returns_valid_action() -> None:
    agent = DQNAgent(action_size=8, seed=5)
    action = agent.act((0, 0, 0, 0, 0, 0, 0, 0))
    assert 0 <= action < 8

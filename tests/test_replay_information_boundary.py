import pytest

from mint_scout.search.replay_env import ReplayEnvironment


def test_replay_blocks_unrevealed_results():
    env = ReplayEnvironment(hidden_results={("PH", 0.1): {"score": 0.2}, ("EIC", 1.0): {"score": 0.9}})
    with pytest.raises(PermissionError):
        env.get_revealed("EIC", 1.0)

    assert env.reveal("PH", 0.1) == {"score": 0.2}
    assert env.get_revealed("PH", 0.1) == {"score": 0.2}
    assert env.available_keys() == (("PH", 0.1),)

"""Tests for the training module."""

import pytest
import torch

from sixmax.network import create_policy_value_network
from sixmax.training import (
    MetricsTracker,
    PPO,
    ParallelSelfPlay,
    RolloutBuffer,
    RunningMeanStd,
    SelfPlayCollector,
    SmallConfig,
    Trainer,
    TrainingConfig,
    explained_variance,
    format_number,
    format_time,
    get_exponential_schedule,
    get_linear_schedule,
    set_seed,
)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.learning_rate == 3e-4
        assert config.clip_epsilon == 0.2
        assert config.value_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.n_epochs == 4
        assert config.batch_size == 4096
        assert config.max_grad_norm == 0.5
        assert config.n_games == 16384
        assert config.device == "auto"

    def test_small_config(self):
        """Test SmallConfig for Mac/CPU testing."""
        config = SmallConfig()

        assert config.n_games == 64
        assert config.batch_size == 128
        assert config.device == "cpu"
        assert config.total_hands == 20000

    def test_config_to_dict(self):
        """Test config serialization."""
        config = TrainingConfig(learning_rate=1e-3, seed=123)
        d = config.to_dict()

        assert d["learning_rate"] == 1e-3
        assert d["seed"] == 123

    def test_config_from_dict(self):
        """Test config deserialization."""
        d = {"learning_rate": 5e-4, "batch_size": 2048}
        config = TrainingConfig.from_dict(d)

        assert config.learning_rate == 5e-4
        assert config.batch_size == 2048
        # Default values preserved
        assert config.clip_epsilon == 0.2


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_empty_buffer(self):
        """Test empty buffer behavior."""
        buffer = RolloutBuffer()
        assert len(buffer) == 0

        # Should handle compute on empty buffer
        buffer.compute_returns_and_advantages(torch.zeros(1))
        assert buffer.advantages is not None
        assert len(buffer.advantages) == 0

    def test_add_transitions(self):
        """Test adding transitions to buffer."""
        buffer = RolloutBuffer()

        state = {"test": torch.randn(10)}
        action = torch.tensor(0)
        legal_mask = torch.ones(10, dtype=torch.bool)

        for _ in range(5):
            buffer.add(
                state=state,
                action=action,
                reward=1.0,
                value=torch.tensor(0.5),
                log_prob=torch.tensor(-1.0),
                done=False,
                legal_mask=legal_mask,
            )

        assert len(buffer) == 5

    def test_gae_computation(self):
        """Test GAE computation."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        state = {"test": torch.randn(10)}
        action = torch.tensor(0)
        legal_mask = torch.ones(10, dtype=torch.bool)

        # Add some transitions
        for i in range(10):
            buffer.add(
                state=state,
                action=action,
                reward=1.0,
                value=torch.tensor(0.5),
                log_prob=torch.tensor(-1.0),
                done=(i == 9),
                legal_mask=legal_mask,
            )

        buffer.compute_returns_and_advantages(torch.zeros(1))

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 10
        assert len(buffer.returns) == 10

    def test_get_batches(self):
        """Test mini-batch generation."""
        buffer = RolloutBuffer()

        state = {"test": torch.randn(10)}
        action = torch.tensor(0)
        legal_mask = torch.ones(10, dtype=torch.bool)

        for i in range(20):
            buffer.add(
                state=state,
                action=action,
                reward=1.0,
                value=torch.tensor(0.5),
                log_prob=torch.tensor(-1.0),
                done=(i == 19),
                legal_mask=legal_mask,
            )

        buffer.compute_returns_and_advantages(torch.zeros(1))

        batches = list(buffer.get_batches(batch_size=8))
        assert len(batches) == 3  # 20 / 8 = 2.5, rounds up to 3

        # Check batch structure
        batch = batches[0]
        assert "states" in batch
        assert "actions" in batch
        assert "advantages" in batch
        assert "returns" in batch

    def test_reset(self):
        """Test buffer reset."""
        buffer = RolloutBuffer()

        state = {"test": torch.randn(10)}
        buffer.add(
            state=state,
            action=torch.tensor(0),
            reward=1.0,
            value=torch.tensor(0.5),
            log_prob=torch.tensor(-1.0),
            done=False,
            legal_mask=torch.ones(10, dtype=torch.bool),
        )

        assert len(buffer) == 1

        buffer.reset()
        assert len(buffer) == 0


class TestPPO:
    """Tests for PPO algorithm."""

    @pytest.fixture
    def ppo(self):
        """Create PPO instance."""
        config = SmallConfig()
        network = create_policy_value_network()
        return PPO(network, config)

    def test_ppo_init(self, ppo):
        """Test PPO initialization."""
        assert ppo.network is not None
        assert ppo.optimizer is not None
        assert ppo.config is not None

    def test_get_action(self, ppo):
        """Test action selection."""
        from sixmax.encoding import StateBatchBuilder
        from sixmax.engine import PokerGame

        game = PokerGame()
        game.reset_hand()

        builder = StateBatchBuilder()
        state_dict = game.get_state_for_player(game.get_current_player())
        state = builder.build_from_dict(state_dict)
        state = {k: v.unsqueeze(0) for k, v in state.items()}

        legal_mask = torch.tensor(game.get_legal_actions(), dtype=torch.bool).unsqueeze(0)

        action, log_prob, value = ppo.get_action(state, legal_mask)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)

        # Action should be legal
        assert legal_mask[0, action.item()].item()

    def test_get_action_deterministic(self, ppo):
        """Test deterministic action selection."""
        from sixmax.encoding import StateBatchBuilder
        from sixmax.engine import PokerGame

        game = PokerGame()
        game.seed(42)
        game.reset_hand()

        builder = StateBatchBuilder()
        state_dict = game.get_state_for_player(game.get_current_player())
        state = builder.build_from_dict(state_dict)
        state = {k: v.unsqueeze(0) for k, v in state.items()}

        legal_mask = torch.tensor(game.get_legal_actions(), dtype=torch.bool).unsqueeze(0)

        # Multiple deterministic calls should return same action
        action1, _, _ = ppo.get_action(state, legal_mask, deterministic=True)
        action2, _, _ = ppo.get_action(state, legal_mask, deterministic=True)

        assert action1.item() == action2.item()


class TestSelfPlayCollector:
    """Tests for SelfPlayCollector."""

    @pytest.fixture
    def collector(self):
        """Create collector instance."""
        network = create_policy_value_network()
        return SelfPlayCollector(
            network=network,
            n_games=4,
            device="cpu",
            seed=42,
        )

    def test_collector_init(self, collector):
        """Test collector initialization."""
        assert len(collector.games) == 4
        assert collector.buffer is not None

    def test_collect_rollout(self, collector):
        """Test collecting rollout data."""
        buffer = collector.collect_rollout(n_hands=10)

        assert len(buffer) > 0
        assert buffer.advantages is not None
        assert buffer.returns is not None

    def test_collect_steps(self, collector):
        """Test collecting fixed steps."""
        buffer = collector.collect_steps(n_steps=20)

        assert len(buffer) == 20


class TestParallelSelfPlay:
    """Tests for ParallelSelfPlay."""

    def test_parallel_init(self):
        """Test parallel self-play initialization."""
        network = create_policy_value_network()
        parallel = ParallelSelfPlay(
            network=network,
            n_games=32,
            games_per_collector=16,
            device="cpu",
            seed=42,
        )

        assert len(parallel.collectors) == 2
        assert parallel.total_games == 32

    def test_parallel_collect(self):
        """Test parallel collection."""
        network = create_policy_value_network()
        parallel = ParallelSelfPlay(
            network=network,
            n_games=8,
            games_per_collector=4,
            device="cpu",
            seed=42,
        )

        buffer = parallel.collect(n_hands_per_collector=5)

        assert len(buffer) > 0


class TestTrainer:
    """Tests for Trainer."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        config = SmallConfig()
        trainer = Trainer(config)

        assert trainer.network is not None
        assert trainer.ppo is not None
        assert trainer.self_play is not None
        assert trainer.total_hands == 0

    def test_trainer_short_training(self):
        """Test a very short training run."""
        config = SmallConfig()
        config.total_hands = 50
        config.n_hands_per_update = 10
        config.log_interval = 25
        config.save_interval = 100

        trainer = Trainer(config)
        metrics = trainer.train()

        assert trainer.total_hands >= 50
        assert metrics["total_hands"] >= 50
        assert "final_policy_loss" in metrics

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint save and load."""
        config = SmallConfig()
        config.checkpoint_dir = str(tmp_path)
        config.total_hands = 20
        config.n_hands_per_update = 10
        config.save_interval = 10

        trainer = Trainer(config)
        trainer.train()

        # Save checkpoint
        path = trainer.save_checkpoint()

        # Create new trainer and load
        trainer2 = Trainer(config)
        trainer2.load_checkpoint(path)

        assert trainer2.total_hands == trainer.total_hands


class TestUtils:
    """Tests for utility functions."""

    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        a = torch.randn(5)

        set_seed(42)
        b = torch.randn(5)

        assert torch.allclose(a, b)

    def test_linear_schedule(self):
        """Test linear schedule."""
        schedule = get_linear_schedule(1.0, 0.1, 100)

        assert schedule(0) == pytest.approx(1.0)
        assert schedule(50) == pytest.approx(0.55, rel=0.01)
        assert schedule(100) == pytest.approx(0.1)
        assert schedule(150) == pytest.approx(0.1)  # Should clamp

    def test_exponential_schedule(self):
        """Test exponential schedule."""
        schedule = get_exponential_schedule(1.0, 0.01, 0.9)

        assert schedule(0) == 1.0
        assert schedule(10) == pytest.approx(0.9**10, rel=0.01)
        assert schedule(1000) >= 0.01  # Should not go below min

    def test_explained_variance(self):
        """Test explained variance computation."""
        # Perfect prediction
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert explained_variance(y, y) == pytest.approx(1.0)

        # Random prediction
        y_pred = torch.randn(100)
        y_true = torch.randn(100)
        ev = explained_variance(y_pred, y_true)
        assert -10 < ev < 1

    def test_format_time(self):
        """Test time formatting."""
        assert format_time(30) == "30.0s"
        assert format_time(90) == "1.5m"
        assert format_time(7200) == "2.0h"

    def test_format_number(self):
        """Test number formatting."""
        assert format_number(500) == "500"
        assert format_number(5000) == "5.0K"
        assert format_number(5000000) == "5.0M"


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_tracker(self):
        """Test metrics tracking."""
        tracker = MetricsTracker(window_size=5)

        for i in range(10):
            tracker.update({"loss": float(i)})

        # Should only keep last 5
        assert tracker.get_mean("loss") == pytest.approx(7.0)  # (5+6+7+8+9)/5

    def test_tracker_get_all_means(self):
        """Test getting all means."""
        tracker = MetricsTracker()
        tracker.update({"a": 1.0, "b": 2.0})
        tracker.update({"a": 3.0, "b": 4.0})

        means = tracker.get_all_means()
        assert means["a"] == 2.0
        assert means["b"] == 3.0

    def test_tracker_reset(self):
        """Test tracker reset."""
        tracker = MetricsTracker()
        tracker.update({"loss": 1.0})
        tracker.reset()

        assert tracker.get_mean("loss") == 0.0


class TestRunningMeanStd:
    """Tests for RunningMeanStd."""

    def test_running_stats(self):
        """Test running statistics."""
        import numpy as np

        rms = RunningMeanStd()

        # Update with batches
        for _ in range(100):
            x = np.random.randn(32)
            rms.update(x)

        # Should converge to standard normal
        assert abs(rms.mean) < 0.3
        assert abs(rms.var - 1.0) < 0.3

    def test_normalize(self):
        """Test normalization."""
        import numpy as np

        rms = RunningMeanStd()

        # Update with known distribution
        for _ in range(100):
            x = np.random.randn(32) * 2 + 5  # mean=5, std=2
            rms.update(x)

        # Normalized values should be roughly standard normal
        test = np.array([5.0, 7.0, 3.0])
        normalized = rms.normalize(test)

        assert abs(normalized[0]) < 0.5  # 5 is mean, should be ~0
        assert normalized[1] > 0  # 7 > mean
        assert normalized[2] < 0  # 3 < mean

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import integrated_pipeline
from integrated_pipeline import AdvancedJanusTrainer, JanusConfig
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable

class DummyEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.intrinsic_calculator = integrated_pipeline.IntrinsicRewardCalculator()
        self.training_controller = integrated_pipeline.AdaptiveTrainingController()
        self.observation_encoder = MagicMock()
        self.current_state = SimpleNamespace()
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=2)

class DummyTrainer:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.optimizer = SimpleNamespace(param_groups=[{'lr':0.1}])
        self.entropy_coef = 0.01
    def train(self, *a, **k):
        pass

class TestAdvancedPipeline:
    def test_create_environment_uses_enhanced_env(self):
        config = JanusConfig(training_mode='advanced', use_curriculum=False, total_timesteps=1, track_emergence=False, wandb_project=None)
        trainer = AdvancedJanusTrainer(config)
        trainer.grammar = ProgressiveGrammar()
        trainer.variables = [Variable('x',0,{})]
        data = np.zeros((1,1))
        with patch.object(integrated_pipeline, 'EnhancedSymbolicDiscoveryEnv', DummyEnv):
            env = trainer.create_environment(data)
        assert isinstance(env, DummyEnv)
        assert hasattr(env, 'intrinsic_calculator')
        assert hasattr(env, 'training_controller')

    def test_adaptive_controller_modifies_trainer(self):
        config = JanusConfig(training_mode='advanced', use_curriculum=False, total_timesteps=1, track_emergence=False, wandb_project=None)
        trainer = AdvancedJanusTrainer(config)
        trainer.grammar = ProgressiveGrammar()
        trainer.variables = [Variable('x',0,{})]
        data = np.zeros((1,1))
        with patch.object(integrated_pipeline, 'EnhancedSymbolicDiscoveryEnv', DummyEnv):
            env = trainer.create_environment(data)
            with patch('integrated_pipeline.HypothesisNet'), \
                 patch('hypothesis_policy_network.PPOTrainer', DummyTrainer):
                trainer.env = env
                tr = trainer.create_trainer()
                params = env.training_controller.adapt_parameters('stagnation', {'discovery_rate':0.0})
                assert 'learning_rate' in params
                assert 'entropy_coeff' in params

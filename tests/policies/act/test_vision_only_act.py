import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


@pytest.mark.parametrize("use_vae", [False, True])
def test_act_trains_with_visual_observation_only(use_vae):
    config = ACTConfig(
        input_features={
            "observation.images.forward": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.images.wrist_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        pretrained_backbone_weights=None,
        chunk_size=4,
        n_action_steps=4,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=use_vae,
    )
    policy = ACTPolicy(config).train()
    batch = {
        "observation.images.forward": torch.rand(2, 3, 32, 32),
        "observation.images.wrist_right": torch.rand(2, 3, 32, 32),
        "action": torch.rand(2, 4, 3),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    loss, metrics = policy(batch)

    assert torch.isfinite(loss)
    assert "l1_loss" in metrics
    assert ("kld_loss" in metrics) is use_vae

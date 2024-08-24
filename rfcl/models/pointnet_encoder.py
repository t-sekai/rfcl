"""PointNetEncoder class"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Union
from .types import NetworkConfig

@dataclass
class PointNetEncoderArchConfig:
    use_stn: bool = True
    activation: Union[Callable, str] = "relu"


@dataclass
class PointNetEncoderConfig(NetworkConfig):
    type = "pointnet_encoder"
    arch_cfg: PointNetEncoderArchConfig

def default_init(scale: Optional[float] = np.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class STN3d(nn.Module):
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    final_ortho_scale: float = np.sqrt(2)

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Conv(64, (1,), kernel_init=default_init())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = nn.Conv(128, (1,), kernel_init=default_init())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = nn.Conv(1024, (1,), kernel_init=default_init())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = jnp.max(x, axis=-2, keepdims=False)

        x = nn.Dense(512, kernel_init=default_init())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = self.fc2 = nn.Dense(256, kernel_init=default_init())(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = nn.Dense(9, kernel_init=default_init(self.final_ortho_scale))(x)

        iden = jnp.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=jnp.float32)

        x = x + iden
        x = x.reshape((3, 3))

        return x


class PointNetEncoder(nn.Module):
    use_stn: bool
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    stn: STN3d = STN3d(activation)

    @nn.compact
    def __call__(self, x, training: bool = True):
        if self.use_stn:
            R = self.stn(x, training)
            x = R @ x

        x = nn.Conv(64, (1,))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = nn.Conv(128, (1,))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)

        x = nn.Conv(1024, (1,))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        x = jnp.max(x, -2, keepdims=False)

        return x
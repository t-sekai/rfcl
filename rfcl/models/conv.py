"""Conv class"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from .types import NetworkConfig

@dataclass
class ConvArchConfig:
    features: List[int]
    strides: List[int]
    activation: Union[Callable, str] = "relu"
    padding: str = "VALID"
    pixel_preprocess: bool = True
    name: str = None


@dataclass
class ConvConfig(NetworkConfig):
    type = "conv"
    arch_cfg: ConvArchConfig


def default_init(scale: Optional[float] = np.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class Conv(nn.Module):
    """
    Parameters
    ----------
    features - hidden units in each layer

    strides - strides for each convolution

    activation - internal activation

    padding - padding mode, default is VALID

    pixel_preprocess - true to divide input by 255
    """

    features: Sequence[int]
    strides: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    padding: str = "VALID"
    pixel_preprocess: bool = True

    @nn.compact
    def __call__(self, x):
        assert len(self.features) == len(self.strides)
        if self.pixel_preprocess:
            x = x.astype(jnp.float32) / 255.0
        for feat, stride in zip(self.features, self.strides):
            x = nn.Conv(feat,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = self.activation(x)

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x
    
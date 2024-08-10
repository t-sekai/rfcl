import gymnasium as gym
from gymnasium.spaces import Box
from typing import Union
import torch
import numpy as np

class SparseRewardWrapper(gym.Wrapper):
    def step(self, action):
        o, _, terminated, truncated, info = self.env.step(action)
        return o, int(info["success"]), terminated, truncated, info


class ContinuousTaskWrapper(gym.Wrapper):
    """
    Makes a task continuous by disabling any early terminations, allowing episode to only end
    when truncated=True (timelimit reached)
    """

    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        terminated = False
        return observation, reward, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Adds additional info. Anything that goes in the stats wrapper is logged to tensorboard/wandb under train_stats and test_stats
    """

    def reset(self, *, seed=None, options=None):
        self.eps_seed = seed
        obs, info = super().reset(seed=seed, options=options)
        self.eps_ret = 0
        self.eps_len = 0
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.eps_ret += reward
        self.eps_len += 1
        info["eps_ret"] = self.eps_ret
        info["eps_len"] = self.eps_len
        info["seed"] = self.eps_seed
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            success_at_end=info["success"],
            success=self.success_once,
        )
        return observation, reward, terminated, truncated, info

class ClipActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs): # Torch GPU variation, taken from the gymnasium clip action wrapper
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipAction
        >>> env = gym.make("Hopper-v4")
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (3,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([5.0, -2.0, 0.0]))
        ... # Executes the action np.array([1.0, -1.0, 0]) in the base environment
    """
    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        self.action_space_low = torch.from_numpy(self.action_space.low).cuda()
        self.action_space_high = torch.from_numpy(self.action_space.high).cuda()

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return torch.clip(action, self.action_space_low, self.action_space_high)
    
class RescaleActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs): # Torch GPU variation, taken from the gymnasium rescale action wrapper
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleAction
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4")
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1,1,1]))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 0.75])
        >>> wrapped_env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.alltrue(obs == wrapped_env_obs)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, torch.tensor],
        max_action: Union[float, int, torch.tensor],
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        gym.utils.RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )
        gym.ActionWrapper.__init__(self, env)

        self.action_space_low = torch.from_numpy(self.action_space.low).cuda()
        self.action_space_high = torch.from_numpy(self.action_space.high).cuda()

        dtype_mapping = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int16): torch.int16,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.uint8): torch.uint8,
            np.dtype(np.bool_): torch.bool
        }

        self.min_action = (
            torch.zeros(env.action_space.shape, dtype=dtype_mapping[env.action_space.dtype]) + min_action
        ).cuda()
        self.max_action = (
            torch.zeros(env.action_space.shape, dtype=dtype_mapping[env.action_space.dtype]) + max_action
        ).cuda()

        self.action_space = Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        assert torch.all(torch.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert torch.all(torch.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.action_space_low
        high = self.action_space_high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = torch.clip(action, low, high)
        return action
import gymnasium as gym


try:
    import mani_skill.envs  # NOQA
    from rfcl.envs.wrappers.common import RecordEpisodeWrapper
    from rfcl.logger import Logger
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
except ImportError:
    pass


def is_mani_skill3_env(env_id: str):
    try:
        import mani_skill.envs  # NOQA
    except ImportError:
        return False
    from mani_skill.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS


def env_factory(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[], record_episode_kwargs=dict(), render_size: int = 512, logger: Logger = None):
    def _init():
        env = gym.make(env_id, disable_env_checker=True, sensor_configs=dict(width=render_size, height=render_size), **env_kwargs)
        env = CPUGymWrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        
        if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
            env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name=f"trajectory_{idx}",
                save_video=record_episode_kwargs["save_video"],
                save_trajectory=record_episode_kwargs["save_trajectory"],
                info_on_video=record_episode_kwargs["info_on_video"],
                logger=logger,
            )
        return env

    return _init

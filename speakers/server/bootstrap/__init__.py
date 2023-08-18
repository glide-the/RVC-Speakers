from speakers.server.bootstrap.base import Bootstrap
from speakers.server.bootstrap.runner_bootstrap import RunnerBootstrapBaseWeb
from speakers.server.bootstrap.bootstrap_register import bootstrap_register

__all__ = [
    "Bootstrap",
    "RunnerBootstrapBaseWeb",
    "load_bootstrap",
    "get_bootstrap"
]

bootstrap_cache = {}


def load_bootstrap(config: dict = None):

    def _build_task_from_cfg(cfg):
        return (
            bootstrap_register.get_bootstrap_class(cfg.name).from_config(cfg)
            if cfg is not None
            else Bootstrap()
        )
    for bootstraps in config:
        for key, bootstrap_cfg in bootstraps.items():  # 使用 .items() 方法获取键值对
            bootstrap = _build_task_from_cfg(bootstrap_cfg)
            bootstrap_cache[key] = bootstrap


def get_bootstrap(key: str) -> RunnerBootstrapBaseWeb:
    if not bootstrap_cache.get(key):
        raise ValueError(f'Could not find bootstrap_cache for: "{key}". '
                         f'Choose from the following: %s' % ','.join(bootstrap_cache))

    return bootstrap_cache[key]

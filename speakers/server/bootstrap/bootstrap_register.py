from speakers.server.bootstrap import Bootstrap


class BootstrapRegister:
    """
    注册管理器
    """
    mapping = {
        "bootstrap": {},
    }

    @classmethod
    def register_bootstrap(cls, name):
        r"""Register system bootstrap to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        print(f"register_bootstrap {name}")

        def wrap(task_cls):
            from speakers.server.bootstrap.base import Bootstrap
            assert issubclass(
                task_cls, Bootstrap
            ), "All tasks must inherit bootstrap class"
            if name in cls.mapping["bootstrap"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["bootstrap"][name]
                    )
                )
            cls.mapping["bootstrap"][name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def get_bootstrap_class(cls, name):
        return cls.mapping["bootstrap"].get(name, None)

    @classmethod
    def list_bootstrap(cls):
        return sorted(cls.mapping["bootstrap"].keys())


bootstrap_register = BootstrapRegister()


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


def get_bootstrap(key: str) -> Bootstrap:
    if not bootstrap_cache.get(key):
        raise ValueError(f'Could not find bootstrap_cache for: "{key}". '
                         f'Choose from the following: %s' % ','.join(bootstrap_cache))

    return bootstrap_cache[key]

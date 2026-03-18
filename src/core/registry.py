from typing import Any, List

class Registry:
    """
    Component registry.
    Allows you to register models and datasets from different modules
    and create them uniformly from configs
    """
    def __init__(self):
        self._items = dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a class"""
        def decorator(cls):
            if name in self._items:
                raise KeyError(f"{name} already registered")
            self._items[name] = cls
            return cls
        return decorator

    def create(self, name: str, **kwargs):
        if name not in self._items:
            available = ", ".join(self._items.keys())
            raise KeyError(f"{name} not registered, available are: {available}")
        return self._items[name](**kwargs)

    def list(self) -> list[str]:
        """List all registered classes"""
        return list(self._items.keys())

# Global registres
MODEL_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
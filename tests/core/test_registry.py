import pytest

from src.core.registry import MODEL_REGISTRY, Registry


class TestRegistry:
    """Tests Registry."""

    def test_register_and_create(self):
        """Register the class and create an instance."""
        reg = Registry()

        @reg.register("my_class")
        class MyClass:
            def __init__(self, x=1):
                self.x = x

        obj = reg.create("my_class", x=42)
        assert obj.x == 42

    def test_create_default_params(self):
        """Create with default parameters."""
        reg = Registry()

        @reg.register("my_class")
        class MyClass:
            def __init__(self, x=10):
                self.x = x

        obj = reg.create("my_class")
        assert obj.x == 10

    def test_list(self):
        """List of registered names."""
        reg = Registry()

        @reg.register("model_a")
        class A:
            pass

        @reg.register("model_b")
        class B:
            pass

        assert set(reg.list()) == {"model_a", "model_b"}

    def test_empty_list(self):
        """Empty registry -> empty list."""
        reg = Registry()
        assert reg.list() == []

    def test_duplicate_raises(self):
        """Re-registering with the same name -> error."""
        reg = Registry()

        @reg.register("dup")
        class A:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @reg.register("dup")
            class B:
                pass

    def test_unknown_raises(self):
        """Creating a non-existent -> error with tooltip."""
        reg = Registry()

        @reg.register("existing")
        class A:
            pass

        with pytest.raises(KeyError, match="not registered"):
            reg.create("nonexistent")

    def test_global_model_registry_exists(self):
        """Global MODEL_REGISTRY is available."""
        assert isinstance(MODEL_REGISTRY, Registry)

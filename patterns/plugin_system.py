"""Plugin System"""

import importlib
import importlib.util
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional
import inspect
import os


class Plugin(ABC):
    """Base plugin interface."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the plugin name."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Return the plugin version."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""
        pass


class PluginManager:
    """Simple plugin manager."""

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[callable]] = {}

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin."""
        plugin.initialize()
        self._plugins[plugin.get_name()] = plugin
        print(f"Registered plugin: {plugin.get_name()} v{plugin.get_version()}")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())

    def register_hook(self, hook_name: str, callback: callable) -> None:
        """Register a callback for a hook."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)

    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all callbacks registered for a hook."""
        results = []
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Hook {hook_name} failed: {e}")
        return results

    def load_plugins_from_directory(self, directory: str) -> None:
        """Load plugins from a directory."""
        if not os.path.exists(directory):
            return

        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                file_path = os.path.join(directory, filename)
                self._load_plugin_from_file(file_path, module_name)

    def _load_plugin_from_file(self, file_path: str, module_name: str) -> None:
        """Load a plugin from a file."""
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find plugin classes in the module
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, Plugin) and
                        obj is not Plugin):
                        plugin_instance = obj()
                        self.register_plugin(plugin_instance)

        except Exception as e:
            print(f"Failed to load plugin from {file_path}: {e}")


# Decorator for plugin registration
def plugin_hook(hook_name: str):
    """Decorator to register a function as a plugin hook."""
    def decorator(func):
        # Store hook info in function metadata
        func._plugin_hook = hook_name
        return func
    return decorator


# Example plugins
class LoggerPlugin(Plugin):
    """Example logging plugin."""

    def get_name(self) -> str:
        return "logger"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        # Register hooks
        manager.register_hook("before_request", self.log_request)
        manager.register_hook("after_request", self.log_response)

    def log_request(self, request_data: dict) -> None:
        print(f"[LOGGER] Request: {request_data}")

    def log_response(self, response_data: dict) -> None:
        print(f"[LOGGER] Response: {response_data}")


class CachePlugin(Plugin):
    """Example caching plugin."""

    def __init__(self):
        self._cache = {}

    def get_name(self) -> str:
        return "cache"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        manager.register_hook("before_request", self.check_cache)
        manager.register_hook("after_request", self.store_cache)

    def check_cache(self, request_data: dict) -> Optional[dict]:
        cache_key = str(request_data)
        if cache_key in self._cache:
            print(f"[CACHE] Cache hit for {cache_key}")
            return self._cache[cache_key]
        return None

    def store_cache(self, response_data: dict, request_data: dict = None) -> None:
        if request_data:
            cache_key = str(request_data)
            self._cache[cache_key] = response_data
            print(f"[CACHE] Stored response for {cache_key}")


class ValidationPlugin(Plugin):
    """Example validation plugin."""

    def get_name(self) -> str:
        return "validator"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self) -> None:
        manager.register_hook("validate_request", self.validate)

    def validate(self, request_data: dict) -> bool:
        # Simple validation example
        required_fields = ["action", "data"]
        for field in required_fields:
            if field not in request_data:
                print(f"[VALIDATOR] Missing required field: {field}")
                return False
        print(f"[VALIDATOR] Request validation passed")
        return True


# Application with plugin support
class PluggableApplication:
    """Example application that supports plugins."""

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager

    def process_request(self, request_data: dict) -> dict:
        """Process a request with plugin hooks."""

        # Validation hook
        validation_results = self.plugin_manager.call_hook("validate_request", request_data)
        if validation_results and not all(validation_results):
            return {"error": "Validation failed"}

        # Before request hook
        self.plugin_manager.call_hook("before_request", request_data)

        # Check cache
        cache_results = self.plugin_manager.call_hook("before_request", request_data)
        for result in cache_results:
            if result:  # Cache hit
                return result

        # Process request (simplified)
        response_data = {
            "status": "success",
            "action": request_data.get("action", "unknown"),
            "result": f"Processed {request_data.get('data', 'no data')}"
        }

        # After request hook
        self.plugin_manager.call_hook("after_request", response_data, request_data)

        return response_data


# Global plugin manager instance
manager = PluginManager()


def demonstrate_plugin_system():
    """Demonstrate the plugin system."""

    print("=== Plugin System Demo ===")

    # Register plugins
    logger_plugin = LoggerPlugin()
    cache_plugin = CachePlugin()
    validator_plugin = ValidationPlugin()

    manager.register_plugin(logger_plugin)
    manager.register_plugin(cache_plugin)
    manager.register_plugin(validator_plugin)

    print(f"\nRegistered plugins: {manager.list_plugins()}")

    # Create application
    app = PluggableApplication(manager)

    print("\n=== Processing Requests ===")

    # Valid request
    request1 = {"action": "create_user", "data": {"name": "Alice"}}
    response1 = app.process_request(request1)
    print(f"Response 1: {response1}")

    # Same request (should hit cache)
    print("\n--- Same request again (cache test) ---")
    response2 = app.process_request(request1)
    print(f"Response 2: {response2}")

    # Invalid request
    print("\n--- Invalid request ---")
    invalid_request = {"action": "create_user"}  # Missing 'data' field
    response3 = app.process_request(invalid_request)
    print(f"Response 3: {response3}")

    print("\n=== Plugin Information ===")
    for plugin_name in manager.list_plugins():
        plugin = manager.get_plugin(plugin_name)
        print(f"Plugin: {plugin.get_name()} v{plugin.get_version()}")


if __name__ == "__main__":
    demonstrate_plugin_system()
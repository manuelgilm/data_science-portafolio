import re

class Service:
    def __init__(self, name: str, url: str, paths: list = None):
        self.name = name
        self.url = url
        self.paths = paths if paths is not None else []
        self.status = "inactive"
        self.metadata = {}

    def activate(self):
        self.status = "active"

    def deactivate(self):
        self.status = "inactive"

    def update_metadata(self, key: str, value: str):
        self.metadata[key] = value

    def get_metadata(self, key: str):
        return self.metadata.get(key, None)

    def add_path(self, path: str):
        if path not in self.paths and self.is_valid_path(path):
            self.paths.append(path)

    def remove_path(self, path: str):
        if path in self.paths:
            self.paths.remove(path)

    def is_valid_path(self, path: str) -> bool:
        # Example validation: path should be a non-empty string and start with a '/'
        return bool(re.match(r'^\/[a-zA-Z0-9_\-\/]*$', path))
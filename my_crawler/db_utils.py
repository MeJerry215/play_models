import shelve
from threading import Lock
from constant import DB_FILE


class DBUtils:
    def __init__(self) -> None:
        self.shelf = shelve.open(DB_FILE, 'c', writeback=True)
        self.lock = Lock()

    def __getitem__(self, key):
        with self.lock:
            return self.shelf[key]

    def __setitem__(self, key, value):
        with self.lock:
            self.shelf[key] = value

    def __delitem__(self, key):
        with self.lock:
            del self.shelf[key]

    def keys(self):
        with self.lock:
            return list(self.shelf.keys())

    def exists(self, key):
        return key in self.shelf
        # return self.shelf.get(key) is not None

    def close(self):
        self.shelf.close()

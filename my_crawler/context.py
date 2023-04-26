from db_utils import DBUtils
from typing import Optional, Dict
from pixivpy3 import AppPixivAPI

class Context:
    _instance = None
    db_utils: Optional[DBUtils] = None
    config: Optional[Dict] = None
    api: Optional[AppPixivAPI] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Context, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass



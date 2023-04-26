from typing import Any
from pixivpy3 import AppPixivAPI, PixivError
from context import Context
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import logging
from datetime import datetime, timedelta
import os


def add_days(date_str: str, days: int) -> str:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj + timedelta(days=days)
    return new_date.strftime("%Y-%m-%d")

class Tag:
    def __init__(self, name, translated_name=None) -> None:
        self.name = name
        self.translated_name = translated_name

class User:
    def __init__(self, id, name, account) -> None:
        self.id = id
        self.name = name
        self.account = account

class RecordEntry:
    def __init__(self, result: bool, task) -> None:
        self.result = result
        self.name = task.name.replace(".jpg", "")
        self.url = task.url
        self.tags = [ Tag(tag.name, tag.translated_name) for tag in task.info.tags]
        self.user = User(task.info.user.id, task.info.user.name, task.info.user.account)

class DownloadStatus:
    def __init__(self, result: bool, task) -> None:
        self.result = result
        self.task = task

    def convert_record_entry(self):
        return RecordEntry(self.result, self.task)


class DownloadTask:
    def __init__(self, url, directory, name, info, filters=[], callbacks=[]) -> None:
        self.url = url
        self.directory = directory
        self.name = name
        self.info = info
        self.filters = filters
        self.callbacks = callbacks

    def __call__(self) -> DownloadStatus:
        result = False
        for filter in self.filters:
            if filter(self.info):
                return DownloadStatus(True, self)
        with Context() as ctx:
            if ctx.api is None or ctx.config is None:
                raise NotImplementedError("ctx api or cfg undefined")
            for i in range(ctx.config['worker_config']['retry_time']):
                try:
                    if not ctx.api.download(self.url, path=self.directory, fname=self.name):
                        logging.warn(
                            f"download {self.url} file {self.name} exists.")
                    else:
                        result = True
                    break
                except Exception as e:
                    logging.warn(
                        f"download {self.url} file {self.name} fails, try retry time {i + 1}: reson {str(e)}.")
        status = DownloadStatus(result, self)
        for callback in self.callbacks:
            callback(status)
        return status


class ThreadPoolDownloader:
    def __init__(self) -> None:
        with Context() as ctx:
            if ctx.config is None:
                raise NotImplementedError("ctx config undefined")
            num_workers = ctx.config['worker_config']['num_worker']
            num_workers = num_workers if num_workers != -1 else cpu_count()
            self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)

    def submit(self, task):
        return self.thread_pool.submit(task)


class IllustRankDownloader(ThreadPoolDownloader):
    def __init__(self, api: AppPixivAPI) -> None:
        super().__init__()
        self.api = api
        with Context() as ctx:
            if ctx.config is None:
                raise NotImplementedError("ctx config undefined")
            illust_rank_config = ctx.config['illust_ranking']
        self.start_date = illust_rank_config['start_date']
        self.current_date = self.start_date
        self.end_date = illust_rank_config['end_date']
        self.mode = illust_rank_config['mode']
        self.pic_req_num = illust_rank_config['pic_req_num']
        self.directory = illust_rank_config['directory']
        self.resolution = illust_rank_config['resolution']
        self.pic_rec_num = 0
        self.next_qs = None
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)

    def query(self):
        if self.current_date > self.end_date:
            return None
        if self.next_qs is None:
            json_result = self.api.illust_ranking(
                mode=self.mode, date=self.current_date)
        else:
            json_result = self.api.illust_ranking(**self.next_qs)
        self.next_qs = self.api.parse_qs(json_result.next_url)
        if self.next_qs is None:
            self.current_date = add_days(self.current_date, 1)
        tasks = []
        for illust in json_result.illusts:
            for i in range(len(illust.meta_pages)):
                name, directory, url = str(
                    illust.id) + f"_{i}.jpg", self.directory, illust.meta_pages[i].image_urls.get(self.resolution)
                task = DownloadTask(url, directory, name, illust)
                tasks.append(task)
        return tasks

    def submit(self, task):
        return super().submit(task)

import json
from constant import *
from pixivpy3 import AppPixivAPI, PixivError
import logging
from auth import login, refresh
import pdb
import argparse
from context import Context
from db_utils import DBUtils
from downloader import IllustRankDownloader

def save_token_config(api: AppPixivAPI, config: dict):
    config["user_config"]["access_token"] = api.access_token
    config["user_config"]["refresh_token"] = api.refresh_token
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


def main(args):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    # https://requests.readthedocs.io/en/latest/api/#requests.Session.get
    api = AppPixivAPI(timeout=config['worker_config']['time_out'])
    with Context() as ctx:
        db_utils = DBUtils()
        ctx.db_utils = db_utils
        ctx.config = config
        ctx.api = api

    # first try auth, if failes, it will prompt and auth again
    try:
        api.auth(refresh_token=config["user_config"]["refresh_token"])
    except PixivError as e:
        logging.info("error refresh code, try logging")
        _, refresh_token, _ = login()
        api.auth(refresh_token=refresh_token)
    finally:
        save_token_config(api, config)

    downloader = IllustRankDownloader(api)
    # main crawl, but during the crawl, the token may changes so the only thing is try save the final token
    try:
        while True:
            tasks = downloader.query()
            if tasks is None:
                break
            futures = [ downloader.submit(task) for task in tasks]
            results = [ future.result() for future in futures]
            print(results)
            pdb.set_trace()
    except Exception as e:
        logging.info("exception occur:", str(e))
    finally:
        save_token_config(api, config)
        db_utils.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(
            'example.log', 'a', 'utf-8'), logging.StreamHandler()]
    )
    parser = argparse.ArgumentParser(description='description')
    # parser.add_argument('-a', '--arg', type=int, help='help message')
    args = parser.parse_args()
    main(args)

import json
from constant import *
import time
from pixivpy3 import AppPixivAPI, PixivError
import logging
from auth import login, refresh
import pdb
import argparse


def main(args):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    print(config)
    # lock = threading.Lock()
    api = AppPixivAPI()
    # t = threading.Thread(target=refresh_code, args=(config, lock))
    # t.daemon = True
    # t.start()
    try:
        api.auth(refresh_token=config["user_config"]["refresh_token"])
    except PixivError as e:
        logging.info("error refresh code, try logging")
        _, refresh_token, _ = login()
        api.auth(refresh_token=refresh_token)
    finally:
        pdb.set_trace()
        config["user_config"]["access_token"] = api.access_token
        config["user_config"]["refresh_token"] = api.refresh_token
        # config["user_config"]["expires_in"] = expires_in
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler('example.log', 'a', 'utf-8'), logging.StreamHandler()]
    )
    parser = argparse.ArgumentParser(description='description')
    # parser.add_argument('-a', '--arg', type=int, help='help message')
    args = parser.parse_args()
    main(args)

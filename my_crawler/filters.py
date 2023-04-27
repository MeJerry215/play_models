import os
import logging

def file_exists_filter(task) -> bool:
    if os.path.exists(os.path.join(task.directory, task.name)):
        logging.info(f"{task.name} exists, skip downloading")
        return True
    else:
        return False

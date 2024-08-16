from typing import Any, List, Set, Tuple, Callable
import time, re, random, os, sys
from typing import Callable, Any
import asyncio
import string
import quart

from config import MODEL_NAME

color_p = re.compile("\\$(.+?)<([\\s\\S]*?)>")


def color(s):
    return color_p.sub("\033[\\1m\\2\033[0m", s)


def prxxx(*args, q: bool = False, from_debug=False, **kwargs):
    if q:
        return
        pass
    if from_debug:
        return
        pass
    print(
        color(time.strftime("RWKV [$33<%Y-%m-%d %H:%M:%S>] ", time.localtime())),
        *[color(a) for a in args],
        **{k:color(v) for (k, v) in kwargs.items()},
    )


prxxx()


def log_call(func):
    def nfunc(*args, **kwargs):
        prxxx(f"$33<Call> $34<{func.__name__}>({args}, {kwargs})", from_debug=True)
        return func(*args, **kwargs)

    return nfunc


def use_async_lock(func):
    lock = asyncio.locks.Lock()

    async def nfunc(*args, **kwargs):
        async with lock:
            return await func(*args, **kwargs)

    return nfunc


def run_in_async_thread(func):
    if sys.version_info.minor < 9:

        async def nfunc(*args, **kwargs):
            return func(*args, **kwargs)
        return nfunc

    async def nfunc(*args, **kwargs):
        thread = asyncio.to_thread(func, *args, **kwargs)
        return await thread
    return nfunc


symbols = "[!@#$%^&*+[\\]{};:/<>?\\|`~]"


def clean_symbols(s):
    return "".join([c for c in s if c not in symbols])


def gen_echo():
    return "".join(random.sample(string.ascii_lowercase + string.digits, 4))


def check_dir(path):
    if path is None:
        return False
    if not os.path.isdir(path):
        os.makedirs(path)


def check_file(path):
    if path is None:
        return False
    return os.path.isfile(path)


def rm_file(path):
    if path is None:
        return False
    return os.remove(path)


@run_in_async_thread
def check_dir_async(path):
    if path is None:
        return False
    if not os.path.isdir(path):
        os.makedirs(path)


@run_in_async_thread
def check_file_async(path):
    if path is None:
        return False
    return os.path.isfile(path)


@run_in_async_thread
def rm_file_async(path):
    if path is None:
        return False
    return
    return os.remove(path)

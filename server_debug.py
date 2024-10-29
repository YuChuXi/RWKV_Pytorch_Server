import json
import torch
import traceback
from typing import Callable, List
from utils import prxxx
from server_pipeline import RWKVEmbryo

def exe_head(func:Callable) -> Callable:
    def nfunc(*args, **kwargs):
        func.__name__ + ": " + func(*args, **kwargs)
    return nfunc

class RWKVDebuger:
    def __init__(self, rwkv:RWKVEmbryo):
        self.rwkv = rwkv

    @exe_head
    def set(self, cmds:List[str]) -> str:
        assert len(cmds) == 2, "key & value"
        setattr(self.rwkv, cmds[0], eval(cmds[1]))
        return "Done"
    
    @exe_head
    def get(self, cmds:List[str]) -> str:
        assert len(cmds) == 1, "key"
        return str(getattr(self.rwkv, cmds[0]))
    
    def execute(self, cmd:str) -> str:
        cmds = cmd.split()
        if hasattr(self, cmds[0]):
            try:
                return getattr(self, cmds[0])(cmds[1:])
            except:
                return traceback.format_exc()
        else:
            return f"no command: {cmds[0]}"
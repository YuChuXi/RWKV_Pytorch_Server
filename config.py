import time
from typing import List
import torch

# ======================================== Model settings ========================================

RWKV_JIT_ON = True
RWKV_DEVICE = "cuda"

# ======================================== Script settings ========================================

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 1.0
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.2
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 0.0
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 1.0
# When the model repeats several words, the penalty will increase sharply and pull the model back, set it to 1.0-1.2 is a good idea.
PRPEAT_PENALTY: float = 1.00
# Mitigating penalties after a certain length of context
PENALTY_MITIGATE: float = 1.02
# How engaged a model is with prompt, which could be used to mitigate Alzheimer's disease in small models
OBSTINATE: float = 0.1

MAX_GENERATION_LENGTH: int = 384
END_OF_TEXT_TOKEN: int = 0
NEW_LINE_OF_TEXT_TOKEN: int = 261
# The bigger you are, the less you talk. 1.000 - 1.005
NEW_LINE_LORA:float = 1.000

THREADS: int = 3
MAX_CHUNK = 1024
SEED = None

MODEL_NAME: str = "weight/RWKV-x060-World-7B-v2.1-20240507-ctx4096"
MODEL_NAME = "weight/RWKV-6-v2-ctx4096.roleplay"
MODEL_NAME = "weight/RWKV-x060-World-1B6-v2.1-20240328-ctx4096"
MODEL_NAME = "weight/RWKV-x060-World-3B-v2.1-Claude-nsfw"
MODEL_NAME = "weight/RWKV-x060-World-7B-v2.1-20240507-ctx4096"


MODEL_STATE_NAME: str = "default.state"

TONKEIZER_DICT: str = "asset/rwkv_vocab_v20230424.txt"



# ========================================= App settings ==========================================

APP_BIND: List[str] = ["0.0.0.0:48088", "[::]:48089"]
APP_AUTOSAVE_TIME: int = 600

APP_TEST_MESSAGE: str = """告诉我关于你的一切。"""


# ========================================= Chat settings =========================================

# English, Chinese, Japanese
CHAT_LANGUAGE: str = "Chinese"
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
CHAT_PROMPT_TYPE: str = "Chat-MoZi-N"
CHAT_PROMPT_TYPE = "Chat-Ella"
CHAT_PROMPT_TYPE = "Chat-XiaoPu"

# ======================================== Gener settings =========================================

NICKGENER_PROMPT: str = """注: 
以下是一张用户名与称呼的对照表，称呼是用户名中最具有特色的部分, 且尽可能短. 

玉子是个废物喵
玉子

沐沐
沐沐

墨子不是猫
墨子

不想加班的朋朋
朋朋

只有鱼骨头吃的喵
鱼骨头喵

猫猫博士凌枫
猫猫博士


"""

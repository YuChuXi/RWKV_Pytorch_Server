# -*- coding: utf-8 -*-
# Provides terminal-based chat interface for RWKV model.
# Usage: python chat_with_bot.py C:\rwkv.cpp-169M.bin
# Prompts and code adapted from https://github.com/Blink/bloEmbryo:/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py

import os
import pickle
import json
import tqdm
import time
import torch
import asyncio
import copy
import re
from src.model import RWKV_RNN
from src import sampler
from src.rwkv_tokenizer import RWKV_TOKENIZER

# from tokenizer_util import get_tokenizer
from typing import List, Dict, Optional, Tuple
from utils import (
    prxxx,
    check_dir,
    check_file,
    log_call,
    use_async_lock,
    check_dir_async,
    check_file_async,
    run_in_async_thread,
)

from config import (
    RWKV_DEVICE,
    TEMPERATURE,
    TOP_P,
    PRESENCE_PENALTY,
    FREQUENCY_PENALTY,
    PRPEAT_PENALTY,
    EXCEPTIONAL_TOKENS,
    OBSTINATE_ALPHA,
    OBSTINATE_BATA,
    NAGGING,
    PENALTY_MITIGATE,
    MAX_GENERATION_LENGTH,
    END_OF_TEXT_TOKEN,
    NEW_LINE_OF_TEXT_TOKEN,
    NEW_LINE_LORA,
    THREADS,
    MAX_CHUNK,
    SEED,
    MODEL_NAME,
    MODEL_STATE_NAME,
    TONKEIZER_DICT,
    CHAT_LANGUAGE,
    CHAT_PROMPT_TYPE,
)

if RWKV_DEVICE == "musa":
    import torch_musa
elif RWKV_DEVICE == "npu":
    import torch_npu

torch.random.manual_seed(int(time.time() * 1e6 % 2**30) if SEED is None else SEED)
torch.set_num_threads(THREADS)

prxxx(f"$32<Loading RWKV model>   $34<file>: {MODEL_NAME}")
model = RWKV_RNN(
    args={
        "MODEL_NAME": MODEL_NAME,
        "vocab_size": 65536,
        "device": RWKV_DEVICE,
        "onnx_opset": "18",
        "dataformat": "bf16",
    }
).to(RWKV_DEVICE)

check_dir("data")
if check_file(f"data/tokenizer.pkl"):
    prxxx(f"$32<Loading tokenizer>   file: data/tokenizer.pkl")
    with open(f"data/tokenizer.pkl", "rb") as f:
        tokenizer: RWKV_TOKENIZER = pickle.load(f)
else:
    prxxx(f"$32<Loading tokenizer>   file: {TONKEIZER_DICT}")
    tokenizer: RWKV_TOKENIZER = RWKV_TOKENIZER(TONKEIZER_DICT)
    with open(f"data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)


@log_call
def tokenizer_encode(s: str) -> List[int]:
    return tokenizer.encode([s])[0]


@log_call
def tokenizer_decode(l: List[int]) -> str:
    return tokenizer.decodeBytes(l).decode(encoding="utf-8", errors="ignore")


# ========================================= Embryo states =========================================

class RWKVState:
    @log_call
    def __init__(self):
        self.logits: torch.Tensor = None
        self.state: torch.Tensor = None
        self.processed_tokens: List[int] = []
        self.processed_tokens_counts: Dict[int, int] = {}

    @log_call
    @run_in_async_thread
    def save(self, state_name: str):
        check_dir(f"data/{state_name}")
        with open(f"data/{state_name}/tokens.pkl", "wb") as f:
            pickle.dump(
                {
                    "processed_tokens": self.processed_tokens,
                    "logits": self.logits.cpu(),
                    "processed_tokens_counts": self.processed_tokens_counts,
                },
                f,
            )
        # np.save(f"data/{state_name}/state.npy", (np.arcsinh(self.state) * 24).clip(-128,127).astype(np.int8))
        torch.save(self.state, f"data/{state_name}/state.pth")
        return self

    @log_call
    def save_sync(self, state_name: str):
        check_dir(f"data/{state_name}")
        with open(f"data/{state_name}/tokens.pkl", "wb") as f:
            pickle.dump(
                {
                    "processed_tokens": self.processed_tokens,
                    "logits": self.logits.cpu(),
                    "processed_tokens_counts": self.processed_tokens_counts,
                },
                f,
            )
        # np.save(f"data/{state_name}/state.npy", (np.arcsinh(self.state) * 24).clip(-128,127).astype(np.int8))
        torch.save(self.state, f"data/{state_name}/state.pth")
        return self

    @log_call
    @run_in_async_thread
    def load(self, state_name: str):
        if not check_file(f"data/{state_name}/state.pth"):
            return None

        have_history = False
        try:
            with open(f"data/{state_name}/tokens.pkl", "rb") as f:
                data: Dict[str, object] = pickle.load(f)
                self.processed_tokens: List[int] = data["processed_tokens"]
                self.logits: torch.Tensor = data["logits"]
                self.processed_tokens_counts: Dict[int, int] = data[
                    "processed_tokens_counts"
                ]
                have_history = True
        except:
            prxxx(f" $31<! State>: {state_name} missing history ! ")
        # self.state: torch.Tensor = np.sinh(np.load(f"data/{state_name}/state.npy").astype(np.float32) / 24)
        with torch.no_grad():
            load_state: torch.Tensor | Dict[str, torch.Tensor] | None = torch.load(
                f"data/{state_name}/state.pth", map_location="cpu"
            )
            if load_state is None:
                return None
            if isinstance(load_state, torch.Tensor):
                self.state = load_state
            else:
                # 32, 64, 64, 64 -> (32*(64+2)), (64*64)
                n_layer= len(load_state)
                n_head, head_size, _ = load_state.popitem()[1].shape
                state = torch.zeros((1, n_layer * (head_size+2), n_head * head_size))
                for i, (key, value) in enumerate(load_state.items()):
                    state[
                        :, ((2 + head_size) * i + 2) : ((2 + head_size) * (i + 1)), :
                    ] = (value.contiguous().permute(0, 2, 1).reshape(head_size, -1))
                self.state = state
            if not have_history:
                self.logits = torch.zeros(model.emb.weight.shape[0])
            self.state = self.state.to(device=model.emb.weight.device, dtype=model.emb.weight.dtype)
            self.logits = self.logits.to(device=model.emb.weight.device, dtype=model.emb.weight.dtype)

        return self

    @log_call
    @run_in_async_thread
    def copy(self):
        nself = RWKVState()
        with torch.no_grad():
            nself.logits = self.logits.clone()
            nself.state = self.state.clone()
        nself.processed_tokens = copy.deepcopy(self.processed_tokens)
        nself.processed_tokens_counts = copy.deepcopy(self.processed_tokens_counts)
        return nself

    @log_call
    async def mix(self, state, weight: float):
        staot0 = await self.copy()
        if weight == 0:
            return staot0
        staot0.state = staot0.state * (1 - weight) + state.state * weight
        staot0.logits = staot0.logits * (1 - weight) + state.logits * weight

        return staot0

    @log_call
    @run_in_async_thread
    def mix_inplace(self, state, weight: float):
        if weight == 0:
            return self
        self.state = self.state * (1 - weight) + state.state * weight
        self.logits = self.logits * (1 - weight) + state.logits * weight

        return self
    
    @log_call
    async def mix_n(self, state, weight: float):
        staot0 = await self.copy()
        if weight == 0:
            return staot0
        
        h = state.state.shape[-2]
        w = torch.arange(0, h, device=staot0.state.device, dtype=staot0.state.dtype) // (model.head_size + 2)
        w = w / w.max()
        w = weight / (OBSTINATE_BATA * (w - 0.5)**2 + 1)
        w = w.reshape(1, -1, 1)
        staot0.state = staot0.state * (1 - w) + state.state * w

        staot0.logits = staot0.logits * (1 - weight) + state.logits * weight
        return staot0
    
    @log_call
    @run_in_async_thread
    def mix_n_inplace(self, state, weight: float):
        if weight == 0:
            return self
        
        h = state.state.shape[-2]
        w = torch.arange(0, h, device=self.state.device, dtype=self.state.dtype) // (model.head_size + 2)
        w = w / w.max()
        w = weight / (OBSTINATE_BATA * (w - 0.5)**2 + 1)
        w = w.reshape(1, -1, 1)
        self.state = self.state * (1 - w) + state.state * w

        self.logits = self.logits * (1 - weight) + state.logits * weight
        return self
    
    @log_call
    async def mix_max(self, state, weight: float):
        staot0 = await self.copy()
        if weight == 0:
            return staot0
        mean = staot0.state.mean()
        staot0.state = torch.maximum(
            staot0.state, state.state / state.state.mean() * weight
        )
        staot0.state = staot0.state / staot0.state.mean() * mean

        mean = staot0.logits.mean()
        staot0.logits = torch.maximum(
            staot0.logits, state.logits / state.logits.mean() * weight
        )
        staot0.logits = staot0.logits / staot0.logits.mean() * mean
        return staot0

    @log_call
    @run_in_async_thread
    def mix_max_inplace(self, state, weight: float):
        if weight == 0:
            return self
        mean = self.state.mean()
        self.state = torch.maximum(
            self.state, state.state / state.state.mean() * weight
        )
        self.state = self.state / self.state.mean() * mean

        mean = self.logits.mean()
        self.logits = torch.maximum(
            self.logits, state.logits / state.logits.mean() * weight
        )
        self.logits = self.logits / self.logits.mean() * mean
        return self

    @log_call
    def size(self):
        return self.state.size()


state_cache: Dict[str, RWKVState] = {}


# ========================================= Embryo prompt =========================================


class RWKVPrompt:
    @log_call
    def __init__(
        self,
        string: str | None = None,
        file: str | None = None,
        language: str | None = CHAT_LANGUAGE,
        type: str | None = CHAT_PROMPT_TYPE,
    ) -> None:
        if string is not None:
            prxxx(f"$32<Loading RWKV prompt>   $34<string>: {self.get_preview(string)}")
            self.prompt = string
            self.user = None
            self.bot = None
            self.separator = None
            self.ignore = None
            self.multiuser = False
            self.split = "\n\n"
            self.format = None
            self.state = None
        else:
            prompt_config = f"prompt/{language}-{type}.json"
            if not file is None:
                prompt_config = file
            prxxx(f"$32<Loading RWKV prompt>   $34<config>: {prompt_config}")
            with open(
                prompt_config, "r", encoding="utf-8", errors="ignore"
            ) as json_file:
                prompt_data: Dict[str, str] = json.load(json_file)
                self.user = prompt_data.get("user", "<|user|>")
                self.bot = prompt_data.get("bot", "<|me|>")
                self.separator = prompt_data.get("separator", ":")
                self.prompt = prompt_data.get("prompt", None)
                self.ignore = prompt_data.get("ignore", "")
                self.multiuser = prompt_data.get("multiuser", False)
                self.split = prompt_data.get("split", "\n\n")
                self.format = prompt_data.get("format", None)
                self.state = prompt_data.get("state", None)
                if check_file(self.prompt):
                    with open(self.prompt, "r", encoding="utf-8", errors="ignore") as f:
                        self.prompt = f.read()

            assert self.prompt != "" or self.state is not None, "Prompt must not be empty"

    @log_call
    def __str__(self):
        if self.prompt is not None:
            return self.get_preview(self.prompt)
        if self.state is not None:
            return f"state:{self.state}"
        return "None" 

    @log_call
    def get_preview(self, string):
        string = string.strip().replace("\n", "\\n")
        return string[: min(16, len(string))]

    @log_call
    def process_ignore(self, string):
        if self.ignore is None or self.ignore == "":
            return string
        if isinstance(self.ignore, str):
            self.ignore = re.compile(self.ignore)
        return self.ignore.sub("", string)

    @log_call
    def process_format(self, name, message="", tail=None):
        tail = self.split if tail is None else tail
        if self.format is None:
            return f"{name}{self.separator} {message}{tail}"
        return self.format % (name, message) + tail


DEFAULT_PROMPT = RWKVPrompt()


# ============================================ Embryo =============================================


class RWKVInterruptException(Exception):
    pass


class RWKVEmbryo:
    @log_call
    def __init__(
        self,
        id: str,
        state_name: str = MODEL_STATE_NAME,
        prompt: RWKVPrompt = DEFAULT_PROMPT,
    ):
        check_dir(f"data/{id}")
        assert len(id) > 0, "ID must not be empty"
        assert not state_name is None and len(state_name) > 0, "State must not be empty"
        assert id != state_name, "ID != State !!!"

        self.id: str = str(id)
        self.prompt: RWKVPrompt = (
            RWKVPrompt(prompt) if isinstance(prompt, str) else prompt
        )
        self.default_state: str = state_name
        self.debug = False
        self.debug_log = None

        self.state: RWKVState = RWKVState()
        self.state_lock: asyncio.Lock = asyncio.Lock()
        self.need_save: bool = False

        self.presence_penalty: float = PRESENCE_PENALTY
        self.frequency_penalty: float = FREQUENCY_PENALTY
        self.repeat_penalty: float = PRPEAT_PENALTY
        self.penalty_mitigate: float = PENALTY_MITIGATE
        self.temperature: float = TEMPERATURE
        self.top_p: float = TOP_P

        self.have_interrupt: bool = False

        self.mlog = open(f"data/{self.id}/model.log", "ab+")
        prxxx(f"$32<Init RWKV>   $34<id>: {id} | $34<state>: {state_name} | $34<prompt>: {prompt}")

    @log_call
    def __del__(self):
        self.mlog.close()

    @log_call
    async def load_state(
        self,
        state_name: str,
        prompt: RWKVPrompt = None,
        reprompt=False,
        q: bool = False,
    ) -> None:
        if (prompt is not None and prompt.prompt is not None) and (
            reprompt
            or (not await check_file_async(f"data/{self.default_state}/tokens.pkl"))
        ):
            prompt_tokens = tokenizer_encode(prompt.prompt)
            ltime = time.time()
            await self.process_tokens(prompt_tokens)
            prxxx(
                f"$32<Processed prompt tokens>   $34<used>: {int(time.time()-ltime)} s", q=q
            )
            await self.save_state(self.id, must=True, q=q)
            await self.save_state(self.default_state, must=True, q=q)
            self.mlog.write(
                f' : Load prompt ["{prompt.prompt}"]\n\n'.encode(encoding="utf-8")
            )
            return


        if prompt is not None and prompt.state is not None:
            await self.state.load(prompt.state)
            await self.save_state(self.id, must=True, q=q)
            await self.save_state(self.default_state, must=True, q=q)

        state_names = [self.default_state, MODEL_STATE_NAME]
        async with self.state_lock:
            loaded = await self.state.load(state_name)
            if loaded:
                prxxx(f"$32<Load state>   $34<name>: {state_name}", q=q)
                self.mlog.write(
                    f" : Load state [{state_name}]\n\n".encode(encoding="utf-8")
                )
            for name in state_names:  # 检查缓存 & 加载
                await asyncio.sleep(0)
                if name not in state_cache:
                    if (state := await RWKVState().load(name)) is not None:
                        state_cache[name] = await state.copy()
                        prxxx(f"$32<Cache state>   $34<name>: {name}", q=q)
                if loaded:
                    continue
                if name in state_cache:
                    loaded = self.state = await state_cache[name].copy()
                    prxxx(f"$32<Load state from cache>   $34<name>: {name}", q=q)
                    self.mlog.write(
                        f" : Load state [{name}]\n\n".encode(encoding="utf-8")
                    )
            return

    @log_call
    async def save_state(
        self, state_name: str, must: bool = False, q: bool = False
    ) -> None:
        if self.need_save or must:
            async with self.state_lock:
                await self.state.save(state_name)
            prxxx(f"$32<Save state>   $34<name>: {state_name}", q=q)
            self.mlog.write(
                f" : Save state [{state_name}]\n\n".encode(encoding="utf-8")
            )
            self.need_save = False
        self.mlog.flush()

    @log_call
    async def reset_state(self, q: bool = False) -> None:
        await self.load_state(self.default_state, q=q)
        await self.save_state(self.id, must=True, q=q)

    @log_call
    async def init_state(self, **kwargs) -> None:
        await self.load_state(self.id, **kwargs)

    @log_call
    def is_busy(self) -> bool:
        return self.state_lock.locked()

    @log_call
    def interrupt(self) -> None:
        self.have_interrupt = True

    @log_call
    def clean_interrupt(self) -> None:
        self.have_interrupt = False

    @log_call
    async def check_state(self):
        if not self.debug:
            return
        logs = []
        logs.append(f"c_l:{len(self.state.processed_tokens)}")
        logs.append(f"s_mx:{self.state.state.max().item()},{self.state.state.min().item()}")        

        logs.append("\n")
        logs = " ".join(logs)
        print(logs)
        self.debug_log.write(logs)
        self.debug_log.flush()
        return
        tt = list(np.where(sampling.sample_probs(self.state.logits.copy()) > 0)[0])
        if tt[0] == 0:
            tt = tt[1:]
        print(tokenizer_decode(tt))
        return
        logit = self.logits[self.logits >= 0]
        prxxx("logits", logit[-128:])
        prxxx("pedt", self.state.processed_tokens_counts)
        pppp = list(
            map(
                lambda x: self.repeat_penalty**x,
                self.state.processed_tokens_counts.values(),
            )
        )
        pppp.sort()
        prxxx("pppp", pppp)
        return
        l = self.logits
        s = self.state
        if "numpy" in dir(s):
            l = l.numpy()
            s = s.numpy()
        s_var = s.var()
        prxxx(
            "*  logits:\tmean\t%.2f\tvar\t%.2f\tmax\t%.2f\tmin %.2f"
            % (l.mean(), l.var(), l.max(), l.min())
        )
        prxxx(
            "*  state:\tmean\t%.2f\tvar\t%.2f\tmax\t%.2f\tmin %.2f"
            % (s.mean(), s_var, s.max(), s.min())
        )
        prxxx(
            "*  san:\t%.3f" % (10 - np.log(s_var) / 0.6214608098422),
            "" if s_var < 500 else "!",
        )
        # self.presence_penalty = s_var/72
        # self.frequency_penalty = s_var/36

    @log_call
    async def process_processed_tokens_counts(self, tokens: List[int]) -> None:
        for token in tokens:
            self.state.processed_tokens.append(token)
            if token not in self.state.processed_tokens_counts:  # 词频统计
                self.state.processed_tokens_counts[token] = 1
            else:
                self.state.processed_tokens_counts[token] += 1

            for token in self.state.processed_tokens_counts:
                self.state.processed_tokens_counts[token] /= self.penalty_mitigate

    @log_call
    async def process_token_penalty(
        self, logits: torch.Tensor, len_gen: int = 0
    ) -> torch.Tensor:
        logits[END_OF_TEXT_TOKEN] = -1e9
        for token in self.state.processed_tokens_counts:
            exc = EXCEPTIONAL_TOKENS[token] if token in EXCEPTIONAL_TOKENS else 1.0
            logits[token] -= exc * (
                # 传统惩罚
                self.presence_penalty
                + self.state.processed_tokens_counts[token] * self.frequency_penalty
                # 新惩罚
                # + self.repeat_penalty ** self.state.processed_tokens_counts[token]
                # - 1
            )

            logits[NEW_LINE_OF_TEXT_TOKEN] *= NEW_LINE_LORA**len_gen
        return logits

    @log_call
    async def process_token(self, token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        await asyncio.sleep(0)

        with torch.no_grad():
            self.state.logits, self.state.state = model.forward(
                torch.tensor([token]), self.state.state
            )
            self.state.logits = self.state.logits[-1, :]
        await self.process_processed_tokens_counts([token])
        self.need_save = True
        await self.check_state()

        self.mlog.write(tokenizer.decodeBytes([token]))
        return self.state.logits, self.state.state

    @log_call
    async def process_tokens(
        self, tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        self.logits, self.state = model.eval_sequence(
            tokens, self.state, self.state, self.logits, use_numpy=True)
        self.state.processed_tokens += tokens
        #self.logits[END_OF_LINE_TOKEN] += new_line_logit_bias
        """

        if len(tokens) == 0:
            return self.state.logits, self.state.state

        if self.is_busy():
            self.interrupt()

        slice_len = 8
        while slice_len * 2 <= len(tokens) and slice_len < MAX_CHUNK:
            slice_len *= 2

        async with self.state_lock:
            with torch.no_grad():
                self.state.logits, self.state.state = (
                    await model.forward_parallel_slices_async(
                        torch.tensor([tokens]).long().to(RWKV_DEVICE),
                        self.state.state,
                        slice_len=slice_len,
                    )
                )
                self.state.logits = self.state.logits[0, -1, :]
        await self.process_processed_tokens_counts(tokens)
        self.need_save = True
        await self.check_state()

        self.mlog.write(tokenizer.decodeBytes(tokens))
        return self.state.logits, self.state.state

    @log_call
    async def gen_future(
        self,
        head: List[int] = [],
        max_len: int = MAX_GENERATION_LENGTH,
        end_of: str = "\n\n",
    ) -> Tuple[str, str]:
        len_head = len(head)
        logits = self.state.logits
        answer: bytes = b""
        end: bytes = end_of.encode(encoding="utf-8")

        async with self.state_lock:
            for i in tqdm.trange(
                max(max_len, len_head),
                desc="Processing future",
                leave=False,
                unit=" tok",
            ):
                await asyncio.sleep(0)
                if i < len_head:
                    token = head[i]
                    logits, _ = await self.process_token(token)
                else:
                    logits = await self.process_token_penalty(logits, i)
                    token: int = (
                        sampler.sample_logits(logits, self.temperature, self.top_p)
                        .cpu()
                        .item()
                    )
                    logits, _ = await self.process_token(token)
                    answer += tokenizer.decodeBytes([token])
                    if end in answer:
                        break

        self.need_save = True
        answer = answer.decode(encoding="utf-8", errors="ignore").strip()
        return self.prompt.process_ignore(answer), answer

    @log_call
    async def call(self, api: str, kwargs: Dict[str, object]):
        return await getattr(self, api)(**kwargs)

    @log_call
    async def get_history(self):
        return tokenizer_decode(self.state.processed_tokens)
    
    @log_call
    async def pre_process_input(self, message) -> str:
        if "-temp=" in message:
            temperature = float(message.split("-temp=")[1].split(" ")[0])
            message = message.replace(f"-temp={temperature:g}", "")
            self.temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in message:
            top_p = float(message.split("-top_p=")[1].split(" ")[0])
            message = message.replace(f"-top_p={top_p:g}", "")
            self.top_p = max(0.0, min(top_p, 1.0))

        if "-debug=" in message:
            debug = message.split("-debug=")[1].split(" ")[0]
            message = message.replace(f"-debug={debug}", "")
            self.debug = {"false":False, "true": True, "0": False, "1":True}[debug.lower()]
            if self.debug:
                self.debug_log = self.debug_log or open(f"data/{self.id}/debug.log", "w")
            elif self.debug is not None:
                self.debug_log.close()
                self.debug_log = None
            prxxx(f"$31<Change debug>   $34<id>: {self.id} | $34<state>: {self.debug}")

        if "+reset" in message:
            await self.reset_state()
            return None # " : Done", " : Done", True
        
        return message


# ======================================== Chater Embryo ==========================================


class RWKVChaterEmbryo(RWKVEmbryo):
    @log_call
    def __init__(
        self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = DEFAULT_PROMPT
    ):
        super().__init__(id, state_name, prompt)

    @log_call
    async def gen_prompt(
        self,
        message_list: List[List[object]],
        time_limit: float = 28800,
        ctx_limit: int = 8192,
    ) -> List[int]:
        """
        [
            [[],[],float],
        #    u  m  t
        ]
        """
        if len(message_list) == 0:
            return []
        now_time = time.time()
        tokens_list = [
            tokenizer_encode(self.prompt.process_format(m[0], f"{m[1]}"))
            for m in message_list
            if now_time - m[2] <= time_limit
        ]
        """
        tokens_list.append(
            tokenizer_encode(f"{self.prompt.bot}{self.prompt.separator}")
        )
        """
        prompt = []
        for tl in tokens_list[::-1]:
            len_token = len(tl)
            if len_token <= ctx_limit:
                ctx_limit -= len_token
                prompt = tl + prompt
            else:
                break

        return prompt

    @log_call
    async def want_to_say(self, head: List[int]) -> float:
        # return 0
        if len(head) == 0:
            return 1.0
        probs = sampler.probs_logits(
            self.state.logits.clone(), self.temperature, self.top_p
        ).cpu()
        return probs[head[0]].item() * NAGGING
    


# ============================================ Chater =============================================


class RWKVChater(RWKVChaterEmbryo):
    @log_call
    def __init__(
        self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = DEFAULT_PROMPT
    ):
        super().__init__(id, state_name, prompt)

    @log_call
    async def chat(
        self,
        message: str,
        chatuser: str = None,
        nickname: str = None,
        addhead: str = "",
    ) -> Tuple[str, str, float]:
        message = await self.pre_process_input(message)
        if message is None:
            return " : Done", " : Done", True

        nickname = self.prompt.bot if nickname is None or nickname == "" else nickname
        message = message.replace(
            nickname, self.prompt.bot
        )  # .strip() # 昵称和提示词不一定一致

        if self.prompt.multiuser:
            user = (
                self.prompt.user
                if (chatuser is None)
                or (chatuser == "")
                or (chatuser == self.prompt.bot)
                else chatuser
            )
        else:
            message = (
                message.replace(chatuser, self.prompt.user)
                if (chatuser is None) or (chatuser == "")
                else message
            )

        head = tokenizer_encode(self.prompt.process_format(self.prompt.bot, tail="") + addhead)
        if message != "" and message[0] != "+":
            prompt = self.prompt.process_format(user, f"{message}")
            await self.process_tokens(tokenizer_encode(prompt))

        if len(message) >= 2 and message[:2] == "++":
            await self.process_tokens(tokenizer_encode(message[2:]))
            head = []

        if self.have_interrupt:
            self.clean_interrupt()
            raise RWKVInterruptException

        answer, original = await self.gen_future(head=head, end_of=self.prompt.split)
        # await self.state.mix_max_inplace(state_cache[self.default_state], OBSTINATE_ALPHA)
        await self.state.mix_n_inplace(state_cache[self.default_state], OBSTINATE_ALPHA)
        # await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE_ALPHA)

        answer = answer.replace(user, chatuser)
        answer = answer.replace(self.prompt.bot, nickname).strip()
        answer = addhead + answer

        return answer, original, await self.want_to_say(head)


# ========================================= Group Chater ==========================================


class RWKVGroupChater(RWKVChaterEmbryo):
    @log_call
    def __init__(
        self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = DEFAULT_PROMPT
    ):
        super().__init__(id, state_name, prompt)
        self.message_cache: List[List[object]] = []
        self.plog = open(f"data/{self.id}/pipeline.log", "a+")

    @log_call
    def save_state(self, state_name: str, must: bool = False, q: bool = False) -> None:
        r = super().save_state(state_name, must, q)
        self.plog.flush()
        return r

    @log_call
    def reset_state(self, q: bool = False):
        r = super().reset_state(q)
        self.message_cache.clear()
        return r

    @log_call
    async def send_message(self, message: str, chatuser: str = None) -> None:
        self.plog.write(f"{chatuser}: {message}\n\n")

        chatuser = (
            self.prompt.user
            if (chatuser is None) or (chatuser == "") or (chatuser == self.prompt.bot)
            else chatuser
        )

        message = await self.pre_process_input(message)
        if message is None:
            return " : Done", " : Done", True

        assert self.prompt.multiuser, "Group Chat need multiuser prompt!"

        self.message_cache.append([chatuser, message, time.time()])
        if len(self.message_cache) > 256:
            self.message_cache = self.message_cache[:200]

        if "+" == message[0]:
            self.message_cache.clear()
            return
            
    @log_call
    async def get_answer(
        self,
        nickname: str = None,
        addhead:str = ""
    ) -> Tuple[str, str, float]:
        self.plog.write(f"{nickname}: ")

        nickname = self.prompt.bot if nickname is None or nickname == "" else nickname
        await self.process_tokens(await self.gen_prompt(self.message_cache))
        self.message_cache.clear()

        if self.have_interrupt:
            self.clean_interrupt()
            raise RWKVInterruptException

        head = tokenizer_encode(self.prompt.process_format(self.prompt.bot, tail="") + addhead)

        answer, original = await self.gen_future(head=head, end_of=self.prompt.split)
        await self.state.mix_n_inplace(state_cache[self.default_state], OBSTINATE_ALPHA)
        # await self.state.mix_max_inplace(state_cache[self.default_state], OBSTINATE_ALPHA)

        answer = answer.replace(self.prompt.bot, nickname).strip()
        answer = addhead + answer

        self.plog.write(f"{answer}\n\n")
        return answer, original, await self.want_to_say(head)


# ========================================== Other ================================================


@log_call
async def process_default_state():
    if await check_file_async(f"data/{MODEL_STATE_NAME}/tokens.pkl"):
        prxxx("$31<Default state was processed>")
    else:
        await RWKVChater(
            id="chat-model", state_name=MODEL_STATE_NAME, prompt=DEFAULT_PROMPT
        ).init_state(prompt = DEFAULT_PROMPT)


"""
print(tokenizer_decode(RWKVChaterEmbryo.gen_prompt(None,[
    ["saefsgrgdr","jgjgjghjghghgjh",time.time()-3600],
    ["hjhjvhvjhb","ftjhvjhjhjhjdsr",time.time()-2400],
    ["guiyutftfd","pohhnkftfgheshj",time.time()-1200],
    ["bnmvnbmcgf","dtrfttdtytyrrr3",time.time()],
    ["uigyfyffrt","jkhfhhgttdhdrrr",time.time()],
    
],time_limit=3600,ctx_limit=1)))
# """

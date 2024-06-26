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
    OBSTINATE,
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

prxxx(f"Loading RWKV model   file: {MODEL_NAME}")
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
    prxxx(f"Loading tokenizer   file: data/tokenizer.pkl")
    with open(f"data/tokenizer.pkl", "rb") as f:
        tokenizer: RWKV_TOKENIZER = pickle.load(f)
else:
    prxxx(f"Loading tokenizer   file: {TONKEIZER_DICT}")
    tokenizer: RWKV_TOKENIZER = RWKV_TOKENIZER(TONKEIZER_DICT)
    with open(f"data/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)


def tokenizer_encode(s: str) -> List[int]:
    return tokenizer.encode([s])[0]


def tokenizer_decode(l: List[int]) -> str:
    return tokenizer.decodeBytes(l).decode(encoding="utf-8", errors="ignore")


# ========================================= Embryo states =========================================


class RWKVState:
    def __init__(self):
        self.logits: torch.Tensor = None
        self.state: torch.Tensor = None
        self.processed_tokens: List[int] = []
        self.processed_tokens_counts: Dict[int, int] = {}

    @run_in_async_thread
    def save(self, state_name: str):
        check_dir(f"data/{state_name}")
        with open(f"data/{state_name}/tokens.pkl", "wb") as f:
            pickle.dump(
                {
                    "processed_tokens": self.processed_tokens,
                    "logits": self.logits,
                    "processed_tokens_counts": self.processed_tokens_counts,
                },
                f,
            )
        # np.save(f"data/{state_name}/state.npy", (np.arcsinh(self.state) * 24).clip(-128,127).astype(np.int8))
        torch.save(self.state, f"data/{state_name}/state.pth")
        return self

    def save_sync(self, state_name: str):
        check_dir(f"data/{state_name}")
        with open(f"data/{state_name}/tokens.pkl", "wb") as f:
            pickle.dump(
                {
                    "processed_tokens": self.processed_tokens,
                    "logits": self.logits,
                    "processed_tokens_counts": self.processed_tokens_counts,
                },
                f,
            )
        # np.save(f"data/{state_name}/state.npy", (np.arcsinh(self.state) * 24).clip(-128,127).astype(np.int8))
        torch.save(self.state, f"data/{state_name}/state.pth")
        return self

    @run_in_async_thread
    def load(self, state_name: str):
        if not check_file(f"data/{state_name}/tokens.pkl"):
            return None

        with open(f"data/{state_name}/tokens.pkl", "rb") as f:
            data: Dict[str, object] = pickle.load(f)

        self.processed_tokens: List[int] = data["processed_tokens"]
        self.logits: torch.Tensor = data["logits"]
        self.processed_tokens_counts: Dict[int, int] = data["processed_tokens_counts"]
        # self.state: torch.Tensor = np.sinh(np.load(f"data/{state_name}/state.npy").astype(np.float32) / 24)
        with torch.no_grad():
            self.state: torch.Tensor = torch.load(f"data/{state_name}/state.pth")
        return self

    @run_in_async_thread
    def copy(self):
        nself = RWKVState()
        with torch.no_grad():
            nself.logits = self.logits.clone()
            nself.state = self.state.clone()
        nself.processed_tokens = copy.deepcopy(self.processed_tokens)
        nself.processed_tokens_counts = copy.deepcopy(self.processed_tokens_counts)
        return nself

    async def mix(self, state, weight: float):
        staot0 = await self.copy()
        if weight == 0:
            return staot0
        staot0.state = staot0.state * (1 - weight) + state.state * weight
        staot0.logits = staot0.logits * (1 - weight) + state.logits * weight

        return staot0

    @run_in_async_thread
    def mix_inplace(self, state, weight: float):
        if weight == 0:
            return self
        self.state = self.state * (1 - weight) + state.state * weight
        self.logits = self.logits * (1 - weight) + state.logits * weight

        return self
    
    async def mix_max(self, state, weight: float):
        staot0 = await self.copy()
        if weight == 0:
            return staot0
        mean = staot0.state.mean()
        staot0.state = torch.maximum(staot0.state, state.state / state.state.mean() * weight)
        staot0.state = staot0.state / staot0.state.mean() * mean

        mean = staot0.logits.mean()
        staot0.logits = torch.maximum(staot0.logits, state.logits / state.logits.mean() * weight)
        staot0.logits = staot0.logits / staot0.logits.mean() * mean
        return staot0

    @run_in_async_thread
    def mix_max_inplace(self, state, weight: float):
        if weight == 0:
            return self
        mean = self.state.mean()
        self.state = torch.maximum(self.state, state.state / state.state.mean() * weight)
        self.state = self.state / self.state.mean() * mean

        mean = self.logits.mean()
        self.logits = torch.maximum(self.logits, state.logits / state.logits.mean() * weight)
        self.logits = self.logits / self.logits.mean() * mean
        return self

    def size(self):
        return self.state.size()


state_cache: Dict[str, RWKVState] = {}


# ========================================= Embryo prompt =========================================


class RWKVPrompt:
    def __init__(
        self,
        string: str = None,
        file: str = None,
        language: str = CHAT_LANGUAGE,
        type: str = CHAT_PROMPT_TYPE,
    ) -> None:
        if not string is None:
            prxxx(f"Loading RWKV prompt   string: {self.get_preview(string)}")
            self.prompt = string
            self.user = None
            self.bot = None
            self.separator = None
            self.ignore = None
            self.multiuser = False
        else:
            prompt_config = f"prompt/{language}-{type}.json"
            if not file is None:
                prompt_config = file
            prxxx(f"Loading RWKV prompt   config: {prompt_config}")
            with open(
                prompt_config, "r", encoding="utf-8", errors="ignore"
            ) as json_file:
                prompt_data: Dict[str, str] = json.load(json_file)
                self.user = prompt_data.get("user", "<|user|>")
                self.bot = prompt_data.get("bot", "<|me|>")
                self.separator = prompt_data.get("separator", ":")
                self.prompt = prompt_data.get("prompt", "")
                self.ignore = prompt_data.get("ignore", "")
                self.multiuser = prompt_data.get("multiuser", False)
                if check_file(self.prompt):
                    with open(self.prompt, "r", encoding="utf-8", errors="ignore") as f:
                        self.prompt = f.read()

            assert self.prompt != "", "Prompt must not be empty"

    def __str__(self):
        return "None" if self.prompt is None else self.get_preview(self.prompt)

    def get_preview(self, string):
        string = string.strip().replace("\n", "\\n")
        return string[: min(16, len(string))]

    def process_ignore(self, string):
        if self.ignore is None or self.ignore == "":
            return string
        if isinstance(self.ignore, str):
            self.ignore = re.compile(self.ignore)
        return self.ignore.sub("", string)


DEFAULT_PROMPT = RWKVPrompt()


# ============================================ Embryo =============================================


class RWKVInterruptException(Exception):
    pass


class RWKVEmbryo:
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
        prxxx(f"Init RWKV   id: {id} | state: {state_name} | prompt: {prompt}")

    def __del__(self):
        self.mlog.close()

    @log_call
    async def load_state(
        self,
        state_name: str,
        prompt: RWKVPrompt = DEFAULT_PROMPT,
        reprompt=False,
        q: bool = False,
    ) -> None:
        if (prompt is not None) and (
            reprompt
            or (not await check_file_async(f"data/{self.default_state}/tokens.pkl"))
        ):
            prompt_tokens = tokenizer_encode(prompt.prompt)
            prxxx(f"Process prompt tokens   length: {len(prompt_tokens)} tok", q=q)
            ltime = time.time()
            await self.process_tokens(prompt_tokens)
            prxxx(f"Processed prompt tokens   used: {int(time.time()-ltime)} s", q=q)
            await self.save_state(self.id, must=True, q=q)
            await self.save_state(self.default_state, must=True, q=q)
            self.mlog.write(
                f' : Load prompt ["{prompt.prompt}"]\n\n'.encode(encoding="utf-8")
            )
            return

        state_names = [self.default_state, MODEL_STATE_NAME]

        async with self.state_lock:
            loaded = await self.state.load(state_name)
            if loaded:
                prxxx(f"Load state   name: {state_name}", q=q)
                self.mlog.write(
                    f" : Load state [{state_name}]\n\n".encode(encoding="utf-8")
                )
            for name in state_names:  # 检查缓存 & 加载
                await asyncio.sleep(0)
                if name not in state_cache:
                    if (state := await RWKVState().load(name)) is not None:
                        state_cache[name] = await (state.copy())
                        prxxx(f"Cache state   name: {name}", q=q)
                if loaded:
                    continue
                if name in state_cache:
                    loaded = self.state = await (state_cache[name].copy())
                    prxxx(f"Load state from cache   name: {name}", q=q)
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
            prxxx(f"Save state   name: {state_name}", q=q)
            self.mlog.write(
                f" : Save state [{state_name}]\n\n".encode(encoding="utf-8")
            )
            self.need_save = False
        self.mlog.flush()

    @log_call
    async def reset_state(self, q: bool = False) -> None:
        await self.load_state(self.default_state, q=q)
        await self.save_state(self.id, must=True, q=q)

    async def init_state(self) -> None:
        await self.load_state(self.id, self.prompt)

    def is_busy(self) -> bool:
        return self.state_lock.locked()

    def interrupt(self) -> None:
        self.have_interrupt = True

    def clean_interrupt(self) -> None:
        self.have_interrupt = False

    @log_call
    async def check_state(self):
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
            logits[token] -= (
                # 传统惩罚
                self.presence_penalty
                + self.state.processed_tokens_counts[token] * self.frequency_penalty
                # 新惩罚
                #+ self.repeat_penalty ** self.state.processed_tokens_counts[token]
                #- 1
            )

            logits[NEW_LINE_OF_TEXT_TOKEN] *= NEW_LINE_LORA ** len_gen
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

    async def call(self, api: str, kwargs: Dict[str, object]):
        return await getattr(self, api)(**kwargs)

    async def get_history(self):
        return tokenizer_decode(self.state.processed_tokens)


# ======================================== Chater Embryo ==========================================


class RWKVChaterEmbryo(RWKVEmbryo):
    def __init__(
        self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = DEFAULT_PROMPT
    ):
        super().__init__(id, state_name, prompt)

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
            tokenizer_encode(f"{m[0]}{self.prompt.separator} {m[1]}\n\n")
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

    async def is_want_to_say(self, head: List[int]) -> float:
        if len(head) == 0:
            return True
        probs = sampler.probs_logits(
            self.state.logits.clone(), self.temperature, self.top_p
        ).cpu()
        return max(probs[head[0]].item() - self.top_p, 0)


# ============================================ Chater =============================================


class RWKVChater(RWKVChaterEmbryo):
    def __init__(
        self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = DEFAULT_PROMPT
    ):
        super().__init__(id, state_name, prompt)

    async def chat(
        self,
        message: str,
        chatuser: str = None,
        nickname: str = None,
        debug: bool = False,
    ) -> Tuple[str, str, float]:
        self.debug = debug

        if "-temp=" in message:
            temperature = float(message.split("-temp=")[1].split(" ")[0])
            message = message.replace("-temp=" + f"{temperature:g}", "")
            self.temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in message:
            top_p = float(message.split("-top_p=")[1].split(" ")[0])
            message = message.replace("-top_p=" + f"{top_p:g}", "")
            self.top_p = max(0.0, min(top_p, 1.0))

        if "+reset" in message:
            await self.reset_state()
            return " : Done", " : Done", True

        nickname = self.prompt.bot if nickname is None or nickname == "" else nickname
        message = message.replace(
            nickname, self.prompt.bot
        )  # .strip() # 昵称和提示词不一定一致

        if self.prompt.multiuser:
            user = self.prompt.user if (chatuser is None) or (chatuser == "") or (chatuser == self.prompt.bot) else chatuser
        else:
            message = message.replace(chatuser, self.prompt.user) if (chatuser is None) or (chatuser == "") else message

        head = tokenizer_encode(f"{self.prompt.bot}{self.prompt.separator}")
        if message != "" and message[0] != "+":
            prompt = f"{user}{self.prompt.separator} {message}\n\n"
            await self.process_tokens(tokenizer_encode(prompt))

        if len(message) >= 2 and message[:2] == "++":
            await self.process_tokens(tokenizer_encode(message[2:]))
            head = []

        if self.have_interrupt:
            self.clean_interrupt()
            raise RWKVInterruptException

        answer, original = await self.gen_future(head=head, end_of="\n\n")
        #await self.state.mix_max_inplace(state_cache[self.default_state], OBSTINATE)
        await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE)
        # await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE)

        answer = answer.replace(user, chatuser)
        answer = answer.replace(self.prompt.bot, nickname).strip()

        return answer, original, await self.is_want_to_say(head)


# ========================================= Group Chater ==========================================


class RWKVGroupChater(RWKVChaterEmbryo):
    def __init__(
        self, id: str, state_name: str = MODEL_STATE_NAME, prompt: str = DEFAULT_PROMPT
    ):
        super().__init__(id, state_name, prompt)
        self.message_cache: List[List[object]] = []
        self.plog = open(f"data/{self.id}/pipeline.log", "a+")

    def save_state(self, state_name: str, must: bool = False, q: bool = False) -> None:
        r = super().save_state(state_name, must, q)
        self.plog.flush()
        return r

    def reset_state(self, q: bool = False):
        r = super().reset_state(q)
        self.message_cache.clear()
        return r

    async def send_message(self, message: str, chatuser: str = None) -> None:
        self.plog.write(f"{chatuser}: {message}\n\n")

        chatuser = self.prompt.user if (chatuser is None) or (chatuser == "") or (chatuser == self.prompt.bot) else chatuser
        if "-temp=" in message:
            temperature = float(message.split("-temp=")[1].split(" ")[0])
            message = message.replace("-temp=" + f"{temperature:g}", "")
            self.temperature = max(0.2, min(temperature, 5.0))

        if "-top_p=" in message:
            top_p = float(message.split("-top_p=")[1].split(" ")[0])
            message = message.replace("-top_p=" + f"{top_p:g}", "")
            self.top_p = max(0.0, min(top_p, 1.0))

        if "+reset" in message:
            await self.reset_state()
            return

        if "+" == message[0]:
            self.message_cache.clear()
            return

        assert self.prompt.multiuser , "Group Chat need multiuser prompt!"

        self.message_cache.append([chatuser, message, time.time()])
        if len(self.message_cache) > 256:
            self.message_cache = self.message_cache[200]

    async def get_answer(
        self,
        nickname: str = None,
    ) -> Tuple[str, str, float]:
        self.plog.write(f"{nickname}: ")

        nickname = self.prompt.bot if nickname is None or nickname == "" else nickname
        await self.process_tokens(await self.gen_prompt(self.message_cache))
        self.message_cache.clear()

        if self.have_interrupt:
            self.clean_interrupt()
            raise RWKVInterruptException

        head = tokenizer_encode(f"{self.prompt.bot}{self.prompt.separator}")

        answer, original = await self.gen_future(head=head, end_of="\n\n")
        await self.state.mix_inplace(state_cache[self.default_state], OBSTINATE)
        # await self.state.mix_max_inplace(state_cache[self.default_state], OBSTINATE)
        
        answer = answer.replace(self.prompt.bot, nickname).strip()

        self.plog.write(f"{answer}\n\n")
        return answer, original, await self.is_want_to_say(head)

# ========================================== Other ================================================


async def process_default_state():
    if await check_file_async(f"data/{MODEL_STATE_NAME}/tokens.pkl"):
        prxxx("Default state was processed")
    else:
        await RWKVChater(
            id="chat-model", state_name=MODEL_STATE_NAME, prompt=RWKVPrompt()
        ).init_state()


"""
print(tokenizer_decode(RWKVChaterEmbryo.gen_prompt(None,[
    ["saefsgrgdr","jgjgjghjghghgjh",time.time()-3600],
    ["hjhjvhvjhb","ftjhvjhjhjhjdsr",time.time()-2400],
    ["guiyutftfd","pohhnkftfgheshj",time.time()-1200],
    ["bnmvnbmcgf","dtrfttdtytyrrr3",time.time()],
    ["uigyfyffrt","jkhfhhgttdhdrrr",time.time()],
    
],time_limit=3600,ctx_limit=1)))
# """

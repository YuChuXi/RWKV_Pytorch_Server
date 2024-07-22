import httpx
import threading
import tqdm
import config


def asend():
    for i in tqdm.tgrange(1e4):
        httpx.post(
                f"http://{config.APP_BIND[0]}/group_chat_send",
                data={"message": "uibroybavouarfersbogboreo", "id": "tester", "user": "user"},
        )

for i in range(16):
    threading.Thread(target=asend).start()
import httpx
import threading
import tqdm
import config


def asend():
    for i in tqdm.trange(10000):
        httpx.post(
                f"http://{config.APP_BIND[0]}/group_chat_send",
                data={"message": "uibroybavouarfersbogboreo", "id": "tester", "user": "user"},
                proxies={}
        )

for i in range(2):
    threading.Thread(target=asend).start()

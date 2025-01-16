import sys
import os
import time

os.system("cls")
from typing import List
import json
import pathlib
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

dotenv.load_dotenv()


def get_config():
    path = pathlib.Path(__file__).parent / "config.json"
    txt = path.read_text()
    config = json.loads(txt)
    return config


GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
CONFIG = get_config()

model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
convo = [
    SystemMessage(
        "You are a chatbot named cloff. You answer human's queries as concisely as possible."
    ),
    AIMessage("Hello! What would you like to know?"),
]


class tutil:
    def to_RGB(text: str, c: List[int]) -> str:
        r, g, b = c
        # 8-bit color
        return f"\033[38;2;{r};{g};{b}m{text}\033[0;0m"

    def cloff_tag(text: str) -> str:
        return tutil.to_RGB(text, CONFIG["cloff"]["tag"])

    def cloff_text(text: str) -> str:
        return tutil.to_RGB(text, CONFIG["cloff"]["text"])

    def human_tag(text: str) -> str:
        return tutil.to_RGB(text, CONFIG["human"]["tag"])

    def human_text(text: str) -> str:
        return tutil.to_RGB(text, CONFIG["human"]["text"])

    def grey(text: str) -> str:
        return tutil.to_RGB(text, [179, 179, 179])


if __name__ == "__main__":
    print(f"{tutil.cloff_tag('[cloff]')} {tutil.cloff_text(convo[-1].content)}\n")
    while True:
        try:
            user_input = input(f"  {tutil.human_tag('[you]')} ")
            if user_input == "exit":
                sys.exit(0)
            if user_input == "clear":
                os.system("cls")
                print(
                    f"{tutil.cloff_tag('[cloff]')} {tutil.cloff_text(convo[-1].content)}\n"
                )
                continue
            convo.append(HumanMessage(user_input))
            response = ""
            start = time.time()
            for i, msg in enumerate(model.stream(convo)):
                if i == 0:
                    print(f"\n{tutil.cloff_tag('[cloff]')} ", end="")
                for ch in msg.content:
                    print(tutil.cloff_text(ch), end="")
                    response += ch
                    # time.sleep(0.001)
            print(tutil.grey(f"({time.time() - start:.2f}s)"), end="")
            convo.append(AIMessage(response))
            print("\n")
        except KeyboardInterrupt:
            print(
                f"""\n\n{tutil.cloff_tag('[cloff]')} {tutil.cloff_text("Type 'exit' to exit.")}\n"""
            )
            continue

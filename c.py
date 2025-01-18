import sys
import os
import time

os.system("cls")
from typing import List, Dict
import json
import pathlib
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

dotenv.load_dotenv()


class Config:
    def __init__(self):
        # Load config
        path = pathlib.Path(__file__).parent / "config.json"
        txt = path.read_text()
        config = json.loads(txt)

        # Configure LLM
        GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
        self.model = ChatGoogleGenerativeAI(model=GEMINI_MODEL)

        # Storing in self
        try:
            self.cloff: Dict[str, List[int]] = {
                "tag": config["cloff"]["tag"],
                "text": config["cloff"]["text"],
            }
            self.human: Dict[str, List[int]] = {
                "tag": config["human"]["tag"],
                "text": config["human"]["text"],
            }
            self.showstats: bool = config["showstats"]
            self.special_keywords = ["clear", "exit", "reset", "history", "help"]
        except KeyError:
            raise ValueError("Invalid config.json")


CONFIG = Config()


class tutil:
    def to_RGB(text: str, c: List[int]) -> str:
        r, g, b = c
        # 8-bit color
        return f"\033[38;2;{r};{g};{b}m{text}\033[0;0m"

    def cloff_tag(text: str) -> str:
        return tutil.to_RGB(text, CONFIG.cloff["tag"])

    def cloff_text(text: str) -> str:
        return tutil.to_RGB(text, CONFIG.cloff["text"])

    def human_tag(text: str) -> str:
        return tutil.to_RGB(text, CONFIG.human["tag"])

    def human_text(text: str) -> str:
        return tutil.to_RGB(text, CONFIG.human["text"])

    def grey(text: str) -> str:
        return tutil.to_RGB(text, [179, 179, 179])

    def red(text: str) -> str:
        return tutil.to_RGB(text, [197, 15, 31])

    def green(text: str) -> str:
        return tutil.to_RGB(text, [22, 198, 12])

    def as_cloff(text: str = "") -> str:
        return f"{tutil.cloff_tag(' [cloff]')} {tutil.cloff_text(text)}"

    def as_system(text: str) -> str:
        return f"{tutil.grey('[system]')} {tutil.grey(text)}"

    def as_you(text: str = "") -> str:
        return f"{tutil.human_tag('   [you]')} {tutil.human_text(text)}"


class Convo:
    def __init__(self):
        self.first_of_cloff = "Hello! What would you like to know?"
        self.last_of_cloff = self.first_of_cloff

        self.reset()

    def append_as(self, text: str, MessageType: BaseMessage):
        self.history.append(MessageType(text))
        if type(MessageType) == AIMessage:
            self.last_of_cloff = text

    def reset(self):
        self.history: List[BaseMessage] = []
        self.append_as(
            "You are a chatbot named cloff. You answer human's queries as concisely as possible.",
            SystemMessage,
        )
        self.append_as("Hello! What would you like to know?", AIMessage)


def get_stats(start: float, tokens: int | None = None):
    if not CONFIG.showstats:
        return ""
    elapsed = time.time() - start
    s = tutil.grey(f"({time.time() - start:.2f}s")
    if tokens:
        tps = tokens / elapsed
        g = f"tps {tps:.2f}"
        if tps < 5:
            g = tutil.red(g)
        elif tps < 10:
            g = tutil.grey(g)
        else:
            g = tutil.green(g)
        s = s + tutil.grey(" | ") + g
    s += tutil.grey(")")
    return s


if __name__ == "__main__":
    CONFIG.convo = Convo()
    print(tutil.as_system("Welcome to cloff! Type 'help' for help."), end="\n\n")
    print(tutil.as_cloff(CONFIG.convo.first_of_cloff), end="\n\n")
    while True:
        try:
            user_input = input(tutil.as_you())
            print()

            if user_input in CONFIG.special_keywords:
                if user_input == "exit":
                    break
                elif user_input == "clear":
                    os.system("cls")
                    print(tutil.as_cloff(CONFIG.convo.last_of_cloff), end="\n\n")
                elif user_input == "reset":
                    CONFIG.convo.reset()
                    print(tutil.as_system("Conversation reset\n"))
                elif user_input == "history":
                    r = [tutil.grey("== Conversation History ==")]
                    for msg in CONFIG.convo.history:
                        if msg.type == "system":
                            r.append(tutil.as_system(msg.content))
                        if msg.type == "ai":
                            r.append(tutil.as_cloff(msg.content))
                        if msg.type == "human":
                            r.append(tutil.as_you(msg.content))
                    r.append(tutil.grey("== End of Conversation History =="))
                    r.append(tutil.grey(f"Messages: {len(CONFIG.convo.history)}"))
                    print("\n".join(r), end="\n\n")
                elif user_input == "help":
                    print(
                        tutil.as_system(
                            """Special Keywords:
clear - Clear the screen
exit - Exit the program
reset - Reset the conversation
history - Show the conversation history
help - Show this message
"""
                        )
                    )
                continue

            CONFIG.convo.append_as(user_input, HumanMessage)
            response = ""
            start = time.time()
            tokens = 0
            print(tutil.as_cloff(), end="")
            st = CONFIG.model.stream(CONFIG.convo.history)
            for i, msg in enumerate(st):
                for ch in msg.content:
                    print(tutil.cloff_text(ch), end="")
                    response += ch
                    # time.sleep(0.001)
            print(get_stats(start, tokens), end="\n\n")
            CONFIG.convo.append_as(response, AIMessage)
        except KeyboardInterrupt:
            print(f"""\n\n{tutil.as_system("Type 'exit' to exit.")}\n""")
            continue

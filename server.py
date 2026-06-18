import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"
BASE_DIR = os.path.dirname(__file__)
CHATS_DIR = os.path.join(BASE_DIR, "chats")

app = FastAPI()

# Load model once at startup, not per-request
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model.eval()
print("Model loaded. Server ready.")

# Server-side conversation state. Reset via /new-chat before each day's
# script begins; /chat appends to this list, uses the full accumulated
# history for generation, and auto-saves to disk after every turn.
conversation_state: dict = {
    "messages": [],
    "belief_id": None,
    "condition": None,
    "instance": None,
    "day": None,
}


class NewChatRequest(BaseModel):
    belief_id: str
    condition: str  # "confirming" or "contradicting"
    instance: str   # "treatment" or "baseline"
    day: int
    system_prompt: str = ""


class NewChatResponse(BaseModel):
    status: str
    messages: list


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    turn_count: int
    saved_to: str


@app.post("/new-chat", response_model=NewChatResponse)
def new_chat(request: NewChatRequest):
    """
    Reset the server-side conversation history and record this day's
    metadata. Call this once at the start of each day's 15-turn script,
    before the first /chat call. The metadata captured here is reused by
    every /chat call to auto-save the growing conversation.
    """
    conversation_state["messages"] = []
    conversation_state["belief_id"] = request.belief_id
    conversation_state["condition"] = request.condition
    conversation_state["instance"] = request.instance
    conversation_state["day"] = request.day

    if request.system_prompt:
        conversation_state["messages"].append(
            {"role": "system", "content": request.system_prompt}
        )
    return NewChatResponse(status="reset", messages=conversation_state["messages"])


def _save_current_chat() -> str:
    """
    Write the current accumulated conversation to chats/day_{N}_chat.json,
    tagged with the metadata captured at /new-chat time. Overwrites the
    same file on every call within a day, so the file always reflects the
    latest state of that day's conversation.
    """
    os.makedirs(CHATS_DIR, exist_ok=True)

    user_turns = [m for m in conversation_state["messages"] if m["role"] == "user"]
    assistant_turns = [m for m in conversation_state["messages"] if m["role"] == "assistant"]

    pairs = [
        {"prompt": u["content"], "response": a["content"]}
        for u, a in zip(user_turns, assistant_turns)
    ]

    entry = {
        "belief_id": conversation_state["belief_id"],
        "condition": conversation_state["condition"],
        "instance": conversation_state["instance"],
        "day": conversation_state["day"],
        "full_messages": conversation_state["messages"],
        "pairs": pairs,
        "turn_count": len(pairs),
    }

    day = conversation_state["day"]
    path = os.path.join(CHATS_DIR, f"day_{day}_chat.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2)

    return path


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Append the user's message to the accumulated conversation history,
    generate a response using the full history as context, append the
    model's response, save the updated conversation to disk, and return
    the response. Each call builds on every prior call since the last
    /new-chat reset. Requires /new-chat to have been called first, so
    day/belief/condition/instance metadata is available for saving.
    """
    if conversation_state["day"] is None:
        return ChatResponse(
            response="ERROR: /new-chat must be called before /chat to set day metadata.",
            turn_count=0,
            saved_to="",
        )

    conversation_state["messages"].append({"role": "user", "content": request.message})

    text = processor.apply_chat_template(
        conversation_state["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=0.95
        )

    response_text = processor.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    conversation_state["messages"].append({"role": "assistant", "content": response_text})

    saved_path = _save_current_chat()

    turn_count = sum(1 for m in conversation_state["messages"] if m["role"] == "user")
    return ChatResponse(response=response_text, turn_count=turn_count, saved_to=saved_path)


@app.get("/history")
def get_history():
    """Inspect the current accumulated conversation, useful for debugging mid-script."""
    return {"messages": conversation_state["messages"]}


@app.get("/health")
def health():
    return {"status": "ok"}
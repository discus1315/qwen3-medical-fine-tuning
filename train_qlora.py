import os
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)

from peft import LoraConfig, TaskType, get_peft_model
import swanlab


# ——————————————— 🔹 QLoRA 配置 ———————————————

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # 核心：启用 4-bit 加载
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",                 # QLoRA NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16      # 训练计算保持 bfloat16
)

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 1024

os.environ["SWANLAB_PROJECT"] = "qwen3-medical-QLoRA"


# ——————————————— 🔹 数据处理方法 ———————————————

def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            input_text = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}"

            messages.append({
                "instruction": PROMPT,
                "input": input_text,
                "output": output,
            })

    with open(new_path, "w", encoding="utf-8") as file:
        for msg in messages:
            file.write(json.dumps(msg, ensure_ascii=False) + "\n")


def process_func(example):
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(example["output"], add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ——————————————— 🔹 推理函数 ———————————————

def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "xpu"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ——————————————— 🔹 启动 SwanLab ———————————————

swanlab.login(api_key="[你的API Key]", save=True)

swanlab.config.update({
    "method": "QLoRA",
    "prompt": PROMPT,
    "max_length": MAX_LENGTH,
})


# ——————————————— 🔹 模型加载（核心 QLoRA） ———————————————

model_id = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config     # ← 核心 QLoRA 就在这里
)

model.enable_input_require_grads()


# ——————————————— 🔹 LoRA adapter 配置（正常 LoRA 流程） ———————————————

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,           # 你可以改成 8
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_cfg)


# ——————————————— 🔹 数据准备 ———————————————

train_path = "train.jsonl"
val_path = "val.jsonl"
train_fmt = "train_format.jsonl"
val_fmt = "val_format.jsonl"

if not os.path.exists(train_fmt):
    dataset_jsonl_transfer(train_path, train_fmt)
if not os.path.exists(val_fmt):
    dataset_jsonl_transfer(val_path, val_fmt)

train_df = pd.read_json(train_fmt, lines=True)
val_df = pd.read_json(val_fmt, lines=True)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_ds = train_ds.map(process_func, remove_columns=train_df.columns)
val_ds = val_ds.map(process_func, remove_columns=val_df.columns)


# ——————————————— 🔹 训练配置 ———————————————

args = TrainingArguments(
    output_dir="./output/qlora_Qwen3",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    eval_strategy="interval",
    eval_steps=100,
    logging_steps=10,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    save_steps=400,
    save_on_each_node=True,
    report_to="swanlab",
    run_name="qwen3-medical-qlora"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)


# ——————————————— 🔹 训练启动 ———————————————

trainer.train()


# ——————————————— 🔹 推理测试 ———————————————

test_df = pd.read_json(val_fmt, lines=True)[:3]

test_outputs = []

for _, row in test_df.iterrows():
    messages = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row["input"]},
    ]

    response = predict(messages, model, tokenizer)

    text_log = f"""
    Question: {row['input']}

    LLM: {response}
    """

    print(text_log)
    test_outputs.append(swanlab.Text(text_log))

swanlab.log({"Prediction": test_outputs})
swanlab.finish()

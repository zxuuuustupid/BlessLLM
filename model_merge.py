import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("============================================================")
print("开始在 CPU 上合并模型...")
print("============================================================")

# --- 1. 路径设置 (请确保和你的 shell 脚本一致) ---
# 原始基础模型的路径
base_model_path = "./model_temp/Shanghai_AI_Laboratory/internlm2-chat-7b"
# 经过 pth_to_hf 转换后的、包含 adapter_config.json 的文件夹路径
lora_adapter_path = "./hf"
# 你希望保存合并后新模型的路径
save_path = "./merge"

# --- 2. 加载模型和分词器 ---
print(f"正在从 '{base_model_path}' 加载基础模型到 CPU...")
# 以 CPU 模式加载模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='cpu'  # 关键！指定在 CPU 上加载
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print(f"正在从 '{lora_adapter_path}' 加载 LoRA 适配器...")
# 加载 LoRA 适配器
model = PeftModel.from_pretrained(model, lora_adapter_path, device_map='cpu')

# --- 3. 合并并保存 ---
print("正在合并模型权重...")
# 合并权重
model = model.merge_and_unload()

print(f"正在将合并后的模型保存到: '{save_path}'")
# 创建保存目录
os.makedirs(save_path, exist_ok=True)
# 保存模型和分词器
model.save_pretrained(save_path, max_shard_size='2GB') # 添加分片保存
tokenizer.save_pretrained(save_path)

print("============================================================")
print("✅ 完成！模型已在 CPU 上成功合并并保存。")
print("============================================================")
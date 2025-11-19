import os
import time
import re
from datasets import load_dataset
from openai import OpenAI

# ================= 配置区域 =================
# 这里以 DeepSeek 为例，如果你用其他厂商，修改 base_url 和 api_key
API_KEY = "sk-4f25b70eb25f45ce87ed63f0251dabcb"  # 替换为你的 API Key
BASE_URL = "https://api.deepseek.com"    # DeepSeek 的地址
MODEL_NAME = "deepseek-chat"             # 或者 "deepseek-coder"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 辅助函数 =================

def extract_answer(text):
    """
    从模型输出或数据集中提取数字答案。
    GSM8K 的标准答案通常在 '#### ' 之后。
    模型输出我们尝试找最后一个数字。
    """
    # 尝试找到 #### 后的内容 (用于处理数据集的标准答案)
    if "####" in text:
        return text.split("####")[1].strip()
    
    # 对于模型输出，我们使用简单的正则寻找最后一个数字
    # 注意：这只是一个简单的提取器，实际项目中可能需要更复杂的正则来处理 "1,000" 或 "$5"
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

def run_baseline(limit=5):
    """
    运行基线测试 (n=1)
    limit: 为了测试代码，我们默认只跑 5 条数据。正式跑的时候设为 None。
    """
    print("正在加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if limit:
        dataset = dataset.select(range(limit))
        print(f"为了测试，仅使用前 {limit} 条数据。")

    correct_count = 0
    total_latency = 0
    total_tokens = 0
    
    results = []

    print(f"{'ID':<5} | {'真实值':<10} | {'预测值':<10} | {'耗时(s)':<8} | {'Tokens':<8} | {'结果'}")
    print("-" * 70)

    for i, item in enumerate(dataset):
        question = item['question']
        ground_truth_str = item['answer']
        ground_truth = extract_answer(ground_truth_str)
        
        # --- 1. 构造 Prompt (CoT) ---
        messages = [
            {"role": "system", "content": "你是一个数学专家。请一步步思考解决问题。最后将你的最终答案放在 '#### ' 之后。例如：#### 42"},
            {"role": "user", "content": question}
        ]

        # --- 2. 调用 API 并计时 (Latency) ---
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,  # 基线测试通常设为 0 以保证确定性
            )
            end_time = time.time()
            
            # --- 3. 收集数据 ---
            latency = end_time - start_time
            output_text = response.choices[0].message.content
            
            # 计算 Token (Cost)
            usage = response.usage
            tokens = usage.total_tokens if usage else 0
            
            # --- 4. 验证答案 (Quality) ---
            prediction = extract_answer(output_text)
            
            # 简单的字符串比对 (实际可能需要浮点数容差)
            is_correct = (prediction == ground_truth)
            
            if is_correct:
                correct_count += 1
                
            # 更新统计
            total_latency += latency
            total_tokens += tokens
            
            status = "✅" if is_correct else "❌"
            print(f"{i:<5} | {ground_truth:<10} | {str(prediction):<10} | {latency:<8.2f} | {tokens:<8} | {status}")

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    # ================= 最终报告 =================
    num_samples = len(dataset)
    avg_latency = total_latency / num_samples
    avg_tokens = total_tokens / num_samples
    accuracy = (correct_count / num_samples) * 100

    print("\n" + "="*30)
    print("   C.1 BASELINE RESULTS (n=1)")
    print("="*30)
    print(f"📊 Quality (准确率): {accuracy:.2f}%")
    print(f"⏱️ Latency (平均耗时): {avg_latency:.2f} s/题")
    print(f"💰 Cost    (平均开销): {avg_tokens:.1f} tokens/题")
    print("="*30)

if __name__ == "__main__":
    run_baseline(limit=5) # 先跑5个试试水
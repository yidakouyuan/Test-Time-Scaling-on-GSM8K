import os
import time
import re
from datasets import load_dataset
from openai import OpenAI
from collections import Counter # 1. 引入 Counter 用于投票

# ================= 配置区域 =================
# 确保这里填写了你正确的 Key
API_KEY = "sk-4f25b70eb25f45ce87ed63f0251dabcb"  # 替换为你的 API Key
BASE_URL = "https://api.deepseek.com"    # DeepSeek 的地址 (或其他)
MODEL_NAME = "deepseek-chat"             

N_SAMPLES = 5 # 2. C.1 关键参数："思考步骤" (n=5)

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 辅助函数 =================

def extract_answer(text):
    """
    从模型输出或数据集中提取数字答案。
    """
    if "####" in text:
        return text.split("####")[1].strip()
    
    # 对于模型输出，我们使用简单的正则寻找最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

def run_self_consistency(limit=100): # 默认跑 100 个样本
    """
    运行 Self-Consistency 测试 (n=N_SAMPLES)
    """
    print(f"正在加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if limit:
        dataset = dataset.select(range(limit))
        print(f"为了测试，仅使用前 {limit} 条数据。")

    correct_count = 0
    total_latency = 0
    total_tokens = 0
    
    print(f"--- 运行 Self-Consistency (n={N_SAMPLES}) ---")
    print(f"{'ID':<5} | {'真实值':<10} | {'预测值':<10} | {'耗时(s)':<8} | {'Tokens':<8} | {'结果'} | {'投票详情'}")
    print("-" * 80)

    for i, item in enumerate(dataset):
        question = item['question']
        ground_truth_str = item['answer']
        ground_truth = extract_answer(ground_truth_str)
        
        messages = [
            {"role": "system", "content": "你是一个数学专家。请一步步思考解决问题。最后将你的最终答案放在 '#### ' 之后。例如：#### 42"},
            {"role": "user", "content": question}
        ]

        predictions = []
        question_total_tokens = 0
        
        # 3. 计时 (Latency) - 包含 N 次调用的总时间
        start_time = time.time()
        
        # 4. 内循环 N_SAMPLES 次
        for _ in range(N_SAMPLES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,  # 关键！设为非 0 以获得 "Diverse thinking"
                )
                
                # 累加成本 (Cost)
                if response.usage:
                    question_total_tokens += response.usage.total_tokens
                
                # 提取答案并加入投票列表
                output_text = response.choices[0].message.content
                pred = extract_answer(output_text)
                if pred:
                    predictions.append(pred)
                    
            except Exception as e:
                print(f"Error processing item {i}, sample {_}: {e}")
                
        end_time = time.time()
        question_latency = end_time - start_time # 该问题总耗时

        # 5. 投票 (Self-Consistency)
        if not predictions:
            final_prediction = None # 如果 N 次都失败了
            vote_details = "No votes"
        else:
            # 使用 Counter 找到票数最多的答案
            count = Counter(predictions)
            final_prediction = count.most_common(1)[0][0]
            vote_details = str(dict(count)) # 记录投票详情

        # 6. 验证答案 (Quality)
        is_correct = (final_prediction == ground_truth)
        
        if is_correct:
            correct_count += 1
            
        # 更新总统计
        total_latency += question_latency
        total_tokens += question_total_tokens
        
        status = "✅" if is_correct else "❌"
        print(f"{i:<5} | {ground_truth:<10} | {str(final_prediction):<10} | {question_latency:<8.2f} | {question_total_tokens:<8} | {status} | {vote_details}")

    # ================= 最终报告 =================
    num_samples = len(dataset)
    avg_latency = total_latency / num_samples
    avg_tokens = total_tokens / num_samples
    accuracy = (correct_count / num_samples) * 100

    print("\n" + "="*30)
    print(f"   C.1 S-C RESULTS (n={N_SAMPLES})")
    print("="*30)
    print(f"📊 Quality (准确率): {accuracy:.2f}%")
    print(f"⏱️ Latency (平均耗时): {avg_latency:.2f} s/题")
    print(f"💰 Cost    (平均开销): {avg_tokens:.1f} tokens/题")
    print("="*30)

if __name__ == "__main__":
    # 你可以修改 limit 来控制跑多少数据
    # 跑 10 个样本试试水
    #run_self_consistency(limit=10) 
    
    # 跑 100 个样本用于你的表格
     run_self_consistency(limit=100)
    
    # 最终跑所有数据 (如果时间和 API 额度允许)
    # run_self_consistency(limit=None)
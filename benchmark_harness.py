
import os
import time
import re
import json
from datasets import load_dataset
from openai import OpenAI
from collections import Counter

# ==========================================================
# ============     你的“唯一”配置区域     ================
# ==========================================================
# 你只需要修改这两个值！

N_SAMPLES = 3   # 1 = 运行基线 (Baseline)
                # 3 = 运行 S-C (n=3)
                # 5 = 运行 S-C (n=5)

LIMIT = 100     # 跑 100 条数据。 (测试时可以改成 5)

# ==========================================================
# ============     API 密钥 (填一次就行)     ===============
# ==========================================================
API_KEY = "sk-4f25b70eb25f45ce87ed63f0251dabcb" # 替换为你的 API Key
BASE_URL = "https://api.deepseek.com"    
MODEL_NAME = "deepseek-chat"             
# ==========================================================


def extract_answer(text):
    """
    从模型输出或数据集中提取数字答案。
    """
    if "####" in text:
        return text.split("####")[1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

def run_benchmark(n_samples, limit):
    """
    运行基准测试。
    n_samples (int): "思考步骤" (n=1 是基线, n>1 是 S-C)
    limit (int): 跑多少条数据
    """
    
    # 1. 自动设置策略和文件名
    strategy = "baseline" if n_samples == 1 else "sc"
    RESULTS_JSON_FILE = f"{strategy}_results_n{n_samples}_limit{limit}.json"
    SUMMARY_TXT_FILE = f"{strategy}_summary_n{n_samples}_limit{limit}.txt"
    
    # 2. 自动设置 Temperature
    # n=1 (基线) 时, T=0 确保确定性
    # n>1 (S-C) 时, T=0.7 确保多样性
    temperature = 0.0 if n_samples == 1 else 0.7

    print("="*50)
    print(f"--- 启动基准测试 ---")
    print(f"策略: {strategy} (n={n_samples})")
    print(f"数据: {limit} 条")
    print(f"Temp: {temperature}")
    print(f"JSON 保存至: {RESULTS_JSON_FILE}")
    print(f"总结 保存至: {SUMMARY_TXT_FILE}")
    print("="*50)

    # 初始化客户端
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # --- 加载数据集 ---
    print(f"正在加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit:
        dataset = dataset.select(range(limit))
    
    correct_count = 0
    total_latency = 0
    total_tokens = 0
    
    detailed_results = [] # 列表来保存所有详细结果

    print(f"\n{'ID':<5} | {'真实值':<10} | {'预测值':<10} | {'耗时(s)':<8} | {'Tokens':<8} | {'结果'} | {'投票详情'}")
    print("-" * 80)

    # --- 开始循环处理数据 ---
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
        raw_outputs = [] # 保存原始的 N 次输出
        
        start_time = time.time()
        
        for _ in range(n_samples): # 内循环 N_SAMPLES 次
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature, # 自动设置的 temperature
                )
                
                if response.usage:
                    question_total_tokens += response.usage.total_tokens
                
                output_text = response.choices[0].message.content
                raw_outputs.append(output_text) # 保存原始输出
                
                pred = extract_answer(output_text)
                if pred:
                    predictions.append(pred)
                    
            except Exception as e:
                print(f"Error processing item {i}, sample {_}: {e}")
                
        end_time = time.time()
        question_latency = end_time - start_time

        # 投票
        if not predictions:
            final_prediction = None
            vote_details = "No votes"
        else:
            count = Counter(predictions)
            final_prediction = count.most_common(1)[0][0]
            vote_details = str(dict(count))

        # 验证
        is_correct = (final_prediction == ground_truth)
        
        if is_correct:
            correct_count += 1
            
        total_latency += question_latency
        total_tokens += question_total_tokens
        
        status = "✅" if is_correct else "❌"
        print(f"{i:<5} | {ground_truth:<10} | {str(final_prediction):<10} | {question_latency:<8.2f} | {question_total_tokens:<8} | {status} | {vote_details}")

        # 将这道题的详细结果存入列表
        detailed_results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "final_prediction": final_prediction,
            "is_correct": is_correct,
            "latency_s": question_latency,
            "total_tokens": question_total_tokens,
            "vote_details": vote_details,
            "n_samples": n_samples,
            "raw_outputs": raw_outputs # 保存原始回答，便于分析
        })

    # ================= 最终报告和保存 =================
    num_items = len(detailed_results)
    avg_latency = total_latency / num_items if num_items > 0 else 0
    avg_tokens = total_tokens / num_items if num_items > 0 else 0
    accuracy = (correct_count / num_items) * 100 if num_items > 0 else 0

    # --- 准备总结报告 ---
    summary_content = f"""
C.1 {strategy.upper()} RESULTS (n={n_samples}, limit={limit})
-----------------------------------
📊 Quality (准确率): {accuracy:.2f}%
⏱️ Latency (平均耗时): {avg_latency:.2f} s/题
💰 Cost    (平均开销): {avg_tokens:.1f} tokens/题
-----------------------------------
Total items processed: {num_items}
Total correct: {correct_count}
"""
    # --- 打印到屏幕 ---
    print("\n" + "="*30)
    print(summary_content.strip())
    print("="*30)

    # --- 保存到文件 ---
    try:
        with open(RESULTS_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=4, ensure_ascii=False)
        print(f"✅ 详细结果已保存到: {RESULTS_JSON_FILE}")
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")

    try:
        with open(SUMMARY_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"✅ 总结报告已保存到: {SUMMARY_TXT_FILE}")
    except Exception as e:
        print(f"❌ 保存 TXT 失败: {e}")


if __name__ == "__main__":
    # 脚本会直接读取顶部的配置来运行
    run_benchmark(n_samples=N_SAMPLES, limit=LIMIT)
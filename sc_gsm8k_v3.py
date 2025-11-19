import os
import time
import re
import json # 1. 导入 json 库
from datasets import load_dataset
from openai import OpenAI
from collections import Counter

# ================= 配置区域 =================
API_KEY = "sk-4f25b70eb25f45ce87ed63f0251dabcb"  # 替换为你的 API Key
BASE_URL = "https://api.deepseek.com"    
MODEL_NAME = "deepseek-chat"             

N_SAMPLES = 5 
CURRENT_LIMIT = 10 # 2. 将 limit 设为变量，方便命名

# 3. 设置输出文件名
RESULTS_JSON_FILE = f"sc_results_n{N_SAMPLES}_limit{CURRENT_LIMIT}.json"
SUMMARY_TXT_FILE = f"sc_summary_n{N_SAMPLES}_limit{CURRENT_LIMIT}.txt"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 辅助函数 =================

def extract_answer(text):
    if "####" in text:
        return text.split("####")[1].strip()
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

def run_self_consistency(limit=100):
    print(f"正在加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if limit:
        dataset = dataset.select(range(limit))
        print(f"为了测试，仅使用前 {limit} 条数据。")

    correct_count = 0
    total_latency = 0
    total_tokens = 0
    
    detailed_results = [] # 4. 创建一个列表来保存所有详细结果

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
        raw_outputs = [] # 用于保存原始的 N 次输出
        
        start_time = time.time()
        
        for _ in range(N_SAMPLES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
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

        # 5. 将这道题的详细结果存入列表
        detailed_results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "final_prediction": final_prediction,
            "is_correct": is_correct,
            "latency_s": question_latency,
            "total_tokens": question_total_tokens,
            "vote_details": vote_details,
            "n_samples": N_SAMPLES,
            "raw_outputs": raw_outputs # 保存原始回答，便于分析 
        })

    # ================= 最终报告 =================
    num_samples = len(dataset)
    avg_latency = total_latency / num_samples if num_samples > 0 else 0
    avg_tokens = total_tokens / num_samples if num_samples > 0 else 0
    accuracy = (correct_count / num_samples) * 100 if num_samples > 0 else 0

    # --- 打印到屏幕 ---
    print("\n" + "="*30)
    print(f"   C.1 S-C RESULTS (n={N_SAMPLES}, limit={limit})")
    print("="*30)
    print(f"📊 Quality (准确率): {accuracy:.2f}%")
    print(f"⏱️ Latency (平均耗时): {avg_latency:.2f} s/题")
    print(f"💰 Cost    (平均开销): {avg_tokens:.1f} tokens/题")
    print("="*30)

    # 6. 将详细结果写入 JSON 文件
    try:
        with open(RESULTS_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=4, ensure_ascii=False)
        print(f"✅ 详细结果已保存到: {RESULTS_JSON_FILE}")
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")

    # 7. 将总结报告写入 TXT 文件
    summary_content = f"""
C.1 S-C RESULTS (n={N_SAMPLES}, limit={limit})
-----------------------------------
📊 Quality (准确率): {accuracy:.2f}%
⏱️ Latency (平均耗时): {avg_latency:.2f} s/题
💰 Cost    (平均开销): {avg_tokens:.1f} tokens/题
-----------------------------------
Total items processed: {num_samples}
Total correct: {correct_count}
"""
    try:
        with open(SUMMARY_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"✅ 总结报告已保存到: {SUMMARY_TXT_FILE}")
    except Exception as e:
        print(f"❌ 保存 TXT 失败: {e}")


if __name__ == "__main__":
    # 运行 10 条数据进行快速测试
    run_self_consistency(limit=CURRENT_LIMIT) 
    
    # 最终运行时，修改 CURRENT_LIMIT = 100 (或更多)
    # 然后重新运行脚本
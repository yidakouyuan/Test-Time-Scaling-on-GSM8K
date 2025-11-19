import os
import time
import re
import json
from datasets import load_dataset
from openai import OpenAI
from collections import Counter

# ==========================================================
# ============     "验证器"配置区域     ================
# ==========================================================
# 我们将生成 3 个"解法"，然后用 1 个"验证器"
N_SOLUTIONS = 3   # 生成 3 个候选解法
LIMIT = 100       # 跑 100 条数据

# ==========================================================
# ============     API 密钥 (填一次就行)     ===============
# ==========================================================
API_KEY = "sk-4f25b70eb25f45ce87ed63f0251dabcb"  # 替换为你的 API Key
BASE_URL = "https://api.deepseek.com"    
MODEL_NAME = "deepseek-chat"             
# ==========================================================


def extract_answer(text):
    """
    从模型输出或数据集中提取数字答案。
    """
    # 尝试从 "#### 123" 格式中提取
    if "####" in text:
        match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if match:
            return match.group(1)
            
    # 如果找不到，就从全文中找最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return None

def run_verifier_benchmark(n_solutions, limit):
    
    # --- 1. 设置 ---
    strategy = "verifier"
    RESULTS_JSON_FILE = f"{strategy}_results_n{n_solutions}_limit{limit}.json"
    SUMMARY_TXT_FILE = f"{strategy}_summary_n{n_solutions}_limit{limit}.txt"
    
    print("="*50)
    print(f"--- 启动 Verifier (验证器) 基准测试 ---")
    print(f"策略: {strategy} (n={n_solutions} 解法 + 1 验证器)")
    print(f"数据: {limit} 条")
    print(f"JSON 保存至: {RESULTS_JSON_FILE}")
    print(f"总结 保存至: {SUMMARY_TXT_FILE}")
    print("="*50)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print(f"正在加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit:
        dataset = dataset.select(range(limit))
    
    correct_count = 0
    total_latency = 0
    total_tokens = 0
    
    detailed_results = [] 

    print(f"\n{'ID':<5} | {'真实值':<10} | {'预测值':<10} | {'耗时(s)':<8} | {'Tokens':<8} | {'结果'}")
    print("-" * 80)

    # --- 2. 开始循环处理数据 ---
    for i, item in enumerate(dataset):
        question = item['question']
        ground_truth_str = item['answer']
        ground_truth = extract_answer(ground_truth_str)
        
        # --- 步骤 A: 生成 N 个解法 (Solutions) ---
        solution_messages = [
            {"role": "system", "content": "你是一个数学专家。请一步步思考解决问题。最后将你的最终答案放在 '#### ' 之后。例如：#### 42"},
            {"role": "user", "content": question}
        ]
        
        raw_solutions = []
        question_total_tokens = 0
        start_time = time.time()
        
        for _ in range(n_solutions):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=solution_messages,
                    temperature=0.7, # 使用高 temperature 获得多样性
                )
                if response.usage:
                    question_total_tokens += response.usage.total_tokens
                raw_solutions.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error in Solution generation {i}: {e}")
        
        # --- 步骤 B: 构建 "验证器" (Verifier) Prompt ---
        verifier_prompt = f"以下是一个数学问题：\n--- 问题 ---\n{question}\n\n---"
        verifier_prompt += f"\n这里有 {n_solutions} 个不同的解法尝试。请仔细检查它们，找出最正确、最合理的答案，并解释原因。\n\n"
        
        for j, solution_text in enumerate(raw_solutions):
            verifier_prompt += f"--- 解法 {j+1} ---\n{solution_text}\n\n"
            
        verifier_prompt += "--- 验证 ---\n请评估上述所有解法。哪个解法是正确的？为什么？最后，请提供正确的最终答案，并将其放在 '####' 之后。\n"
        
        verifier_messages = [
            {"role": "system", "content": "你是一个严谨的数学核查员(Verifier)。你的工作是评估多个解法，找出最正确的一个，并给出最终答案。"},
            {"role": "user", "content": verifier_prompt}
        ]

        # --- 步骤 C: 调用 "验证器" ---
        try:
            verifier_response = client.chat.completions.create(
                model=MODEL_NAME, # 可以用更强的模型
                messages=verifier_messages,
                temperature=0.0, # 验证器必须是确定性的
            )
            
            if verifier_response.usage:
                question_total_tokens += verifier_response.usage.total_tokens
            
            verifier_output = verifier_response.choices[0].message.content
            final_prediction = extract_answer(verifier_output)
            
        except Exception as e:
            print(f"Error in Verifier call {i}: {e}")
            final_prediction = None
            verifier_output = f"Verifier Failed: {e}"

        end_time = time.time()
        question_latency = end_time - start_time

        # --- 步骤 D: 评估和记录 ---
        is_correct = (final_prediction == ground_truth)
        
        if is_correct:
            correct_count += 1
            
        total_latency += question_latency
        total_tokens += question_total_tokens
        
        status = "✅" if is_correct else "❌"
        print(f"{i:<5} | {ground_truth:<10} | {str(final_prediction):<10} | {question_latency:<8.2f} | {question_total_tokens:<8} | {status}")

        detailed_results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "final_prediction": final_prediction,
            "is_correct": is_correct,
            "latency_s": question_latency,
            "total_tokens": question_total_tokens,
            "n_solutions": n_solutions,
            "raw_solutions": raw_solutions, # 保存所有原始解法
            "verifier_output": verifier_output # 保存验证器的输出
        })

    # ================= 最终报告和保存 =================
    num_items = len(detailed_results)
    avg_latency = total_latency / num_items if num_items > 0 else 0
    avg_tokens = total_tokens / num_items if num_items > 0 else 0
    accuracy = (correct_count / num_items) * 100 if num_items > 0 else 0

    summary_content = f"""
C.1 {strategy.upper()} RESULTS (n={n_solutions}+1, limit={limit})
-----------------------------------
📊 Quality (准确率): {accuracy:.2f}%
⏱️ Latency (平均耗时): {avg_latency:.2f} s/题
💰 Cost    (平均开销): {avg_tokens:.1f} tokens/题
-----------------------------------
Total items processed: {num_items}
Total correct: {correct_count}
"""
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
    run_verifier_benchmark(n_solutions=N_SOLUTIONS, limit=LIMIT)
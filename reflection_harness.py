import os
import time
import re
import json
from datasets import load_dataset
from openai import OpenAI

# ==========================================================
# ============     "反思循环"配置区域     ================
# ==========================================================
LIMIT = 100     # 跑 100 条数据。 (测试时可以改成 5)

# ==========================================================
# ============     API 密钥 (填一次就行)     ===============
# ==========================================================
API_KEY = "sk-4f25b70eb25f45ce87ed63f0251dabcb" # 替换为你的 API Key
BASE_URL = "https://api.deepseek.com"    
MODEL_NAME = "deepseek-chat"             
# ==========================================================


def flexible_extract_answer(text):
    """
    更强大的答案提取器：
    1. 寻找 '#### 123'
    2. 寻找带分数的 LaTeX '\\(\\frac{400}{11}\\)'
    3. 寻找最后的数字 (处理 16.00)
    """
    try:
        # 1. 寻找 '#### 123'
        if "####" in text:
            match = re.search(r'####\s*(-?\d+\.?\d*)', text)
            if match:
                return float(match.group(1)) # 转换为浮点数

        # 2. 寻找分数 LaTeX: \frac{A}{B}
        frac_match = re.search(r'\\frac{(\d+)}{(\d+)}', text)
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            return numerator / denominator
            
        # 3. 寻找最后的数字 (处理 16.00)
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        if numbers:
            return float(numbers[-1])
            
    except Exception:
        return None # 无法解析
    return None

def run_reflection_benchmark(limit):
    
    # 1. 设置策略和文件名
    strategy = "reflection"
    RESULTS_JSON_FILE = f"{strategy}_results_limit{limit}.json"
    SUMMARY_TXT_FILE = f"{strategy}_summary_limit{limit}.txt"
    
    print("="*50)
    print(f"--- 启动 Reflection (反思循环) 基准测试 ---")
    print(f"策略: {strategy} (1x Generate, 1x Reflect, 1x Re-Generate)")
    print(f"数据: {limit} 条")
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

    print(f"\n{'ID':<5} | {'真实值':<10} | {'预测值':<10} | {'耗时(s)':<8} | {'Tokens':<8} | {'结果'}")
    print("-" * 80)

    # --- 开始循环处理数据 ---
    for i, item in enumerate(dataset):
        question = item['question']
        ground_truth_str = item['answer']
        # 使用强大的提取器处理真实答案
        ground_truth = flexible_extract_answer(ground_truth_str)
        
        question_total_tokens = 0
        start_time = time.time()
        
        raw_solution_1 = ""
        reflection_output = ""
        final_solution_output = ""
        
        try:
            # --- 步骤 1: 生成 (Generate) ---
            solution_messages = [
                {"role": "system", "content": "你是一个数学专家。请一步步思考解决问题。最后将你的最终答案放在 '#### ' 之后。例如：#### 42"},
                {"role": "user", "content": question}
            ]
            response_1 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=solution_messages,
                temperature=0.7, # 第一次尝试允许一点多样性
            )
            raw_solution_1 = response_1.choices[0].message.content
            if response_1.usage:
                question_total_tokens += response_1.usage.total_tokens

            # --- 步骤 2: 反思 (Reflect) ---
            reflection_prompt = f"你是一名助手，任务是批判性地评估一个数学解法。原始问题是：'{question}'。 提供的解法是：'{raw_solution_1}'。请不要自己解题，而是分析提供的解法，指出其中可能的逻辑缺陷、计算错误或假设问题。如果解法看起来正确，就说明它正确。"
            reflection_messages = [
                {"role": "system", "content": "你是一个严谨的数学评估员。"},
                {"role": "user", "content": reflection_prompt}
            ]
            response_2 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=reflection_messages,
                temperature=0.0, # 反思需要严谨
            )
            reflection_output = response_2.choices[0].message.content
            if response_2.usage:
                question_total_tokens += response_2.usage.total_tokens

            # --- 步骤 3: 改进 (Re-Generate) ---
            final_prompt = f"你是一个数学专家。你需要解决以下问题：'{question}'。你之前的第一遍尝试是：'{raw_solution_1}'。对你第一遍尝试的批判性反思是：'{reflection_output}'。请结合你的第一遍尝试和这份反思，给出一个改进后的、最终的、最正确的答案。"
            final_solution_messages = [
                {"role": "system", "content": "你是一个会吸取教训并改进答案的数学专家。"},
                {"role": "user", "content": final_prompt}
            ]
            response_3 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=final_solution_messages,
                temperature=0.0, # 最终答案需要确定性
            )
            final_solution_output = response_3.choices[0].message.content
            if response_3.usage:
                question_total_tokens += response_3.usage.total_tokens

        except Exception as e:
            print(f"Error processing item {i}: {e}")
            
        end_time = time.time()
        question_latency = end_time - start_time
        
        # --- 评估 ---
        final_prediction = flexible_extract_answer(final_solution_output)
        
        is_correct = False
        if ground_truth is not None and final_prediction is not None:
            if abs(ground_truth - final_prediction) < 1e-5: # 浮点数比较
                is_correct = True
        
        if is_correct:
            correct_count += 1
            
        total_latency += question_latency
        total_tokens += question_total_tokens
        
        status = "✅" if is_correct else "❌"
        print(f"{i:<5} | {str(ground_truth):<10} | {str(final_prediction):<10} | {question_latency:<8.2f} | {question_total_tokens:<8} | {status}")

        detailed_results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "final_prediction": final_prediction,
            "is_correct": is_correct,
            "latency_s": question_latency,
            "total_tokens": question_total_tokens,
            "raw_solution_1": raw_solution_1,
            "reflection_output": reflection_output,
            "final_solution_output": final_solution_output
        })

    # ================= 最终报告和保存 =================
    num_items = len(detailed_results)
    avg_latency = total_latency / num_items if num_items > 0 else 0
    avg_tokens = total_tokens / num_items if num_items > 0 else 0
    accuracy = (correct_count / num_items) * 100 if num_items > 0 else 0

    summary_content = f"""
C.1 {strategy.upper()} RESULTS (n=1+1+1, limit={limit})
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
    run_reflection_benchmark(limit=LIMIT)
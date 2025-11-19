import json
import re

# --- 新增：这是生成的报告文件名 ---
FINAL_TABLE_FILE = "FINAL_REPORT_TABLE.txt"
# ---

def super_flexible_extract_answer(text):
    """
    终极答案提取器：
    1. 优先寻找 \boxed{...}
    2. 其次寻找 #### ...
    3. 最后寻找文末的数字
    """
    try:
        # 确保输入是字符串
        text = str(text)

        # 1. 优先寻找 \boxed{...} (例如: \boxed{18})
        box_match = re.search(r'\\boxed{(\d+\.?\d*)}', text)
        if box_match:
            return float(box_match.group(1))

        # 2. 其次寻找 #### ... (例如: #### 12)
        final_answer_match = re.search(r'####\s*(-?\d+\.?\d*)', text)
        if final_answer_match:
            return float(final_answer_match.group(1))

        # 3. 寻找分数 LaTeX: \frac{A}{B}
        frac_match = re.search(r'\\frac{(\d+)}{(\d+)}', text)
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            if denominator == 0:
                return None
            return numerator / denominator
            
        # 4. 降级：寻找最后的数字
        numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        if numbers:
            return float(numbers[-1])
            
    except Exception:
        return None # 无法解析
    return None

def reevaluate_json(filepath, answer_key):
    """
    读取一个 JSON 日志文件，并用终极提取器重新计分。
    answer_key: 是 "final_prediction" 还是 "final_solution_output"
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filepath}")
        # 返回 (准确率, 耗时, 开销, 总数, 正确数)
        return 0, 0, 0, 0, 0 

    total_count = 0
    correct_count = 0
    total_latency = 0
    total_tokens = 0

    for item in data:
        total_count += 1
        total_latency += item['latency_s']
        total_tokens += item['total_tokens']
        
        # --- FIX (修正点) ---
        gt_text = str(item['ground_truth'])
        ground_truth = super_flexible_extract_answer(gt_text)
        
        prediction_text = item.get(answer_key, "")
        prediction = super_flexible_extract_answer(prediction_text)
        # --- FIX END ---

        if ground_truth is not None and prediction is not None:
            if abs(ground_truth - prediction) < 1e-5: 
                correct_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    avg_latency = total_latency / total_count if total_count > 0 else 0
    avg_tokens = total_tokens / total_count if total_count > 0 else 0

    report_str = f"""
--- 重新评估: {filepath} ---
📊 真实准确率: {accuracy:.2f}% ({correct_count} / {total_count})
⏱️ 平均耗时: {avg_latency:.2f} s/题
💰 平均开销: {avg_tokens:.1f} tokens/题
"""
    print(report_str)
    
    # 返回所有需要的数据
    return accuracy, avg_latency, avg_tokens, total_count, correct_count

if __name__ == "__main__":
    print("--- 运行最终真实准确率评估 ---")
    
    # 用于保存所有输出的列表
    output_lines = ["--- 运行最终真实准确率评估 ---\n"]
    
    print("\n--- 读取已修正的 Baseline 和 S-C 结果 ---")
    output_lines.append("\n--- 读取已修正的 Baseline 和 S-C 结果 ---\n")
    
    # 你上次运行 reevaluate_results.py 的结果是可信的
    # (注意：这些是你上次运行 reevaluate_results.py 时的结果，不是 verifier 或 reflection 的)
    acc_n1, lat_n1, cost_n1 = 98.00, 8.33, 296.4
    acc_n3, lat_n3, cost_n3 = 97.00, 26.34, 896.9
    acc_n5, lat_n5, cost_n5 = 97.00, 42.10, 1476.5
    
    print(f"Baseline (n=1): {acc_n1:.2f}%")
    print(f"S-C (n=3): {acc_n3:.2f}%")
    print(f"S-C (n=5): {acc_n5:.2f}%")
    output_lines.append(f"Baseline (n=1): {acc_n1:.2f}%\n")
    output_lines.append(f"S-C (n=3): {acc_n3:.2f}%\n")
    output_lines.append(f"S-C (n=5): {acc_n5:.2f}%\n")
    
    # 2. 评估 Verifier (使用 verifier_output 键)
    acc_ver, lat_ver, cost_ver, _, _ = reevaluate_json("verifier_results_n3_limit100.json", "verifier_output")
    
    # 3. 评估 Reflection (使用 final_solution_output 键)
    acc_ref, lat_ref, cost_ref, _, _ = reevaluate_json("reflection_results_limit100.json", "final_solution_output")

    
    # --- 构建最终表格字符串 ---
    table_header = "\n\n--- 最终修正后的 3D 边界表格 ---\n"
    table_divider = f"|{'-'*24}|{'-'*17}|{'-'*17}|{'-'*17}|\n"
    table = table_header
    table += f"| {'策略':<22} | {'📊 真实准确率':<15} | {'⏱️ Latency (s)':<15} | {'💰 Cost (tokens)':<15} |\n"
    table += table_divider
    table += f"| {'Baseline (n=1)':<22} | {acc_n1:<15.2f}% | {lat_n1:<15.2f} | {cost_n1:<15.1f} |\n"
    table += f"| {'S-C (n=3)':<22} | {acc_n3:<15.2f}% | {lat_n3:<15.2f} | {cost_n3:<15.1f} |\n"
    table += f"| {'S-C (n=5)':<22} | {acc_n5:<15.2f}% | {lat_n5:<15.2f} | {cost_n5:<15.1f} |\n"
    table += f"| {'Verifier (n=3+1)':<22} | {acc_ver:<15.2f}% | {lat_ver:<15.2f} | {cost_ver:<15.1f} |\n"
    table += f"| {'Reflection (1+1+1)':<22} | {acc_ref:<15.2f}% | {lat_ref:<15.2f} | {cost_ref:<15.1f} |\n"
    table += table_divider
    
    # 打印到终端
    print(table)
    
    # --- 自动保存到文件 ---
    try:
        with open(FINAL_TABLE_FILE, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"\n✅ 成功! 最终表格已保存到: {FINAL_TABLE_FILE}")
    except Exception as e:
        print(f"\n❌ 保存 TXT 失败: {e}")
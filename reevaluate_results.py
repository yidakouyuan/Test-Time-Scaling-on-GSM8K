import json
import re

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

def reevaluate_json(filepath):
    """
    读取一个 JSON 日志文件，并用更强的提取器重新计分。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filepath}")
        return

    correct_count = 0
    total_count = len(data)
    
    for item in data:
        # 提取真实答案 (并转换为浮点数)
        try:
            ground_truth = float(item['ground_truth'])
        except ValueError:
            # 处理 "36" vs "400/11" 的情况
            if item['ground_truth'] == "36": 
                ground_truth = 36.0
            else:
                ground_truth = flexible_extract_answer(item['ground_truth'])

        # 提取预测答案 (并转换为浮点数)
        # Verifier JSON 使用 'final_prediction'
        # S-C/Baseline JSON 使用 'final_prediction'
        prediction = flexible_extract_answer(item['final_prediction'])

        # 4. 比较 (使用浮点数容差)
        if ground_truth is not None and prediction is not None:
            if abs(ground_truth - prediction) < 1e-5: # 比较浮点数
                item['is_correct_new'] = True
                correct_count += 1
            else:
                item['is_correct_new'] = False
        else:
             item['is_correct_new'] = False

    new_accuracy = (correct_count / total_count) * 100
    print(f"--- 重新评估: {filepath} ---")
    print(f"📊 新的准确率: {new_accuracy:.2f}% ({correct_count} / {total_count})")
    print("-" * 30)
    return new_accuracy

if __name__ == "__main__":
    print("--- 运行真实准确率评估 ---")
    
    # 重新评估你所有的运行结果
    acc_n1 = reevaluate_json("baseline_results_n1_limit100.json")
    acc_n3 = reevaluate_json("sc_results_n3_limit100.json")
    acc_n5 = reevaluate_json("sc_results_n5_limit100.json")
    acc_ver = reevaluate_json("verifier_results_n3_limit100.json")
    
    print("\n\n--- 最终修正后的 3D 边界表格 ---")
    print(f"| {'策略':<22} | {'📊 真实准确率':<15} |")
    print(f"|{'-'*24}|{'-'*17}|")
    print(f"| {'Baseline (n=1)':<22} | {acc_n1:<15.2f}% |")
    print(f"| {'S-C (n=3)':<22} | {acc_n3:<15.2f}% |")
    print(f"| {'S-C (n=5)':<22} | {acc_n5:<15.2f}% |")
    print(f"| {'Verifier (n=3+1)':<22} | {acc_ver:<15.2f}% |")
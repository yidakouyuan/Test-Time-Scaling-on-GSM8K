import json

# --- 配置：确保你的 JSON 文件名与之一致 ---
FILE_N1 = "baseline_results_n1_limit100.json"
FILE_N3 = "sc_results_n3_limit100.json"
FILE_N5 = "sc_results_n5_limit100.json"

# --- 新增：这是生成的报告文件名 ---
ANALYSIS_REPORT_FILE = "analysis_report.txt"
# ---

def load_results(filepath):
    """ 辅助函数：加载 JSON 并将其转换为 {id: item} 的字典 """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 按 'id' 键创建字典，方便快速查找
            return {item['id']: item for item in data}
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filepath}。请确保文件在同一文件夹下。")
        return None
    except Exception as e:
        print(f"❌ 错误: 读取 {filepath} 失败。{e}")
        return None

def compare_runs():
    print("--- 正在加载 .json 日志文件... ---")
    results_n1 = load_results(FILE_N1)
    results_n3 = load_results(FILE_N3)
    results_n5 = load_results(FILE_N5)

    if not all([results_n1, results_n3, results_n5]):
        print("--- 分析中止，缺少必要的 JSON 文件。 ---")
        return

    # 使用 'with open' 来自动写入文件
    try:
        with open(ANALYSIS_REPORT_FILE, 'w', encoding='utf-8') as f:
            print(f"--- 正在生成报告: {ANALYSIS_REPORT_FILE} ... ---")
            
            f.write("="*50 + "\n")
            f.write("--- 定性分析：查找结果差异 ---\n")
            f.write("="*50 + "\n")

            # 找出 n=3 修正了 n=1 的案例
            f.write("\n🔍 [分析 A] n=3 修正了 n=1 的错误 (成功案例):\n")
            found_A = 0
            # 假设所有文件都有 0-99 的 id
            for i in range(len(results_n1)): 
                if i not in results_n1 or i not in results_n3: continue
                
                if not results_n1[i]['is_correct'] and results_n3[i]['is_correct']:
                    found_A += 1
                    f.write(f"\n  > ID: {i} (n=1 错, n=3 对)\n")
                    f.write(f"    真实答案: {results_n1[i]['ground_truth']}\n")
                    f.write(f"    n=1 预测: {results_n1[i]['final_prediction']}\n")
                    f.write(f"    n=3 预测: {results_n3[i]['final_prediction']} (投票: {results_n3[i]['vote_details']})\n")
            if found_A == 0:
                f.write("  未发现 n=3 修正 n=1 的案例。\n")


            # 找出 n=5 搞砸了 n=3 的案例
            f.write("\n🔍 [分析 B] n=5 搞砸了 n=3 的正确答案 (失败案例):\n")
            found_B = 0
            for i in range(len(results_n1)):
                if i not in results_n3 or i not in results_n5: continue
                
                if results_n3[i]['is_correct'] and not results_n5[i]['is_correct']:
                    found_B += 1
                    f.write(f"\n  > ID: {i} (n=3 对, n=5 错)\n")
                    f.write(f"    真实答案: {results_n3[i]['ground_truth']}\n")
                    f.write(f"    n=3 预测: {results_n3[i]['final_prediction']} (投票: {results_n3[i]['vote_details']})\n")
                    f.write(f"    n=5 预测: {results_n5[i]['final_prediction']} (投票: {results_n5[i]['vote_details']})\n")
            if found_B == 0:
                f.write("  未发现 n=5 搞砸 n=3 的案例。\n")
            
            
            # 找出 n=3 或 n=5 搞砸了 n=1 的案例 (如果 n=1 是对的)
            f.write("\n🔍 [分析 C] n=3 或 n=5 搞砸了 n=1 的正确答案 (失败案例):\n")
            found_C = 0
            for i in range(len(results_n1)):
                if i not in results_n1 or i not in results_n3 or i not in results_n5: continue
                
                if results_n1[i]['is_correct'] and (not results_n3[i]['is_correct'] or not results_n5[i]['is_correct']):
                    found_C += 1
                    f.write(f"\n  > ID: {i} (n=1 对, 但 S-C 错了)\n")
                    f.write(f"    真实答案: {results_n1[i]['ground_truth']}\n")
                    f.write(f"    n=1 预测: {results_n1[i]['final_prediction']}\n")
                    f.write(f"    n=3 预测: {results_n3[i]['final_prediction']} (投票: {results_n3[i]['vote_details']})\n")
                    f.write(f"    n=5 预测: {results_n5[i]['final_prediction']} (投票: {results_n5[i]['vote_details']})\n")
            if found_C == 0:
                f.write("  未发现 S-C 搞砸 n=1 的案例。\n")
                
        print(f"\n--- ✅ 分析完成！结果已保存到 {ANALYSIS_REPORT_FILE} ---")
        
    except Exception as e:
        print(f"❌ 错误: 写入文件失败。{e}")


if __name__ == "__main__":
    compare_runs()
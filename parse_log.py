import re
import json

# --- 配置 ---
INPUT_LOG_FILE = "raw_output_n5_limit100.txt" # 你保存的原始日志
OUTPUT_JSON_FILE = "sc_results_n5_limit100.json"
OUTPUT_SUMMARY_FILE = "sc_summary_n5_limit100.txt"
# ---

def parse_log():
    print(f"--- 正在读取 {INPUT_LOG_FILE} ---")
    
    detailed_results = []
    summary_lines = []
    
    in_data_section = False
    in_summary_section = False

    try:
        with open(INPUT_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 寻找数据部分的开始
                if "--- 运行 Self-Consistency" in line:
                    in_data_section = True
                    continue
                
                # 寻找数据部分的结束 / 总结部分的开始
                if "==============================" in line:
                    in_data_section = False
                    in_summary_section = True
                    summary_lines.append(line)
                    continue
                    
                # 处理数据行
                if in_data_section:
                    # 跳过标题栏的分割线
                    if "------------------" in line:
                        continue
                        
                    # 真正的数据行
                    parts = line.split('|')
                    if len(parts) == 7:
                        try:
                            item_id = int(parts[0].strip())
                            ground_truth = parts[1].strip()
                            prediction = parts[2].strip()
                            latency_s = float(parts[3].strip())
                            tokens = int(parts[4].strip())
                            result_icon = parts[5].strip()
                            vote_details = parts[6].strip()
                            
                            detailed_results.append({
                                "id": item_id,
                                "ground_truth": ground_truth,
                                "final_prediction": prediction,
                                "is_correct": "✅" in result_icon,
                                "latency_s": latency_s,
                                "total_tokens": tokens,
                                "vote_details": vote_details
                            })
                        except ValueError as e:
                            print(f"跳过无法解析的行: {line} -> 错误: {e}")
                        except IndexError as e:
                            print(f"跳过索引错误的行: {line} -> 错误: {e}")
                            
                # 处理总结行
                if in_summary_section:
                    summary_lines.append(line)
                    
        # 1. 写入详细的 JSON 结果
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=4, ensure_ascii=False)
        print(f"✅ 成功! 详细结果已保存到: {OUTPUT_JSON_FILE}")
        
        # 2. 写入总结的 TXT 结果
        # 我们把PS D:\...> 这行去掉
        if summary_lines and "PS D:" in summary_lines[-1]:
            summary_lines.pop()
            
        with open(OUTPUT_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        print(f"✅ 成功! 总结报告已保存到: {OUTPUT_SUMMARY_FILE}")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {INPUT_LOG_FILE}。请确保它和本脚本在同一个文件夹下。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    parse_log()
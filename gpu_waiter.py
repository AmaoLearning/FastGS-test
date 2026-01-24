import subprocess
import time
import os
import sys

# ================= 配置区域 =================
# 你的实验脚本命令 (例如: "bash train.sh" 或 "python main.py")
MY_COMMAND = "sh train_hash.sh" 

# 判定为空闲的阈值
MAX_MEMORY_USED = 1000  # 显存已用小于 1000MB 视为可用 (4090通常有24G)
MAX_GPU_UTIL = 5        # GPU利用率小于 5% 视为可用
CHECK_INTERVAL = 30     # 每隔 30 秒检查一次
# ===========================================

def get_free_gpus():
    """
    使用 nvidia-smi 查询显卡状态
    返回可用显卡的 ID 列表 (例如 [0, 2])
    """
    try:
        # 查询 索引, 显存已用, GPU利用率
        # --format=csv,noheader,nounits 输出纯数字，方便解析
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        free_gpus = []
        lines = result.strip().split('\n')
        
        for line in lines:
            if not line: continue
            idx, mem_used, util = line.split(',')
            idx = int(idx)
            mem_used = int(mem_used)
            util = int(util)
            
            # 判断显卡是否满足空闲条件
            if mem_used < MAX_MEMORY_USED and util < MAX_GPU_UTIL:
                free_gpus.append(idx)
                
        return free_gpus
        
    except Exception as e:
        print(f"Error querying GPU status: {e}")
        return []

def main():
    print(f"--- 开始监控显卡 (检测间隔: {CHECK_INTERVAL}s) ---")
    print(f"目标命令: {MY_COMMAND}")
    print(f"空闲标准: 显存已用 < {MAX_MEMORY_USED}MB, 利用率 < {MAX_GPU_UTIL}%")
    
    start_time = time.time()
    
    while True:
        free_gpus = get_free_gpus()
        
        if free_gpus:
            # 默认选择第一个空闲的显卡
            target_gpu = free_gpus[0]
            print(f"\n[Success] 发现空闲显卡: GPU {target_gpu}")
            print(f"正在启动任务...")
            
            # 设置环境变量，只让程序看到这张卡
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
            
            # 记录等待时间
            wait_hours = (time.time() - start_time) / 3600
            print(f"任务已等待: {wait_hours:.2f} 小时")
            print("-" * 30)
            
            # 执行你的脚本
            # 使用 shell=True 允许执行 bash 脚本或复杂的 shell 命令
            try:
                subprocess.run(MY_COMMAND, shell=True, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"任务执行出错，返回码: {e.returncode}")
            
            break # 任务跑完后退出监控
        
        else:
            # 打印一个动态的小点，表示正在活着
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
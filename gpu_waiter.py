import subprocess
import time
import os
import sys
import argparse

# ================= 默认配置 =================
DEFAULT_MAX_MEMORY = 1000  # 显存已用小于 1000MB 视为可用
DEFAULT_MAX_UTIL = 5       # GPU利用率小于 5% 视为可用
DEFAULT_INTERVAL = 30      # 每隔 30 秒检查一次
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="等待空闲 GPU 并自动执行实验命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python gpu_waiter.py "python train.py --lr 0.001"
  python gpu_waiter.py "sh train.sh" --max_memory 2000
  python gpu_waiter.py "bash experiment.sh" --interval 60 --max_util 10
        """
    )
    parser.add_argument("command", type=str, help="要执行的实验命令 (用引号包裹)")
    parser.add_argument("--max_memory", type=int, default=DEFAULT_MAX_MEMORY,
                        help=f"显存已用阈值(MB)，低于此值视为空闲 (默认: {DEFAULT_MAX_MEMORY})")
    parser.add_argument("--max_util", type=int, default=DEFAULT_MAX_UTIL,
                        help=f"GPU利用率阈值(%%)，低于此值视为空闲 (默认: {DEFAULT_MAX_UTIL})")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"检查间隔(秒) (默认: {DEFAULT_INTERVAL})")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定使用的 GPU ID，不指定则自动选择第一个空闲的")
    return parser.parse_args()


def get_free_gpus(max_memory, max_util):
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
            if mem_used < max_memory and util < max_util:
                free_gpus.append(idx)
                
        return free_gpus
        
    except Exception as e:
        print(f"Error querying GPU status: {e}")
        return []

def main():
    args = parse_args()
    
    print(f"--- 开始监控显卡 (检测间隔: {args.interval}s) ---")
    print(f"目标命令: {args.command}")
    print(f"空闲标准: 显存已用 < {args.max_memory}MB, 利用率 < {args.max_util}%")
    if args.gpu is not None:
        print(f"指定 GPU: {args.gpu}")
    
    start_time = time.time()
    
    while True:
        free_gpus = get_free_gpus(args.max_memory, args.max_util)
        
        # 如果指定了 GPU，检查该 GPU 是否空闲
        if args.gpu is not None:
            if args.gpu in free_gpus:
                target_gpu = args.gpu
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(args.interval)
                continue
        elif free_gpus:
            target_gpu = free_gpus[0]
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(args.interval)
            continue
        
        # 找到可用 GPU
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
        try:
            subprocess.run(args.command, shell=True, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"任务执行出错，返回码: {e.returncode}")
        
        break  # 任务跑完后退出监控

if __name__ == "__main__":
    main()
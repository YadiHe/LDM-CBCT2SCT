#!/usr/bin/env python
"""
GPU显存监控脚本 - 实时监控训练时的显存使用情况

用法:
    # 在另一个终端运行,实时监控训练过程
    python scripts/monitor_gpu_memory.py

    # 指定采样间隔(秒)
    python scripts/monitor_gpu_memory.py --interval 2
"""
import argparse
import time
import subprocess
from datetime import datetime

def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        if output:
            mem_used, mem_total, mem_free, gpu_util = output.split(',')
            return {
                'used': int(mem_used),
                'total': int(mem_total),
                'free': int(mem_free),
                'util': int(gpu_util),
                'usage_pct': int(mem_used) / int(mem_total) * 100
            }
    except Exception as e:
        print(f"Error: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="GPU显存实时监控")
    parser.add_argument('--interval', type=int, default=1, help='采样间隔(秒)')
    parser.add_argument('--alert-threshold', type=int, default=90, help='显存使用率报警阈值(%)')
    parser.add_argument('--log', type=str, default=None, help='保存日志文件路径')
    args = parser.parse_args()

    print("=" * 70)
    print("GPU显存监控启动")
    print(f"采样间隔: {args.interval}s | 报警阈值: {args.alert_threshold}%")
    print("=" * 70)
    print(f"{'时间':<20} {'已用(MB)':<12} {'空闲(MB)':<12} {'总计(MB)':<12} {'使用率':<10} {'GPU利用率'}")
    print("-" * 70)

    log_file = None
    if args.log:
        log_file = open(args.log, 'w')
        log_file.write("timestamp,used_mb,free_mb,total_mb,usage_pct,gpu_util\n")

    max_used = 0
    try:
        while True:
            info = get_gpu_memory()
            if info:
                timestamp = datetime.now().strftime('%H:%M:%S')

                # 更新最大显存使用
                if info['used'] > max_used:
                    max_used = info['used']

                # 打印状态
                status = f"{timestamp:<20} {info['used']:<12} {info['free']:<12} {info['total']:<12} {info['usage_pct']:<9.1f}% {info['util']}%"

                # 报警标识
                alert = ""
                if info['usage_pct'] >= args.alert_threshold:
                    alert = " ⚠️ 接近显存上限!"
                    status = f"\033[91m{status}{alert}\033[0m"  # 红色
                elif info['usage_pct'] >= args.alert_threshold - 10:
                    alert = " ⚡ 显存使用偏高"
                    status = f"\033[93m{status}{alert}\033[0m"  # 黄色

                print(status)

                # 写入日志
                if log_file:
                    log_file.write(f"{timestamp},{info['used']},{info['free']},{info['total']},{info['usage_pct']:.2f},{info['util']}\n")
                    log_file.flush()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print(f"监控结束 | 峰值显存: {max_used} MB ({max_used/info['total']*100:.1f}%)")
        print("=" * 70)
        if log_file:
            log_file.close()
            print(f"日志已保存: {args.log}")

if __name__ == '__main__':
    main()

"""分析 Curator JSONL 日志, 汇总 token 节省与成本

用法:
  python analyze_curator_log.py <log_path.jsonl>
"""
import json, sys
from collections import defaultdict


def analyze(log_path: str):
    turns = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))

    if not turns:
        print("日志为空")
        return

    total_cost = defaultdict(int)
    total_saved = defaultdict(int)
    skipped_count = 0
    active_turns = 0

    print(f"{'Turn':>5} {'Budget%':>8} {'Active':>6} {'Dormant':>7} {'Tokens':>7} "
          f"{'Saved':>7} {'Spent':>7} {'Net':>7} {'Skipped':>7}")
    print("-" * 75)

    for t in turns:
        turn = t["turn"]
        budget = t.get("budget", {})
        result = t.get("result", {})
        cost = t.get("cost", {})
        skipped = t.get("skipped", False)

        if skipped:
            skipped_count += 1
            continue

        active_turns += 1

        saved = cost.get("saved", {})
        spent = cost.get("spent", {})

        ts = saved.get("total", 0)
        tp = spent.get("total", 0)
        net = cost.get("net", 0)

        print(f"{turn:>5} {budget.get('pct', 0):>7}% {result.get('active', 0):>6} "
              f"{result.get('dormant', 0):>7} {result.get('tokens', 0):>7} "
              f"{ts:>7} {tp:>7} {net:>+7}")

        for k, v in saved.items():
            total_saved[k] += v
        total_cost["prompt"] += spent.get("prompt", 0)
        total_cost["response"] += spent.get("response", 0)
        total_cost["total"] += spent.get("total", 0)
        total_cost["net"] += net

    print("-" * 75)
    print(f"\n汇总 ({active_turns} 轮有效, {skipped_count} 轮跳过):")
    print(f"  节省总计:")
    print(f"    删除:   {total_saved['deletion']:,} tokens")
    print(f"    压缩:   {total_saved['compression']:,} tokens")
    print(f"    休眠:   {total_saved['dormancy']:,} tokens")
    print(f"    合计:   {total_saved['total']:,} tokens")

    print(f"\n  消耗总计:")
    print(f"    Prompt:  {total_cost['prompt']:,} tokens")
    print(f"    Response:{total_cost['response']:,} tokens")
    print(f"    合计:   {total_cost['total']:,} tokens")

    print(f"\n  净节省:   {total_cost['net']:+,} tokens")
    if total_saved['total'] > 0:
        roi = total_cost['net'] / total_saved['total'] * 100
        print(f"  净收益率: {roi:.1f}% (净节省/总节省)")

    # 时序统计
    if active_turns > 1:
        avg_saved = total_saved['total'] / active_turns
        avg_spent = total_cost['total'] / active_turns
        avg_net = total_cost['net'] / active_turns
        print(f"\n  平均每轮: 节省 {avg_saved:,.0f} | 消耗 {avg_spent:,.0f} | 净 {avg_net:+,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1])

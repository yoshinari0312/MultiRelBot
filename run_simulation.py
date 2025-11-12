"""
会話シミュレーション実行スクリプト

使用方法:
    python run_simulation.py [--num-episodes N] [--output OUTPUT_DIR]

例:
    python run_simulation.py --num-episodes 10
    python run_simulation.py --num-episodes 20 --output results/sim_20250113
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import config
from simulation_environment import SimulationEnvironment


def calculate_statistics(all_stats: List[Dict]) -> Dict:
    """
    全エピソードの統計を計算

    Args:
        all_stats: 各エピソードの統計リスト

    Returns:
        集計統計
    """
    if not all_stats:
        return {}

    # 安定達成したエピソード
    stable_episodes = [s for s in all_stats if s["final_stable"]]
    stable_rate = len(stable_episodes) / len(all_stats) if all_stats else 0.0

    # 疎外ノードが発生したエピソード
    isolated_episodes = [s for s in all_stats if len(s["final_isolated_nodes"]) > 0]
    isolated_rate = len(isolated_episodes) / len(all_stats) if all_stats else 0.0

    # 安定達成エピソードの統計
    if stable_episodes:
        avg_human_utterances_to_stable = sum(
            s["human_utterance_count"] for s in stable_episodes
        ) / len(stable_episodes)
        avg_robot_utterances_to_stable = sum(
            s["robot_utterance_count"] for s in stable_episodes
        ) / len(stable_episodes)
    else:
        avg_human_utterances_to_stable = None
        avg_robot_utterances_to_stable = None

    # 全エピソードの平均
    avg_human_utterances = sum(s["human_utterance_count"] for s in all_stats) / len(
        all_stats
    )
    avg_robot_utterances = sum(s["robot_utterance_count"] for s in all_stats) / len(
        all_stats
    )
    avg_duration = sum(s["duration_seconds"] for s in all_stats) / len(all_stats)

    # 不安定三角形数の平均
    avg_unstable_triads = sum(s["final_unstable_triads"] for s in all_stats) / len(
        all_stats
    )

    stats = {
        "total_episodes": len(all_stats),
        "stable_episodes": len(stable_episodes),
        "stable_rate": stable_rate,
        "isolated_episodes": len(isolated_episodes),
        "isolated_rate": isolated_rate,
        "avg_human_utterances": avg_human_utterances,
        "avg_robot_utterances": avg_robot_utterances,
        "avg_human_utterances_to_stable": avg_human_utterances_to_stable,
        "avg_robot_utterances_to_stable": avg_robot_utterances_to_stable,
        "avg_unstable_triads": avg_unstable_triads,
        "avg_duration_seconds": avg_duration,
    }

    return stats


def save_results(all_stats: List[Dict], summary_stats: Dict, output_dir: str):
    """
    結果を保存

    Args:
        all_stats: 各エピソードの統計リスト
        summary_stats: 集計統計
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # サマリーを保存
    summary_path = output_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    print(f"\n💾 サマリーを保存: {summary_path}")

    # 各エピソードの詳細を保存
    for stats in all_stats:
        episode_id = stats["episode_id"]
        episode_path = output_path / f"episode_{episode_id}.json"

        # ログは別ファイルに保存（サイズが大きくなるため）
        logs = stats.pop("logs", [])
        robot_utterances = stats.pop("robot_utterances", [])

        with open(episode_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 会話ログを保存
        log_path = output_path / f"episode_{episode_id}_conversation.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"エピソード {episode_id}\n")
            f.write(f"話題: {stats['topic']}\n")
            if stats.get("topic_trigger"):
                f.write(f"トリガー: {stats['topic_trigger']}\n")
            f.write("=" * 80 + "\n\n")

            for log in logs:
                speaker = log.get("speaker", "?")
                utterance = log.get("utterance", "")
                f.write(f"[{speaker}] {utterance}\n")

    print(f"💾 各エピソードの詳細を保存: {output_path}")


def print_summary(stats: Dict):
    """
    サマリー統計を表示

    Args:
        stats: 集計統計
    """
    print(f"\n{'='*80}")
    print(f"📊 シミュレーション結果サマリー")
    print(f"{'='*80}")
    print(f"総エピソード数: {stats['total_episodes']}")
    print(
        f"安定達成エピソード数: {stats['stable_episodes']} ({stats['stable_rate']*100:.1f}%)"
    )
    print(
        f"疎外ノード発生エピソード数: {stats['isolated_episodes']} ({stats['isolated_rate']*100:.1f}%)"
    )
    print(f"\n平均人間発話数: {stats['avg_human_utterances']:.1f}")
    print(f"平均ロボット介入回数: {stats['avg_robot_utterances']:.1f}")

    if stats["avg_human_utterances_to_stable"] is not None:
        print(f"\n【安定達成エピソードのみ】")
        print(
            f"  平均人間発話数（安定まで）: {stats['avg_human_utterances_to_stable']:.1f}"
        )
        print(
            f"  平均ロボット介入回数（安定まで）: {stats['avg_robot_utterances_to_stable']:.1f}"
        )

    print(f"\n平均不安定三角形数: {stats['avg_unstable_triads']:.2f}")
    print(f"平均所要時間: {stats['avg_duration_seconds']:.1f}秒")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="会話シミュレーション実行")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="実行するエピソード数（デフォルト: config.yamlの設定）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: results/YYYYMMDD_HHMMSS）",
    )

    args = parser.parse_args()

    # 設定読み込み
    cfg = config.get_config()
    sim_cfg = getattr(cfg, "simulation", None)

    num_episodes = (
        args.num_episodes
        if args.num_episodes is not None
        else getattr(sim_cfg, "num_episodes", 10)
    )

    # 出力ディレクトリ
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/simulation_{timestamp}"

    print(f"{'='*80}")
    print(f"🚀 会話シミュレーション開始")
    print(f"{'='*80}")
    print(f"エピソード数: {num_episodes}")
    print(f"出力ディレクトリ: {output_dir}")

    # シミュレーション環境
    env = SimulationEnvironment()

    # 各エピソードを実行
    all_stats = []
    for i in range(1, num_episodes + 1):
        try:
            stats = env.run_episode(i)
            all_stats.append(stats)
        except KeyboardInterrupt:
            print("\n⚠️ 中断されました")
            break
        except Exception as e:
            print(f"\n❌ エピソード {i} でエラー: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_stats:
        print("❌ 実行されたエピソードがありません")
        return

    # 統計計算
    summary_stats = calculate_statistics(all_stats)

    # サマリー表示
    print_summary(summary_stats)

    # 結果保存
    save_results(all_stats, summary_stats, output_dir)

    print(f"\n✅ シミュレーション完了")


if __name__ == "__main__":
    main()

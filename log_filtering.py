"""
会話ログのフィルタリングユーティリティ

参考: https://github.com/yoshinari0312/policyRL_relation_robot/blob/main/app/utils/log_filtering.py
"""

from typing import List, Dict


def filter_logs_by_human_count(
    logs: List[Dict], human_count: int, exclude_robot: bool = False
) -> List[Dict]:
    """
    直近 human_count 個の人間発話と、その間に含まれるロボット発話を全て返す

    Args:
        logs: 全会話ログのリスト（各要素は {'speaker': str, 'utterance': str, ...} 形式）
        human_count: 含める人間発話の数
        exclude_robot: True の場合、ロボット発話を除外して人間発話のみを返す

    Returns:
        フィルタリングされた会話ログのリスト

    処理フロー:
        1. 人間発話のインデックスをリストアップ
        2. 直近 human_count 番目の人間発話の位置を特定
        3. その位置から最後までのログを抽出（ロボット発話も含まれる）
        4. exclude_robot=True なら、その中からロボット発話を除外
    """
    if not logs:
        return []

    # 人間発話のインデックスを取得（ロボット以外）
    human_indices = [i for i, log in enumerate(logs) if log.get("speaker") != "ロボット"]

    # 人間発話が指定数以下なら全ログを返す
    if len(human_indices) <= human_count:
        if exclude_robot:
            # ロボット発話を除外
            return [log for log in logs if log.get("speaker") != "ロボット"]
        else:
            # 全ログを返す
            return logs

    # 直近 human_count 番目の人間発話位置を取得
    start_idx = human_indices[-human_count]

    # その位置から終端までを抽出
    filtered_logs = logs[start_idx:]

    if exclude_robot:
        # ロボット発話を除外
        filtered_logs = [log for log in filtered_logs if log.get("speaker") != "ロボット"]

    return filtered_logs

"""
会話txtファイルから関係性推定を行い、CSV出力するスクリプト

使用方法:
    1. ファイル上部の設定セクションで入力ファイルやパラメータを設定
    2. python relation_estimator_from_txt.py を実行
    3. 出力CSVファイルが生成される
"""

import re
from collections import defaultdict, deque
from itertools import combinations
from typing import List, Dict, Tuple, Any
import csv
import config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from log_filtering import filter_logs_by_human_count

# ========== 設定 ==========
INPUT_FILE = "conversation.txt"
OUTPUT_FILE = "relation_scores.csv"
LLM_MODEL = None  # Noneの場合はconfig.yamlのrelation_modelを使用
USE_EMA = True
DECAY_FACTOR = 1.5
MAX_HISTORY_HUMAN = 9  # 人間発話数（その間のロボット発話も含む）
DEBUG = True
NUM_TRIALS = 5  # 各推定を繰り返す回数
# ==========================


class EMAScorer:
    """各試行ごとに独立したEMA状態を保持するクラス"""

    def __init__(self, use_ema: bool = True, decay_factor: float = 1.5):
        self.use_ema = use_ema
        self.decay_factor = decay_factor
        self.scores = defaultdict(float)  # ペアごとのEMAスコア
        self.history = defaultdict(lambda: deque(maxlen=3))  # ペアごとの過去発話数

    def update(self, pair: Tuple[str, str], raw_score: float, utterance_counts: Dict[str, int]) -> float:
        """
        EMAスコアを更新して返す

        Args:
            pair: ペア（例: ('A', 'B')）
            raw_score: LLMの生スコア
            utterance_counts: 各参加者の発話数

        Returns:
            更新後のスコア（EMA適用済み）
        """
        a, b = pair
        session_utterance = min(utterance_counts.get(a, 0), utterance_counts.get(b, 0))

        if pair not in self.scores:
            # 初回はそのままセット
            self.scores[pair] = raw_score
            self.history[pair].append(session_utterance)
            if DEBUG and self.use_ema:
                print(f"    🔢 α計算: {pair}, session={session_utterance}, past=0, α=1.00 (初回)")
                print(f"    🔁 EMA初期化: {pair} = {raw_score:+.1f}")
            return raw_score
        else:
            if self.use_ema:
                past_utterances = list(self.history[pair])
                total_past = sum(past_utterances)
                total_with_current = total_past + session_utterance

                ratio = session_utterance / total_with_current if total_with_current > 0 else 1.0
                alpha = max(0.01, min(1.0, self.decay_factor * ratio))

                prev = self.scores[pair]
                updated = alpha * raw_score + (1 - alpha) * prev
                self.scores[pair] = updated

                if DEBUG:
                    print(f"    🔢 α計算: {pair}, session={session_utterance}, past={total_past}, total={total_with_current}, α={alpha:.2f}")
                    print(f"    🔁 EMA更新: {pair} = {alpha:.2f}×{raw_score:+.1f} + {(1-alpha):.2f}×{prev:+.1f} → {updated:+.1f}")

                self.history[pair].append(session_utterance)
                return updated
            else:
                # EMA無効時は生スコアをそのまま使用
                self.scores[pair] = raw_score
                self.history[pair].append(session_utterance)
                if DEBUG:
                    print(f"    🔄 スコア更新（EMA無効）: {pair} = {raw_score:+.1f}")
                return raw_score


def parse_conversation_file(file_path: str) -> List[Dict[str, str]]:
    """
    会話txtファイルを読み込み、ログリストに変換

    形式: [話者名] 発言内容

    Args:
        file_path: 入力txtファイルパス

    Returns:
        [{'speaker': 'A', 'utterance': '...'}, ...]
    """
    logs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # [話者名] 発言内容 の形式をパース
            match = re.match(r'^\[(.+?)\]\s*(.+)$', line)
            if match:
                speaker = match.group(1).strip()
                utterance = match.group(2).strip()
                logs.append({
                    'speaker': speaker,
                    'utterance': utterance
                })

    return logs


def detect_participants(logs: List[Dict[str, str]]) -> List[str]:
    """
    ログから参加者を検出（ロボット除外）

    Args:
        logs: 会話ログリスト

    Returns:
        参加者リスト（ソート済み）
    """
    participants = set()
    for log in logs:
        speaker = log['speaker']
        if speaker != 'ロボット':
            participants.add(speaker)

    return sorted(list(participants))


def split_into_rounds(logs: List[Dict[str, str]], participants: List[str]) -> List[int]:
    """
    ラウンド終了位置（推定タイミング）のインデックスリストを返す

    ラウンド: 全参加者が1回ずつ発言した単位
    3人なら3発話ごと、4人なら4発話ごとに推定

    Args:
        logs: 会話ログリスト
        participants: 参加者リスト

    Returns:
        推定を行うログインデックスのリスト
    """
    round_end_indices = []
    num_participants = len(participants)
    last_participant = participants[-1]  # 参加者リストの最後（CまたはD）

    for i, log in enumerate(logs):
        speaker = log['speaker']
        if speaker == last_participant:
            # 最後の参加者が発言したタイミングをラウンド終了とする
            round_end_indices.append(i)

    return round_end_indices


def estimate_relation_once(
    logs: List[Dict[str, str]],
    participants: List[str],
    max_history: int,
    llm_model: str,
    _CFG: Any
) -> Dict[Tuple[str, str], float]:
    """
    1回の関係性推定を実行

    Args:
        logs: 会話ログリスト
        participants: 参加者リスト
        max_history: 使用する人間発話数
        llm_model: 使用するLLMモデル（deployment名）
        _CFG: 設定オブジェクト

    Returns:
        {('A', 'B'): 0.5, ...}
    """
    # 会話履歴をフィルタリング（ロボット発話も含む）
    filtered_logs = filter_logs_by_human_count(logs, max_history, exclude_robot=False)

    # プロンプト作成（community_analyzer.pyと同じ）
    conversation = "\n".join(
        [f"[{log['speaker']}] {log['utterance']}" for log in filtered_logs]
    )
    pair_lines = "\n".join(
        [f"- {a} × {b}" for a, b in combinations(participants, 2)]
    )
    output_format = "\n".join(
        [f"{a}-{b}: " for a, b in combinations(participants, 2)]
    )

    prompt = f"""
以下の会話を読み、{', '.join(participants)}の「仲の良さ（親密度）」を -1.0 〜 +1.0 の間の**実数（小数第1位まで）**で評価してください。
0.0 は特に親しさも対立も感じない「中立的な状態」です。
そこから -1.0（強い対立） 〜 +1.0（非常に親しい） に向けて、どれくらい離れているかを評価してください。
出力形式を厳守し、理由・補足説明などは一切加えないでください。

具体例：
-1.0 例：皮肉、批判、無視、相手を無視して話を進める
+1.0 例：共感、褒める、相手に話題を振る、一緒に行動する

評価対象は以下のペアです（ロボットは含みません）：
{pair_lines}

会話：
{conversation}

出力形式：
{output_format}
"""

    if DEBUG:
        print(f"\n  📝 LLMへの入力プロンプト:")
        print(f"  {'='*60}")
        print(prompt)
        print(f"  {'='*60}")

    # Azure OpenAI API呼び出し
    client, deployment = get_azure_chat_completion_client(_CFG.llm, model_type="relation")
    if not client or not deployment:
        raise RuntimeError("Failed to obtain Azure OpenAI client for relation scoring.")

    # LLM_MODELが指定されている場合はそちらを使用
    if llm_model:
        deployment = llm_model

    messages = [{"role": "user", "content": prompt}]
    relation_temperature = getattr(_CFG.llm, "relation_temperature", 0.3)
    params = build_chat_completion_params(
        deployment, messages, _CFG.llm, temperature=relation_temperature
    )

    res = client.chat.completions.create(**params)
    response_text = res.choices[0].message.content.strip()

    if DEBUG:
        print(f"\n  🤖 LLM生応答:")
        print(f"  {response_text}")

    # レスポンスをパース
    scores = parse_scores_from_response(response_text, participants)

    if DEBUG:
        print(f"\n  📊 パース結果:")
        for pair, score in sorted(scores.items()):
            print(f"    {pair}: {score:+.1f}")

    return scores


def parse_scores_from_response(response_text: str, participants: List[str]) -> Dict[Tuple[str, str], float]:
    """
    LLMレスポンスから関係性スコアをパース

    Args:
        response_text: LLMの応答テキスト
        participants: 参加者リスト

    Returns:
        {('A', 'B'): 0.5, ...}
    """
    scores = {}
    lines = response_text.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # 説明行をスキップ
        if re.match(r"^\s*(評価理由|理由)\W*", line):
            if DEBUG:
                print(f"    ⚠️ 説明行をスキップ: {line.strip()}")
            i += 1
            continue

        # ペア形式を検索: A-B 形式
        pair_match = re.search(r"([^\s:]+)\s*-\s*([^\s:]+)", line)
        if not pair_match:
            # 日本語形式: XとY 形式
            pair_match = re.search(
                r"([A-Za-z0-9_\u4E00-\u9FFF\u3040-\u30FF]+)と([A-Za-z0-9_\u4E00-\u9FFF\u3040-\u30FF]+)",
                line,
            )
            if not pair_match:
                if DEBUG and line.strip():
                    print(f"    ⚠️ ペア形式エラー: {line.strip()}")
                i += 1
                continue

        a, b = pair_match.group(1).strip(), pair_match.group(2).strip()

        # 数値を抽出
        match = re.search(r"[+-]?\d+(?:\.\d+)?", line)

        # 数値が見つからない場合、次の行を探す（最大2行先まで）
        lookahead = 0
        while match is None and lookahead < 2 and (i + 1 + lookahead) < len(lines):
            next_line = lines[i + 1 + lookahead]
            if re.match(r"^\s*(評価理由|理由)\W*", next_line):
                lookahead += 1
                continue
            match = re.search(r"[+-]?\d+(?:\.\d+)?", next_line)
            if match:
                i = i + 1 + lookahead
                break
            lookahead += 1

        if match:
            key = tuple(sorted([a, b]))
            try:
                scores[key] = float(match.group())
            except ValueError:
                if DEBUG:
                    print(f"    ⚠️ 数値変換失敗: {line.strip()}")
        else:
            if DEBUG:
                print(f"    ⚠️ 数値抽出失敗: {a}-{b}")

        i += 1

    return scores


def main():
    """メイン処理"""
    # 設定読み込み
    _CFG = config.get_config()

    if DEBUG:
        print("=" * 80)
        print("🚀 関係性推定スクリプト開始")
        print("=" * 80)
        print(f"\n📋 設定:")
        print(f"  入力ファイル: {INPUT_FILE}")
        print(f"  出力ファイル: {OUTPUT_FILE}")
        print(f"  LLMモデル: {LLM_MODEL if LLM_MODEL else '(config.yamlのrelation_model)'}")
        print(f"  EMA使用: {USE_EMA}")
        print(f"  Decay Factor: {DECAY_FACTOR}")
        print(f"  会話履歴長: {MAX_HISTORY_HUMAN} 人間発話")
        print(f"  試行回数: {NUM_TRIALS}")

    # txtファイルを読み込み
    logs = parse_conversation_file(INPUT_FILE)
    if DEBUG:
        print(f"\n📄 会話ログ読み込み完了: {len(logs)} 発話")

    # 参加者を検出
    participants = detect_participants(logs)
    if DEBUG:
        print(f"👥 参加者: {participants}")

    if len(participants) < 2:
        print("❌ エラー: 参加者が2人未満です")
        return

    # ペアリストを生成（ソート順）
    pairs = list(combinations(participants, 2))
    if DEBUG:
        print(f"🔗 ペア数: {len(pairs)}")
        for pair in pairs:
            print(f"  - {pair[0]}-{pair[1]}")

    # ラウンド分割
    round_end_indices = split_into_rounds(logs, participants)
    if DEBUG:
        print(f"\n🔄 ラウンド数: {len(round_end_indices)}")
        for idx, end_idx in enumerate(round_end_indices, 1):
            print(f"  ラウンド {idx}: ログインデックス {end_idx} まで")

    # 各試行用のEMAスコアラーを初期化
    ema_scorers = [EMAScorer(USE_EMA, DECAY_FACTOR) for _ in range(NUM_TRIALS)]

    # 結果を格納: {pair: [[trial1_round1, trial1_round2, ...], [trial2_round1, ...], ...]}
    results = defaultdict(lambda: [[] for _ in range(NUM_TRIALS)])

    # 各ラウンドで推定を実行
    for round_idx, end_idx in enumerate(round_end_indices, 1):
        if DEBUG:
            print(f"\n{'='*80}")
            print(f"🔍 ラウンド {round_idx} の推定開始（ログインデックス 0〜{end_idx}）")
            print(f"{'='*80}")

        # このラウンドまでのログ
        round_logs = logs[:end_idx + 1]

        # 各参加者の発話数をカウント
        utterance_counts = defaultdict(int)
        for log in round_logs:
            if log['speaker'] in participants:
                utterance_counts[log['speaker']] += 1

        if DEBUG:
            print(f"\n📊 発話数: {dict(utterance_counts)}")

        # NUM_TRIALS回推定を実行
        for trial_idx in range(NUM_TRIALS):
            if DEBUG:
                print(f"\n--- 試行 {trial_idx + 1}/{NUM_TRIALS} ---")

            # 関係性推定
            raw_scores = estimate_relation_once(
                round_logs, participants, MAX_HISTORY_HUMAN, LLM_MODEL, _CFG
            )

            # EMA適用
            ema_scorer = ema_scorers[trial_idx]
            for pair in pairs:
                if pair in raw_scores:
                    ema_score = ema_scorer.update(pair, raw_scores[pair], utterance_counts)
                    results[pair][trial_idx].append(ema_score)
                else:
                    # スコアが得られなかった場合は前回値を使用（初回なら0.0）
                    prev_score = ema_scorer.scores.get(pair, 0.0)
                    results[pair][trial_idx].append(prev_score)
                    if DEBUG:
                        print(f"    ⚠️ {pair} のスコアが得られませんでした（前回値 {prev_score:+.1f} を使用）")

        # このラウンドの結果サマリー
        if DEBUG:
            print(f"\n📈 ラウンド {round_idx} 結果サマリー:")
            for pair in pairs:
                trial_scores = [results[pair][t][-1] for t in range(NUM_TRIALS)]
                avg = sum(trial_scores) / len(trial_scores)
                print(f"  {pair[0]}-{pair[1]}: {trial_scores} → 平均 {avg:+.1f}")

    # CSV出力
    if DEBUG:
        print(f"\n{'='*80}")
        print(f"💾 CSV出力: {OUTPUT_FILE}")
        print(f"{'='*80}")

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # ヘッダー行
        writer.writerow(['ペア-ラウンド', '試行1', '試行2', '試行3', '試行4', '試行5', '平均'])

        # ペアごとにまとめて出力
        for pair in pairs:
            pair_name = f"{pair[0]}-{pair[1]}"

            # 各ラウンドの結果を出力
            for round_idx in range(len(round_end_indices)):
                row_label = f"{round_idx + 1}-{pair_name}"

                # 5試行分のスコア
                trial_scores = [
                    round(results[pair][trial_idx][round_idx], 1)
                    for trial_idx in range(NUM_TRIALS)
                ]

                # 平均（小数第1位に四捨五入）
                avg = round(sum(trial_scores) / len(trial_scores), 1)

                writer.writerow([row_label] + trial_scores + [avg])

    if DEBUG:
        print(f"✅ 完了: {OUTPUT_FILE} を出力しました")
        print(f"\n📊 出力形式:")
        print(f"  - A列: ペア-ラウンド（例: 1-A-B, 2-A-B, ...）")
        print(f"  - B〜F列: 5回の試行結果（EMA適用済み）")
        print(f"  - G列: 平均値")


if __name__ == "__main__":
    main()

import os
import re
import networkx as nx
from itertools import combinations
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from intervention_planner import InterventionPlanner
import sys
import config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params

load_dotenv()
_CFG = config.get_config()


# === 会話ログ読み込み ===
def load_session_logs(filepath: str) -> List[Dict]:
    logs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\[(.*?)\]\s+\[(.*?)\]\s+(.*)", line.strip())
            if match:
                timestamp, speaker, utterance = match.groups()
                logs.append(
                    {"time": timestamp, "speaker": speaker, "utterance": utterance}
                )
    return logs


# === GPTでペアごとの関係性を推定 ===
def estimate_relationship_scores(
    logs: List[Dict], participants: List[str]
) -> Dict[Tuple[str, str], float]:
    conversation_text = "\n".join(
        [f"[{log['speaker']}] {log['utterance']}" for log in logs]
    )
    pairs = list(combinations(participants, 2))
    pair_lines = "\n".join([f"- {a} × {b}" for a, b in pairs])
    output_format = "\n".join([f"{a}-{b}:" for a, b in pairs])

    prompt = f"""
以下の会話を読み、参加者それぞれの「仲の良さ（親密度）」を -1.0 〜 1.0 の間の実数（小数第1位まで） で評価してください。
-1.0 は明確な対立、0 は中立、+1.0 は非常に親しい関係を表します。
評価対象は以下のペアです：
{pair_lines}

会話：
{conversation_text}

出力形式：
{output_format}
"""

    # Azure OpenAI (RELATION_MODEL) を使用
    client, deployment = get_azure_chat_completion_client(
        _CFG.llm, model_type="relation"
    )
    if not client or not deployment:
        raise RuntimeError("Failed to obtain Azure OpenAI client for relation scoring.")

    messages = [{"role": "user", "content": prompt}]
    params = build_chat_completion_params(
        deployment, messages, _CFG.llm, temperature=0.3
    )
    response = client.chat.completions.create(**params)

    scores = {}
    for line in response.choices[0].message.content.strip().split("\n"):
        if ":" in line:
            print(line)
            pair, value = line.strip().split(":")
            a, b = [p.strip() for p in pair.split("-")]
            try:
                scores[(a, b)] = float(value.strip())
            except ValueError:
                print(f"⚠️ 無効なスコア: {line}")
    return scores


# === グラフと三角形構造の構築 ===
def build_graph_and_triangles(
    scores: Dict[Tuple[str, str], float],
) -> Tuple[nx.Graph, Dict[Tuple[str, str, str], Tuple[str, float]]]:
    G = nx.Graph()
    for (a, b), s in scores.items():
        G.add_edge(a, b, score=s)

    triangle_scores = {}
    for a, b, c in combinations(G.nodes, 3):
        if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a):
            s = {
                (a, b): G[a][b]["score"],
                (b, c): G[b][c]["score"],
                (c, a): G[c][a]["score"],
            }
            signs = ["+" if s[p] >= 0 else "-" for p in [(a, b), (b, c), (c, a)]]
            structure = "".join(signs)
            average = sum(s.values()) / 3
            triangle_scores[(a, b, c)] = (structure, average)
    return G, triangle_scores


# === メイン処理 ===
if __name__ == "__main__":
    filepath = f"logs/conversation{sys.argv[1]}.txt"
    logs = load_session_logs(filepath)
    participants = list({log["speaker"] for log in logs})

    print("🧠 GPTで関係性を推定中...")
    relationship_scores = estimate_relationship_scores(logs, participants)
    graph, triangle_scores = build_graph_and_triangles(relationship_scores)

    print("🤖 ロボットの介入を計画中...")
    planner = InterventionPlanner(graph, triangle_scores)
    plan = planner.plan_intervention()
    if plan:
        utterance = planner.generate_robot_utterance(plan, logs)
        print("✅ ロボットの発言:")
        print(utterance)
    else:
        print("🔍 介入すべき対象が見つかりませんでした。")

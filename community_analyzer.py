import os
import threading
import queue
from collections import defaultdict, deque
from typing import Dict, Tuple
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib
import socketio
import re
import time
import socket
from intervention_planner import InterventionPlanner
from datetime import datetime
import config

pepper_ip = "192.168.11.48"  # PepperのIPアドレス
pepper_port = 2002  # Android アプリのポート
use_robot = True  # Pepperを使用するかどうか
robot_included = False  # ロボットを関係性学習に組み込むかどうか
mode = "proposal"  # "proposal", "few_utterances", "random_target"


matplotlib.use("Agg")  # GUI 非対応の描画エンジンを指定
socketio_cli = socketio.Client()

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
plt.rcParams['font.family'] = 'Hiragino Sans'


def try_connect_socketio(url="http://localhost:8888", max_retries=10, interval_sec=2):
    """Flask Socket.IO サーバへの接続をリトライしながら試みる"""
    for attempt in range(1, max_retries + 1):
        try:
            socketio_cli.connect(url)
            print("✅ Socket.IO に接続しました")
            return
        except Exception as e:
            print(f"⚠️ Socket.IO 接続失敗 (試行 {attempt}/{max_retries}): {e}")
            time.sleep(interval_sec)
    print("❌ Socket.IO に接続できませんでした。Web UI 機能は無効になります。")


try_connect_socketio()


def send_to_pepper(message: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((pepper_ip, pepper_port))
            s.sendall(f"say:{message}\n".encode('utf-8'))
            # 応答待ち（オプション）
            # response = s.recv(1024).decode('utf-8')
            # print(f"🤖 Pepper応答: {response}")
    except Exception as e:
        print(f"⚠️ Pepperへの送信失敗: {e}")


def send_to_pepper_async(message: str):
    threading.Thread(target=send_to_pepper, args=(message,), daemon=True).start()


# === Pepperの腕を持ち上げるアニメーションを送信 ===
def perform_arm_lift(self, seconds=2):
    """Pepperの腕をseconds秒間持ち上げるアニメーション指示を送信"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((pepper_ip, pepper_port))
            # アニメーションコマンド: "anim:<動作名>,<秒数>"
            s.sendall(f"anim:liftRightArm,{seconds}\n".encode('utf-8'))
    except Exception as e:
        print(f"⚠️ Pepperアニメーション送信失敗: {e}")


def perform_arm_lift_async(self):
    threading.Thread(target=self.perform_arm_lift, daemon=True).start()


class CommunityAnalyzer:
    def __init__(self, decay_factor=1.5):
        self.graph_count = 0
        self.scores = defaultdict(float)
        self.history = defaultdict(lambda: deque(maxlen=3))  # ペアごとの過去発話数（最大3件）
        self.decay_factor = decay_factor
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # === ワーカースレッドで非同期に処理 ===
    def _worker(self):
        while True:
            session = self.task_queue.get()
            self._analyze(session)
            self.task_queue.task_done()

    def update(self, session):
        self.task_queue.put(session)
        # print(f"関係性推定開始：{datetime.now()}")

    # === セッションごと関係性を更新 ===
    def _analyze(self, session_logs):
        participants = list({log["speaker"] for log in session_logs})
        # ロボットは常に除外
        # participants = [p for p in {log["speaker"] for log in session_logs} if p != "ロボット"]
        print(f"👥 参加者: {participants}")
        if len(participants) < 2:
            print("⚠️ 参加者が1人以下のためスキップ")
            return

        # セッション内の発話数を記録
        utterance_counts = defaultdict(int)
        for log in session_logs:
            utterance_counts[log["speaker"]] += 1
        print(f"🗣️ 発話数: {utterance_counts}")

        gpt_scores = self._get_gpt_friendship_scores(session_logs, participants)
        print(f"🧠 GPTスコア: {gpt_scores}")

        for (a, b), score in gpt_scores.items():
            key = tuple(sorted([a, b]))
            session_utterance = min(utterance_counts[a], utterance_counts[b])
            past_utterances = self.history[key]
            total_past = sum(past_utterances)

            x_t = score
            if key not in self.scores:
                self.scores[key] = x_t
                print(f"🆕 初期スコア: {key} = {x_t:.2f}")
            else:
                ratio = session_utterance / (session_utterance + total_past)  # 今までの発話数に対するセッション内の発話数の比率
                alpha = max(0.01, min(1.0, self.decay_factor * ratio))  # decay_factorを掛けて時間減衰を考慮。しかし、αは0.01以上1.0以下に制限
                print(f"🔢 α計算: {key}, session={session_utterance}, past={total_past}, α={alpha:.2f}")
                prev = self.scores[key]
                updated = alpha * x_t + (1 - alpha) * prev
                self.scores[key] = updated
                print(f"🔁 EMA更新: {key} = {alpha:.2f}×{x_t:.2f} + {(1-alpha):.2f}×{prev:.2f} → {updated:.2f}")
            # 直近履歴に追加（最大3件）
            self.history[key].append(session_utterance)

        self._draw_graph()

        if socketio_cli.connected:
            socketio_cli.emit("graph_updated", {"index": self.graph_count, "log_root": os.path.basename(config.LOG_ROOT)})  # === グラフ更新通知をWebに送信 ===

        # print(f"関係性推定終了：{datetime.now()}")

        if "ロボット" not in participants:
            # ロボット介入戦略の実行
            self._run_intervention(session_logs=session_logs)

    # === GPTに仲の良さを尋ねるプロンプト ===
    def _get_gpt_friendship_scores(self, session_logs, participants):
        conversation = "\n".join([f"[{log['speaker']}] {log['utterance']}" for log in session_logs])
        pair_lines = "\n".join([f"- {a} × {b}" for a, b in combinations(participants, 2)])
        output_format = "\n".join([f"{a}-{b}:" for a, b in combinations(participants, 2)])

#     prompt = f"""
# 以下の会話を読み、参加者それぞれの「仲の良さ（親密度）」を -1.0 〜 1.0 の間の実数（小数第1位まで）で評価してください。
# -1.0 は明確な対立、0 は中立、+1.0 は非常に親しい関係を表します。
# 評価対象は以下のペアです：

# {pair_lines}

# 会話：
# {conversation}

# 出力形式：
# {output_format}
# """
        prompt = f"""
以下の会話を読み、参加者それぞれの「仲の良さ（親密度）」を -1.0 〜 +1.0 の間の**実数（小数第1位まで）**で評価してください。
0.0 は特に親しさも対立も感じない「中立的な状態」です。
そこから -1.0（強い対立） 〜 +1.0（非常に親しい） に向けて、どれくらい離れているかを評価してください。

具体例：
-1.0 例：皮肉、批判、無視、相手を無視して話を進める
+1.0 例：共感、褒める、相手に話題を振る、一緒に行動する

評価対象は以下のペアです：
{pair_lines}

会話：
{conversation}

出力形式：
{output_format}
"""

        res = client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return self._parse_scores_from_response(res.choices[0].message.content)

    # === GPTの返答を辞書に変換 ===
    def _parse_scores_from_response(self, response_text):
        scores = {}
        lines = response_text.strip().split("\n")
        for line in lines:
            if ":" in line:
                pair, score = line.split(":")
                a, b = [x.strip() for x in pair.split("-")]
                # 数値部分だけを抽出（-1.0 ～ 1.0 を想定）
            match = re.search(r"-?\d+(\.\d+)?", score)
            if match:
                scores[(a, b)] = float(match.group())
            else:
                print(f"⚠️ 数値抽出失敗: {line.strip()}")
        return scores

    def _run_intervention(self, session_logs):
        """
        関係性グラフ・三角形構造をもとに、介入戦略を自動生成して実行する
        """
        # print(f"ロボット介入開始：{datetime.now()}")
        planner = InterventionPlanner(graph=self._build_graph_object(), triangle_scores=self._compute_triangle_scores(), mode=mode)
        plan = planner.plan_intervention(session_logs=session_logs)

        if not plan:
            print("🤖 介入対象なし（安定状態）")
            return
        else:
            if use_robot:
                # Pepperの腕を2秒間持ち上げるアニメーション
                # self.perform_arm_lift_async()
                # フィラーとして「あのー」を挟む
                send_to_pepper_async("あのー")

        utterance = planner.generate_robot_utterance(plan, session_logs)

        # ロボットの発話ログを構成
        robot_log = {
            "time": datetime.now(),
            "speaker": "ロボット",
            "utterance": utterance
        }

        # ロボット発話したら、次の５発話分はロボットを識別対象にする
        import realtime_communicator as rc
        rc.set_robot_count(5)
        print("🤖 ロボット発話を識別対象に設定（次の5発話分）")

        # 画面表示
        if socketio_cli.connected:
            socketio_cli.emit("robot_speak", {
                "speaker": "ロボット",
                "utterance": utterance
            })
            print(f"💬 ロボット発話送信: {utterance}")

        if use_robot:
            # --- Pepperに喋らせる（非同期） ---
            send_to_pepper_async(utterance)

        # print(f"ロボット介入終了：{datetime.now()}")

        if robot_included:
            # 🔁 ロボットの発話も関係性学習に反映させる
            session_with_robot = session_logs + [robot_log]
            self.update(session_with_robot)

    def _build_graph_object(self) -> nx.Graph:
        G = nx.Graph()
        for (a, b), score in self.scores.items():
            a, b = sorted([a, b])
            G.add_edge(a, b, score=score)
        return G

    # === NetworkXでグラフ描画 ===
    def _draw_graph(self):
        G = self._build_graph_object()

        # 1. まず計算だけする
        preliminary_weights = {}
        for u, v in G.edges():
            score = G[u][v]['score']
            scaled = (score + 1.0) / 2.0  # 0.0～1.0に変換
            preliminary_weights[(u, v)] = scaled ** 6 * 200

        # 2. 最大weightを取得
        max_weight = max(preliminary_weights.values())
        min_weight = max_weight / 7
        print(f"🔎 最大weight: {max_weight:.2f}, 最小weight設定: {min_weight:.2f}")

        # 3. 最小値を考慮してweightをセット
        for (u, v), prelim in preliminary_weights.items():
            adjusted_weight = max(min_weight, prelim)
            G[u][v]['weight'] = adjusted_weight
            print(f"スコア調整: {u} - {v}: prelim={prelim:.2f} → weight={adjusted_weight:.2f}")

        # 配置計算（スコアが強いほど引き合う）
        pos = nx.spring_layout(G, weight='weight', seed=42)

        # 線の太さ（親密度の強さに応じて）
        edge_weights = [max(0.5, 5 * abs(G[u][v]['score'])) for u, v in G.edges()]

        # スコアラベル（±1.0）
        edge_labels = {(u, v): f"{G[u][v]['score']:.1f}" for u, v in G.edges()}

        # 敵対関係は赤、親密な関係は青。0以上は青、0未満は赤
        edge_colors = ['red' if G[u][v]['score'] < 0 else 'skyblue' for u, v in G.edges()]

        plt.figure(figsize=(6, 6))  # 図のキャンバスを正方形サイズで作る
        nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1000)  # ノード（＝人）を「空色」で大きめ（サイズ1000）に描画する
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors)  # 辺（＝関係性）を重みに応じた太さ（edge_weights）で描画する
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="Hiragino Sans")  # 各ノードに人の名前をラベルとして表示
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_family="Hiragino Sans")  # 各辺に「スコア」を表示（小数点1桁まで）

        plt.title("関係性グラフ")
        plt.axis("off")
        plt.tight_layout()
        # グラフ画像をログディレクトリに連番で保存
        image_path = os.path.join(config.LOG_ROOT, f"relation_graph{self.graph_count}.png")
        plt.savefig(image_path)
        plt.close()
        print(f"✅ グラフ画像保存完了: {image_path}")
        self.graph_count += 1

    # === 三角形の構造とスコア平均を計算 ===
    def _compute_triangle_scores(self) -> Dict[Tuple[str, str, str], Tuple[str, float]]:
        from itertools import combinations

        def triangle_structure(a, b, c, g):
            s = {
                (a, b): g[a][b]['score'],
                (b, c): g[b][c]['score'],
                (c, a): g[c][a]['score']
            }
            signs = ['+' if s[pair] >= 0 else '-' for pair in [(a, b), (b, c), (c, a)]]  # スコアの符号を取得
            return ''.join(signs), sum(s.values()) / 3  # 例：++-, スコアの平均

        G = self._build_graph_object()
        triangle_scores = {}
        # 三角形の組み合わせを取得
        for a, b, c in combinations(G.nodes, 3):
            if G.has_edge(a, b) and G.has_edge(b, c) and G.has_edge(c, a):
                struct, avg = triangle_structure(a, b, c, G)
                triangle_scores[(a, b, c)] = (struct, avg)
                print(f"🔺 三角形: {a}, {b}, {c} → 構造: {struct}, スコア平均: {avg:.2f}")
        return triangle_scores

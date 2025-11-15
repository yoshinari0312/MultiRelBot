import os
import networkx as nx
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from typing import Any
import random
import config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params

load_dotenv()
_CFG = config.get_config()


class InterventionPlanner:
    # 過去のロボット発話を保存するリスト（全モード共通）
    past_utterances: List[str] = []

    def __init__(
        self,
        graph: nx.Graph,
        triangle_scores: Dict[Tuple[str, str, str], Tuple[str, float]],
        isolation_threshold: float = 0.0,
        mode: str = "proposal",
        num_participants: int = 3,
    ):
        """
        :param graph: NetworkXグラフ。ノードは人物、エッジは関係性スコア（-1〜1）を持つ
        :param triangle_scores: 三角形ごとの構造タイプ（---, ++- など）とスコア平均
        :param isolation_threshold: 孤立判定のスコア閾値（これ未満の関係しか持たないと孤立とみなす）
        :param mode: 介入モード ("proposal", "few_utterances", "random_target")
        :param num_participants: 参加者数 (3 or 4)
        """
        self.graph = graph
        self.triangle_scores = triangle_scores
        self.theta_iso = isolation_threshold
        self.mode = mode  # "proposal", "few_utterances", "random_target"
        self.num_participants = num_participants

    def detect_structural_isolation(self) -> Optional[str]:
        """
        Step 1: 孤立を検出
        - ノードの全エッジが theta_iso 以下なら孤立とみなす
        - 複数の孤立ノードがある場合、最も改善が見込めるノード（負のエッジの絶対値が最小）を選択
        """
        candidates = []
        for node in self.graph.nodes():
            edges = [
                self.graph[node][nbr]["score"] for nbr in self.graph.neighbors(node)
            ]
            # 全てのエッジが theta_iso 以下なら孤立候補
            if len(edges) >= 2 and all(edge < self.theta_iso for edge in edges):
                # 負のエッジから最小絶対値（最も改善しやすい）を見つける
                negative_edges = [e for e in edges if e < 0]
                if negative_edges:
                    min_abs_score = min(abs(e) for e in negative_edges)
                    candidates.append((node, min_abs_score))
                else:
                    # 負のエッジがない場合は平均スコアを使用
                    avg_score = sum(edges) / len(edges)
                    candidates.append((node, abs(avg_score)))

        if not candidates:
            return None

        # 最小絶対値（最も改善が見込める）ノードを返す
        return min(candidates, key=lambda x: x[1])[0]

    def sort_triangles(self) -> List[Tuple[str, str, str]]:
        """
        Step 2: 三角形を構造タイプとスコア平均に基づいてソート
        優先度：---（最も不安定） > ++-系（認知的不協和） > その他
        スコア平均が低いものを優先
        """

        # 三角形の構造タイプを優先度に基づいてソートするためのヘルパー関数
        def structure_priority(struct_type: str) -> int:
            return {"---": 0, "++-": 1, "+-+": 1, "-++": 1}.get(struct_type, 999)

        # 三角形を構造タイプとスコア平均でソート
        sorted_triangles = sorted(
            self.triangle_scores.items(),
            key=lambda item: (
                structure_priority(item[1][0]),  # 構造タイプの優先度（--- が最優先）
                item[1][1],  # スコア平均（低い方が優先）
            ),
        )
        return [
            tri[0] for tri in sorted_triangles
        ]  # [(a, b, c), (b, c, d), ...] の形で返す

    def select_intervention_triangle(self) -> Optional[Tuple[str, str, str]]:
        """
        Step 3: ソートされた三角形から、介入対象とするもの（--- または ++-系）を選ぶ
        """
        sorted_tri = self.sort_triangles()
        for tri in sorted_tri:
            struct, _ = self.triangle_scores[tri]  # 構造タイプを取得
            print(struct)
            if struct in ("---", "++-", "+-+", "-++"):
                return tri
        return None

    def choose_target_edge(self, triangle: Tuple[str, str, str]) -> Tuple[str, str]:
        """
        Step 4: 三角形の構造に応じて、改善対象エッジを選定
        - --- 構造：絶対値が最も小さいエッジ（最も改善しやすい弱い対立）
        - ++-系 構造：負のエッジ（弱い関係）
        """
        a, b, c = triangle
        scores = {
            (a, b): self.graph[a][b]["score"],
            (b, c): self.graph[b][c]["score"],
            (c, a): self.graph[c][a]["score"],
        }

        if self.triangle_scores[triangle][0] == "---":
            # 絶対値が最小のエッジを選択
            edges = [
                ((a, b), abs(scores[(a, b)])),
                ((b, c), abs(scores[(b, c)])),
                ((c, a), abs(scores[(c, a)])),
            ]
            min_edge = min(edges, key=lambda x: x[1])
            return min_edge[0]

        elif (
            self.triangle_scores[triangle][0] == "++-"
            or self.triangle_scores[triangle][0] == "+-+"
            or self.triangle_scores[triangle][0] == "-++"
        ):
            # 負のエッジを返す
            for (u, v), w in scores.items():
                if w < 0:
                    return (u, v)

        # フォールバック（通常は到達しない）
        return (triangle[0], triangle[1])

    def plan_intervention(
        self, session_logs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        介入戦略の全体フローを実行し、介入対象ノードとその構造を返す
        """

        # --- ファシリテーション発話 ---
        if self.mode == "few_utterances" and session_logs:
            print(f"🤖 ファシリテーション発話モード")
            return {"type": "few_utterances"}

        # --- ランダム対象に固定フレーズ ---
        if self.mode == "random_target" and session_logs:
            participants = list({log["speaker"] for log in session_logs})
            target = random.choice(participants)
            print(f"🤖 ランダム対象: {target}さん")
            return {"type": "random_target", "target": target}

        # Step 1: 孤立検出
        isolated = self.detect_structural_isolation()
        if isolated:
            # 孤立ノードの負のエッジから絶対値が最小のものを選択
            edges = [
                (isolated, nbr, self.graph[isolated][nbr]["score"])
                for nbr in self.graph.neighbors(isolated)
            ]
            negative_edges = [(n1, n2, score) for n1, n2, score in edges if score < 0]
            if negative_edges:
                # 絶対値が最小（改善しやすい）エッジを選択
                target_edge = min(negative_edges, key=lambda x: abs(x[2]))
                return {
                    "type": "isolation",
                    "target": isolated,
                    "edge": (target_edge[0], target_edge[1]),
                }
            else:
                # 負のエッジがない場合は、最もスコアが低いエッジを選択
                target_edge = min(edges, key=lambda x: x[2])
                return {
                    "type": "isolation",
                    "target": isolated,
                    "edge": (target_edge[0], target_edge[1]),
                }

        # Step 2〜4: 不安定三角形に基づく介入
        triangle = self.select_intervention_triangle()
        if triangle:
            target_edge = self.choose_target_edge(triangle)
            return {
                "type": "triangle",
                "structure": self.triangle_scores[triangle][
                    0
                ],  # 例: '---', '++-', '+-+', '-++'
                "triangle": triangle,  # 例: ('A', 'B', 'C')
                "edge": target_edge,  # 例: ('A', 'B')（改善対象エッジ）
            }

        # Step 5: 安定状態だが弱リンク(0.0～0.2)がある場合 → 関係形成支援介入
        # ※ 以下の処理はコメントアウト（孤立または不安定の時のみ介入する仕様に変更）
        # weak_pairs_scores = []
        # for u, v, data in self.graph.edges(data=True):
        #     score = data.get("score", 0.0)
        #     if 0.0 <= score <= 0.2:
        #         weak_pairs_scores.append(((u, v), score))
        # if weak_pairs_scores:
        #     # 最もスコアが小さいペアを1つだけ選択
        #     target_pair, _ = min(weak_pairs_scores, key=lambda x: x[1])
        #     return {"type": "promotion", "pairs": [target_pair]}

        return None  # 介入なし

    def generate_robot_utterance(
        self, plan: Dict[str, str], session_logs: List[Dict]
    ) -> str:
        """
        会話履歴と介入プランをもとに、GPTによるロボット発話を生成する
        :param plan: plan_intervention() の出力（dict）
        :param session_logs: セッション内の会話ログ（辞書のリスト）
        :return: ロボットが発話すべき内容（文字列）
        """
        context = "\n".join(
            [f"[{log['speaker']}] {log['utterance']}" for log in session_logs]
        )  # 会話履歴をテキスト化

        # ── ファシリテーション発話生成 ──
        if plan["type"] == "few_utterances":
            full_prompt = f"""
現在の会話履歴：
{context}

――――――――――

あなたは会話をサポートするファシリテーションロボットです。
上記の会話をもとに、ロボットの適切な発言を1文で考えてください。

【発話に関するルール】
- 出力はロボットの自然な日本語1文のみ
- 「ロボット：」などのラベルは不要
- 説明や理由は述べない
- 会話の話題を大きく変えず、直前の会話の流れに基づいて自然な一言にする
"""
            # 過去発話回避をプロンプトに追加
            if self.past_utterances:
                full_prompt += "\n\n【注意】過去に以下の発言を行っています。似た内容は繰り返さないようにしてください:\n"
                for utt in self.past_utterances:
                    full_prompt += f"- {utt}\n"

            client, deployment = get_azure_chat_completion_client(
                _CFG.llm, model_type="robot"
            )
            if not client or not deployment:
                raise RuntimeError(
                    "Failed to obtain Azure OpenAI client for robot utterance generation."
                )

            messages = [{"role": "user", "content": full_prompt}]
            # config.yamlから介入用の温度パラメータを読み込み
            intervention_cfg = getattr(_CFG, "intervention", None)
            temperature = (
                getattr(intervention_cfg, "temperature", 1.0)
                if intervention_cfg
                else 1.0
            )
            params = build_chat_completion_params(
                deployment, messages, _CFG.llm, temperature=temperature
            )
            res = client.chat.completions.create(**params)

            new_utt = (
                res.choices[0]
                .message.content.replace("[ロボット]", "")
                .replace("「", "")
                .replace("」", "")
                .strip()
            )
            self.past_utterances.append(new_utt)
            print(f"🤖 past_utterances: {self.past_utterances}\n")
            return new_utt

        elif plan["type"] == "random_target":
            return f"{plan['target']}さんはどう思いますか？"

        # ──────────────────────────────────────
        # 以下、proposal モード（構造ベース介入）の新プロンプト
        # ──────────────────────────────────────

        # 【共通 system プロンプト】（参加者数に応じて動的に生成）
        if self.num_participants == 3:
            participants_desc = "三者会話（A, B, C）"
            group_desc = "三者"
        elif self.num_participants == 4:
            participants_desc = "四者会話（A, B, C, D）"
            group_desc = "四者"
        else:
            participants_desc = f"{self.num_participants}者会話"
            group_desc = f"{self.num_participants}者"

        system_prompt = f"""あなたは、{participants_desc}における「関係エッジ」を安定（ポジティブ）にするための
ロボット発話を、自然で適切な日本語1文で生成する専門AIアシスタントです。

【目的】
- 会話文脈と現在の関係スコアに基づき、指定された「改善対象エッジ」を
  より安定的で協力的な関係に近づける発言を生成します。
- 発言は会話の流れを遮らず、自然に文脈へ溶け込む必要があります。

【発話に関するルール】
- 出力はロボットの自然な日本語1文のみ。
- 「ロボット：」などのラベルは不要。
- 説明や理由は述べない。
- 相手の名前を必ず入れてください。片方に話す場合は1名、両方に話す場合は2名の名前を入れてください。
- 会話の話題を大きく変えず、直前の会話の流れに基づいて自然な一言にしてください。

【改善の考え方】
- 改善対象エッジの2名が、互いに理解・協力・共感しやすくなるよう促す発言を選びます。
- その際、「どちらに話しかけるか」または「両方に言うか」は、あなたが最も効果的と判断してください。
  ただし、必ず名前を入れてください。"""

        # 過去発話回避をsystemに統合
        if self.past_utterances:
            system_prompt += "\n\n【注意】過去に以下の発言を行っています。似た内容は繰り返さないようにしてください:\n"
            for utt in self.past_utterances:
                system_prompt += f"- {utt}\n"

        # 条件別追加ブロック
        additional_context = ""

        if plan["type"] == "isolation":
            # 孤立検出の場合
            node1, node2 = plan["edge"]
            isolated_node = plan["target"]  # 孤立しているノード
            other_node = node2 if node1 == isolated_node else node1
            print(f"🤖 孤立検出: {isolated_node}さん（エッジ: {node1} ⇄ {node2}）")
            additional_context = f"""
【追加条件：孤立ノードの改善】
- 改善対象エッジ：{node1}さん ⇄ {node2}さん
- {isolated_node}さんは、現在グループ全体から距離を置かれており、孤立している状態です。
- {isolated_node}さんが{other_node}さんとの関係を再構築しやすくなるような、
  優しく・具体的で・会話の流れに合った一言を生成してください。
- 発言対象はあなたが決めて構いません。
  （{isolated_node}さんに直接話す／{other_node}さんに話す／両方に話すなど最適な方法を選ぶ）
- ただし、必ず名前を入れてください。"""

        elif plan["type"] == "triangle":
            node1, node2 = plan["edge"]
            struct = plan["structure"]
            triangle = plan["triangle"]  # (A, B, C)

            # 三角形の各エッジのスコアを取得
            a, b, c = triangle
            score_ab = self.graph[a][b]["score"]
            score_bc = self.graph[b][c]["score"]
            score_ca = self.graph[c][a]["score"]

            # スコアの説明を生成
            def score_desc(score):
                if score < -0.5:
                    return f"{score:.1f}（強い対立）"
                elif score < 0:
                    return f"{score:.1f}（対立）"
                elif score < 0.3:
                    return f"{score:.1f}（中立・弱い）"
                elif score < 0.7:
                    return f"{score:.1f}（良好）"
                else:
                    return f"{score:.1f}（強い協力）"

            relationships = f"""- {a}さんと{b}さんの関係: {score_desc(score_ab)}
- {b}さんと{c}さんの関係: {score_desc(score_bc)}
- {c}さんと{a}さんの関係: {score_desc(score_ca)}"""

            if struct == "---":
                print(f"🤖 ---検出: エッジ ({node1}, {node2})")
                additional_context = f"""
【追加条件：全員が対立している不安定（---）構造】
- 三角形の構成員：{a}さん、{b}さん、{c}さん
- 現在の関係性：
{relationships}
- 改善対象エッジ：{node1}さん ⇄ {node2}さん
- 改善対象エッジは、{group_desc}間の対立の中で「最も関係が修復しやすい弱い対立」に該当します。
- この2人の関係性をよくすることで、+--の安定三角形を目指したい状況です。
- あなたは、この2人のどちらに話しかけるか（または2人へまとめて話すか）を、
  会話文脈に基づいて最も自然で効果的な形で判断してください。
- ただし、必ず名前を入れてください。
- 発話内容は、敵対ではなく協力・理解・確認・視点整理につながる、
  過度に踏み込みすぎない一文にしてください。"""

            elif struct in ("++-", "+-+", "-++"):
                # 改善対象エッジ以外の人物（良好な関係を持つ人物）を特定
                third_person = [p for p in triangle if p not in (node1, node2)][0]

                print(f"🤖 ++-系検出: エッジ ({node1}, {node2})")
                additional_context = f"""
【追加条件：部分的な不均衡（++-構造）】
- 三角形の構成員：{a}さん、{b}さん、{c}さん
- 現在の関係性：
{relationships}
- 改善対象エッジ：{node1}さん ⇄ {node2}さん
- 改善対象エッジは、他の2エッジが良好である一方、この1本だけ弱い関係です。
- この弱い関係を自然に強めることで、{group_desc}のバランスが良くなります。
- {third_person}さんは、{node1}さんと{node2}さんの両方と良好な関係を持っています。
- 誰に話すべきかは固定しません。
  {third_person}さんを引き合いに出して{node1}さんや{node2}さんに働きかけるか、
  {node1}さんと{node2}さんの両者に向けた声かけをするかは、あなたが最も効果的と判断する形にして構いません。
- ただし、必ず名前を入れてください。
- ただし「橋渡し」「仲介」などの直接的な語は使わず、
  会話の流れに合った自然な形で方向性を整える一言にしてください。"""

        # userプロンプト（会話履歴）
        user_prompt = f"""現在の会話履歴：
{context}

――――――――――

上記の会話を踏まえて、ロボットの発言を1文で生成してください。"""

        # メッセージ構築
        messages = [
            {"role": "system", "content": system_prompt + "\n" + additional_context},
            {"role": "user", "content": user_prompt},
        ]

        # Azure OpenAI を使用
        client, deployment = get_azure_chat_completion_client(
            _CFG.llm, model_type="robot"
        )
        if not client or not deployment:
            raise RuntimeError(
                "Failed to obtain Azure OpenAI client for robot utterance generation."
            )

        # config.yamlから介入用の温度パラメータを読み込み
        intervention_cfg = getattr(_CFG, "intervention", None)
        temperature = (
            getattr(intervention_cfg, "temperature", 1.0) if intervention_cfg else 1.0
        )
        params = build_chat_completion_params(
            deployment, messages, _CFG.llm, temperature=temperature
        )
        res = client.chat.completions.create(**params)

        new_utt = (
            res.choices[0]
            .message.content.replace("[ロボット]", "")
            .replace("「", "")
            .replace("」", "")
            .strip()
        )
        self.past_utterances.append(new_utt)
        print(f"🤖 past_utterances: {self.past_utterances}\n")
        return new_utt

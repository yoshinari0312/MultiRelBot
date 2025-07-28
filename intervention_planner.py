import os
import networkx as nx
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class InterventionPlanner:
    def __init__(self, graph: nx.Graph, triangle_scores: Dict[Tuple[str, str, str], Tuple[str, float]], isolation_threshold: float = 0.0):
        """
        :param graph: NetworkXグラフ。ノードは人物、エッジは関係性スコア（-1〜1）を持つ
        :param triangle_scores: 三角形ごとの構造タイプ（---, ++- など）とスコア平均
        :param isolation_threshold: 孤立判定のスコア閾値（これ以下の関係しか持たないと孤立とみなす）
        """
        self.graph = graph
        self.triangle_scores = triangle_scores
        self.theta_iso = isolation_threshold

    def detect_structural_isolation(self) -> Optional[str]:
        """
        Step 1: 完全孤立ノード（すべての隣接スコアがθ_iso未満）を全て列挙し、
        平均スコアが最も低いノードを返す
        """
        candidates = []

        for node in self.graph.nodes:
            edges = [self.graph[node][nbr]['score'] for nbr in self.graph.neighbors(node)]
            if len(edges) >= 2 and all(edge < self.theta_iso for edge in edges):
                avg_score = sum(edges) / len(edges)
                candidates.append((node, avg_score))

        if not candidates:
            return None

        # 平均スコアが最も低いノードを返す
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
                item[1][1]                       # スコア平均（低い方が優先）
            )
        )
        return [tri[0] for tri in sorted_triangles]  # [(a, b, c), (b, c, d), ...] の形で返す

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

    def choose_target_node(self, triangle: Tuple[str, str, str]) -> str:
        """
        Step 4: 三角形の構造に応じて、ロボットが話しかけるべき対象ノードを選定
        - --- 構造：他の2人とのスコア平均が最も高いノードを選ぶ（調停者）
        - ++-系 構造：2本の+を持つノード（共通の友人）を選ぶ（橋渡し役）
        """
        a, b, c = triangle
        scores = {
            (a, b): self.graph[a][b]['score'],
            (b, c): self.graph[b][c]['score'],
            (c, a): self.graph[c][a]['score'],
        }

        if self.triangle_scores[triangle][0] == '---':
            avg_scores = {
                a: (scores[(a, b)] + scores[(c, a)]) / 2,
                b: (scores[(a, b)] + scores[(b, c)]) / 2,
                c: (scores[(b, c)] + scores[(c, a)]) / 2
            }
            return max(avg_scores.items(), key=lambda x: x[1])[0]  # スコア平均が最も高いノードを返す

        elif self.triangle_scores[triangle][0] == '++-' or self.triangle_scores[triangle][0] == '+-+' or self.triangle_scores[triangle][0] == '-++':
            counts = {n: 0 for n in triangle}  # エッジが+の数をカウントするための辞書
            for (u, v), w in scores.items():
                if w > 0:
                    counts[u] += 1
                    counts[v] += 1
            return max(counts.items(), key=lambda x: x[1])[0]

        return triangle[0]

    def plan_intervention(self) -> Optional[Dict[str, str]]:
        """
        介入戦略の全体フローを実行し、介入対象ノードとその構造を返す
        """
        # Step 1: 孤立検出
        isolated = self.detect_structural_isolation()
        if isolated:
            return {
                "type": "isolation",
                "target": isolated
            }

        # Step 2〜4: 不安定三角形に基づく介入
        triangle = self.select_intervention_triangle()
        if triangle:
            target_node = self.choose_target_node(triangle)
            return {
                "type": "triangle",
                "structure": self.triangle_scores[triangle][0],
                "triangle": triangle,
                "target": target_node
            }

        # Step 5: 安定状態だが弱リンク(0.0～0.2)がある場合 → 関係形成支援介入
        weak_pairs_scores = []
        for u, v, data in self.graph.edges(data=True):
            score = data.get("score", 0.0)
            if 0.0 <= score <= 0.2:
                weak_pairs_scores.append(((u, v), score))
        if weak_pairs_scores:
            # 最もスコアが小さいペアを1つだけ選択
            target_pair, _ = min(weak_pairs_scores, key=lambda x: x[1])
            return {
                "type": "promotion",
                "pairs": [target_pair]
            }

        return None  # 介入なし

    def generate_robot_utterance(self, plan: Dict[str, str], session_logs: List[Dict]) -> str:
        """
        会話履歴と介入プランをもとに、GPTによるロボット発話を生成する
        :param plan: plan_intervention() の出力（dict）
        :param session_logs: セッション内の会話ログ（辞書のリスト）
        :return: ロボットが発話すべき内容（文字列）
        """
        context = "\n".join([f"[{log['speaker']}] {log['utterance']}" for log in session_logs])  # 会話履歴をテキスト化
        # print(f"🤖 会話履歴:\n{context}\n")

        if plan['type'] == 'isolation':
            print(f"🤖 孤立検出: {plan['target']}さん")
            name = plan['target']
            full_prompt = f"""
現在の会話履歴（抜粋）：
{context}

現在、{name}さんは他の人から距離を置かれており、対立関係にあります。

あなたは空気を読みながら発言するロボットです。
{name}さんが他の人と関係を改善するために、過去の会話を考慮した上で話しかける内容を考えてください。

例：
- 「{name}さん、〇〇してみたらいいんじゃないかな」
- 「そういえば、{name}さん前に〇〇って言ってましたよね」

自然な日本語で、1文だけ話しかけてください。必ず対話の流れに沿って発言してください。
返答の形式はセリフのみで、「ロボット：」などはつけないでください。
"""

        elif plan['type'] == 'triangle':
            a, b, c = plan['triangle']
            struct = plan['structure']
            target = plan['target']
            others = [n for n in (a, b, c) if n != target]
            other1, other2 = others[0], others[1]

            if struct == '---':
                print(f"🤖 ---検出: {target}さん（{other1}さん・{other2}さん）")
                full_prompt = f"""
現在の会話履歴（抜粋）：
{context}

現在、{target}さんは{other1}さん・{other2}さんとの関係が悪化しています。

あなたは会話をサポートするロボットです。
{target}さんが他の2人のうち、一方と関係を改善するために、過去の会話を考慮した上で{target}さんに話しかける内容を考えてください。

例：
- 「{target}さん、{other1}さんも同じようなこと言ってましたね」
- 「意見は違うけど、□□の部分は似てるかもしれませんね」
- 「{target}さん、〇〇してみたらいいんじゃないかな」

自然な日本語で、1文だけ話しかけてください。必ず対話の流れに沿って発言してください。
返答の形式はセリフのみで、「ロボット：」などはつけないでください。
"""

            elif struct == '++-' or struct == '+-+' or struct == '-++':
                print(f"🤖 ++-系検出: {target}さん（{other1}さん・{other2}さん）")
                full_prompt = f"""
現在の会話履歴（抜粋）：
{context}

現在、{target}さんは{other1}さん・{other2}さんと良好な関係を持っています。
そのため、{target}さんは2人の関係を橋渡しするような発話をすることで、全員がより良い関係を築くことができるかもしれません。

あなたは会話をサポートするロボットです。
{target}さんが他の2人の関係を橋渡しするために、過去の会話を考慮した上で{target}さんに話しかける内容を考えてください。

例：
- 「{target}さん、{other1}さんと{other2}さんに関して、意外と〇〇については共通点ありそうですね」

自然な日本語で、1文だけ話しかけてください。必ず対話の流れに沿って発言してください。
返答の形式はセリフのみで、「ロボット：」などはつけないでください。
"""
            else:
                full_prompt = ""

        elif plan['type'] == 'promotion':
            # 弱リンクを選んで、一文で関係形成支援を促す
            a, b = plan['pairs'][0]
            full_prompt = f"""
現在の会話履歴（抜粋）：
{context}

現在、{a}さんと{b}さんの関係は中立的で、やや弱いつながりです。
あなたは会話をサポートするロボットです。
{a}さんと{b}さんがより深くつながれるよう、過去の会話を考慮した上で共通点を示したり質問を促したりする一文を考えてください。

例：
- 「{a}さんと{b}さんは意見は違うけど、〇〇の部分は似ていますね」
- 「{a}さんと{b}さんの共通点について、もう少し話してみませんか？」
- 「{a}さんの意見について、{b}さんはどう思いましたか？」

自然な日本語で、1文だけ話しかけてください。必ず対話の流れに沿って発言してください。
返答の形式はセリフのみで、「ロボット：」などはつけないでください。
"""
        else:
            return ""

        res = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "あなたは空気を読みながら適切に発言するロボットです。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )
        return res.choices[0].message.content.replace("[ロボット]", "").replace("「", "").replace("」", "").strip()

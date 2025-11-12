"""
会話シミュレーション環境
人間LLM 3名 + ロボット介入のシミュレーション
"""

import networkx as nx
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random
import config
from human_llm_simulator import HumanLLMSimulator
from intervention_planner import InterventionPlanner
from community_analyzer import CommunityAnalyzer
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params


class SimulationEnvironment:
    """会話シミュレーション環境"""

    def __init__(self):
        """初期化"""
        self._CFG = config.get_config()

        # ペルソナ設定
        self.personas = ["A", "B", "C"]
        personas_cfg = getattr(self._CFG.env, "personas", {})
        self.persona_triggers = {
            name: info.get("triggers", []) if isinstance(info, dict) else []
            for name, info in personas_cfg.items()
        }

        # シミュレーション設定
        sim_cfg = getattr(self._CFG, "simulation", None)
        self.max_human_utterances = getattr(sim_cfg, "max_human_utterances", 45)
        self.stability_check_interval = getattr(sim_cfg, "stability_check_interval", 3)
        self.consecutive_stable_threshold = getattr(
            sim_cfg, "consecutive_stable_threshold", 2
        )

        # 評価設定
        self.evaluation_horizon = getattr(self._CFG.env, "evaluation_horizon", 3)

        # 人間LLMシミュレーター
        self.human_llm = HumanLLMSimulator(self.personas, self.persona_triggers)

        # 話題生成用の既使用地雷リスト
        self.used_triggers = []

    def generate_topic(self) -> Tuple[str, Optional[str]]:
        """
        話題を生成

        Returns:
            (topic, topic_trigger): 話題とトリガーとなった地雷
        """
        # 全地雷からランダムに1つ選択（既に使用した地雷は除外）
        all_triggers = set()
        for triggers in self.persona_triggers.values():
            all_triggers.update(triggers)

        available_triggers = list(all_triggers - set(self.used_triggers))

        if not available_triggers:
            # 全て使い切ったらリセット
            self.used_triggers = []
            available_triggers = list(all_triggers)

        if not available_triggers:
            return "自由な話題", None

        selected_trigger = random.choice(available_triggers)
        self.used_triggers.append(selected_trigger)

        # Azure OpenAI で話題生成
        client, deployment = get_azure_chat_completion_client(
            self._CFG.llm, model_type="topic"
        )
        if not client or not deployment:
            return f"{selected_trigger}について", selected_trigger

        generation_prompt = f"""
あなたは人間3人が話すための、話題を1つだけ提案するアシスタントです。

以下のカテゴリーに関連する話題を生成してください：
{selected_trigger}

このカテゴリーに直接的に関連する具体的な話題を提案してください。
例：
- 「お金」カテゴリー → 「最近の物価高について」「投資の始め方」
- 「ゲーム」カテゴリー → 「最新のゲームトレンド」「eスポーツの将来」
- 「映画」カテゴリー → 「今年のアカデミー賞作品」「好きな映画監督」

話題を1フレーズ（3-10語）で1つだけ提案してください。
"""

        messages = [{"role": "user", "content": generation_prompt}]
        params = build_chat_completion_params(
            deployment, messages, self._CFG.llm, temperature=0.9
        )

        try:
            res = client.chat.completions.create(**params)
            topic = res.choices[0].message.content.strip()
            return topic, selected_trigger
        except Exception as e:
            print(f"⚠️ 話題生成エラー: {e}")
            return f"{selected_trigger}について", selected_trigger

    def evaluate_relationships(self, logs: List[Dict]) -> Tuple[Dict, bool, bool]:
        """
        関係性を評価

        Args:
            logs: 会話ログ

        Returns:
            (metrics, is_stable, has_isolated): 関係性メトリクス、安定判定、疎外ノード有無
        """
        # CommunityAnalyzerを使用して関係性を評価
        analyzer = CommunityAnalyzer()

        # ロボット発話を除外した会話ログを作成
        max_history_relation = getattr(self._CFG.env, "max_history_relation", 3) or 3
        human_logs = [log for log in logs if log.get("speaker") != "ロボット"]
        filtered_logs = human_logs[-max_history_relation:]

        if len(filtered_logs) < 3:
            # 3発話未満は評価不可
            return {}, False, False

        # 関係性スコアを取得
        scores = analyzer.estimate_relation(filtered_logs)

        # グラフ構築
        graph = nx.Graph()
        for (a, b), score in scores.items():
            graph.add_edge(a, b, score=score)

        # 三角形のスコアを計算
        triangle_scores = analyzer.analyze_triangles(graph)

        # メトリクスを計算
        metrics = {"edges": scores, "unstable_triads": 0, "isolated_nodes": []}

        # 不安定三角形をカウント
        for triangle, (struct, avg_score) in triangle_scores.items():
            if struct in ("---", "++-", "+-+", "-++"):
                metrics["unstable_triads"] += 1

        # 疎外ノードを検出（全てのエッジが0.0以下）
        for node in self.personas:
            edges = [
                (
                    scores.get((node, other), 0.0)
                    if node < other
                    else scores.get((other, node), 0.0)
                )
                for other in self.personas
                if other != node
            ]
            if all(edge <= 0.0 for edge in edges):
                metrics["isolated_nodes"].append(node)

        # 安定判定: 不安定三角形数が0 かつ 疎外ノードが存在しない
        is_stable = (
            metrics["unstable_triads"] == 0 and len(metrics["isolated_nodes"]) == 0
        )
        has_isolated = len(metrics["isolated_nodes"]) > 0

        return metrics, is_stable, has_isolated

    def should_intervene(
        self, logs: List[Dict], metrics: Dict
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        介入判定

        Args:
            logs: 会話ログ
            metrics: 関係性メトリクス

        Returns:
            (should_intervene, plan, robot_utterance): 介入判定、介入プラン、ロボット発話
        """
        # 不安定三角形または疎外ノードがある場合に介入
        unstable_triads = metrics.get("unstable_triads", 0)
        isolated_nodes = metrics.get("isolated_nodes", [])

        if unstable_triads == 0 and len(isolated_nodes) == 0:
            # 安定状態 → 介入しない
            return False, None, None

        # CommunityAnalyzer を使用してグラフと三角形スコアを構築
        analyzer = CommunityAnalyzer()
        human_logs = [log for log in logs if log.get("speaker") != "ロボット"]
        max_history_relation = getattr(self._CFG.env, "max_history_relation", 3) or 3
        filtered_logs = human_logs[-max_history_relation:]

        scores = analyzer.estimate_relation(filtered_logs)
        graph = nx.Graph()
        for (a, b), score in scores.items():
            graph.add_edge(a, b, score=score)

        triangle_scores = analyzer.analyze_triangles(graph)

        # InterventionPlannerで介入計画
        planner = InterventionPlanner(
            graph=graph,
            triangle_scores=triangle_scores,
            isolation_threshold=0.0,
            mode="proposal",
        )

        # 最新10発話を取得（ロボット発話を含む）
        session_logs = logs[-10:]
        plan = planner.plan_intervention(session_logs)

        if plan is None:
            return False, None, None

        # ロボット発話を生成
        robot_utterance = planner.generate_robot_utterance(plan, session_logs)

        return True, plan, robot_utterance

    def run_episode(self, episode_id: int) -> Dict:
        """
        1エピソードを実行

        Args:
            episode_id: エピソードID

        Returns:
            エピソード統計
        """
        print(f"\n{'='*80}")
        print(f"📺 エピソード {episode_id}")
        print(f"{'='*80}")

        # 話題生成
        topic, topic_trigger = self.generate_topic()
        print(f"📌 話題: {topic}")
        if topic_trigger:
            print(f"   (トリガー: {topic_trigger})")

        # 初期化
        logs = []
        human_utterance_count = 0
        consecutive_stable_count = 0
        robot_utterances = []
        intervention_count = 0

        # エピソード開始時刻
        start_time = datetime.now()

        while human_utterance_count < self.max_human_utterances:
            # 人間発話を1つ生成
            human_replies = self.human_llm.generate_human_reply(
                logs, topic=topic, topic_trigger=topic_trigger, num_speakers=1
            )

            if not human_replies:
                print("⚠️ 人間発話生成失敗")
                break

            logs.extend(human_replies)
            human_utterance_count += 1

            # 発話を表示
            for reply in human_replies:
                print(f"  [{reply['speaker']}] {reply['utterance']}")

            # stability_check_interval ごとに関係性評価
            if human_utterance_count % self.stability_check_interval == 0:
                print(f"\n📊 関係性評価 ({human_utterance_count}発話時点)")

                metrics, is_stable, has_isolated = self.evaluate_relationships(logs)

                print(f"  不安定三角形数: {metrics.get('unstable_triads', 0)}")
                print(f"  疎外ノード: {metrics.get('isolated_nodes', [])}")
                print(f"  安定状態: {'✅ はい' if is_stable else '❌ いいえ'}")

                # エッジスコアを表示
                edges = metrics.get("edges", {})
                if edges:
                    print(f"  関係性スコア:")
                    for (a, b), score in sorted(edges.items()):
                        print(f"    {a}-{b}: {score:+.2f}")

                if is_stable:
                    consecutive_stable_count += 1
                    print(
                        f"  連続安定回数: {consecutive_stable_count}/{self.consecutive_stable_threshold}"
                    )

                    if consecutive_stable_count >= self.consecutive_stable_threshold:
                        print(
                            f"\n🎉 {self.consecutive_stable_threshold}回連続で安定 → エピソード終了"
                        )
                        break
                else:
                    consecutive_stable_count = 0

                    # 介入判定
                    should_intervene, plan, robot_utterance = self.should_intervene(
                        logs, metrics
                    )

                    if should_intervene and robot_utterance:
                        intervention_count += 1
                        print(f"\n🤖 ロボット介入 ({intervention_count}回目)")
                        print(f"  介入タイプ: {plan.get('type', '不明')}")
                        print(f"  発話: {robot_utterance}")

                        robot_entry = {
                            "speaker": "ロボット",
                            "utterance": robot_utterance,
                            "plan": plan,
                        }
                        logs.append(robot_entry)
                        robot_utterances.append(robot_entry)

                        # evaluation_horizon 回人間発話を生成
                        print(f"  (介入後 {self.evaluation_horizon}発話生成中...)")
                        for _ in range(self.evaluation_horizon):
                            eval_replies = self.human_llm.generate_human_reply(
                                logs,
                                topic=topic,
                                topic_trigger=topic_trigger,
                                num_speakers=1,
                            )
                            if eval_replies:
                                logs.extend(eval_replies)
                                human_utterance_count += 1
                                for reply in eval_replies:
                                    print(
                                        f"  [{reply['speaker']}] {reply['utterance']}"
                                    )

                            if human_utterance_count >= self.max_human_utterances:
                                break

        # 終了時刻
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 最終評価
        final_metrics, final_stable, final_has_isolated = self.evaluate_relationships(
            logs
        )

        print(f"\n📈 エピソード終了")
        print(f"  総人間発話数: {human_utterance_count}")
        print(f"  ロボット介入回数: {intervention_count}")
        print(f"  最終状態: {'✅ 安定' if final_stable else '❌ 不安定'}")
        print(f"  最終不安定三角形数: {final_metrics.get('unstable_triads', 0)}")
        print(f"  最終疎外ノード: {final_metrics.get('isolated_nodes', [])}")
        print(f"  所要時間: {duration:.1f}秒")

        # 統計を返す
        stats = {
            "episode_id": episode_id,
            "topic": topic,
            "topic_trigger": topic_trigger,
            "human_utterance_count": human_utterance_count,
            "robot_utterance_count": intervention_count,
            "final_stable": final_stable,
            "final_unstable_triads": final_metrics.get("unstable_triads", 0),
            "final_isolated_nodes": final_metrics.get("isolated_nodes", []),
            "duration_seconds": duration,
            "logs": logs,
            "robot_utterances": robot_utterances,
        }

        return stats

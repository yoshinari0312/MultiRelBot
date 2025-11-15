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
from log_filtering import filter_logs_by_human_count


class SimulationEnvironment:
    """会話シミュレーション環境"""

    def __init__(self):
        """初期化"""
        self._CFG = config.get_config()

        # ペルソナ設定（config.yamlから読み込み）
        personas_cfg = getattr(self._CFG.env, "personas", {})
        self.personas = list(personas_cfg.keys())
        self.persona_triggers = {
            name: info.get("triggers", []) if isinstance(info, dict) else []
            for name, info in personas_cfg.items()
        }

        # シミュレーション設定
        sim_cfg = getattr(self._CFG, "simulation", None)
        self.max_human_utterances = getattr(sim_cfg, "max_human_utterances", 45)
        # 関係性推定周期: 参加者数発話ごと（ロボット発話は除外）
        num_participants = getattr(self._CFG.participants, "num_participants", 3)
        self.stability_check_interval = num_participants
        self.consecutive_stable_threshold = getattr(
            sim_cfg, "consecutive_stable_threshold", 2
        )

        # 人間LLMシミュレーター
        self.human_llm = HumanLLMSimulator(self.personas, self.persona_triggers)

        # CommunityAnalyzerのインスタンスを保持（EMA履歴を維持するため）
        self.analyzer = CommunityAnalyzer()

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

        # config.yamlから話題生成プロンプトを読み込み
        topic_cfg = getattr(self._CFG, "topic_manager", None)
        if topic_cfg and hasattr(topic_cfg, "generation_prompt"):
            prompt_template = topic_cfg.generation_prompt
            # {trigger_examples}を実際のトリガーで置き換え
            generation_prompt = prompt_template.replace(
                "{trigger_examples}", selected_trigger
            )
        else:
            # フォールバック
            generation_prompt = f"""
あなたは人間{len(self.personas)}人が話すための、話題を1つだけ提案するアシスタントです。

以下のカテゴリーに関連する話題を生成してください：
{selected_trigger}

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
        # CommunityAnalyzerを使用して関係性を評価（インスタンスを再利用）
        analyzer = self.analyzer

        # 直近 max_history_relation 個の人間発話 + その間のロボット発話を抽出
        # ロボット発話に対する人間の反応を正しく理解するため、ロボット発話も含める
        max_history_relation = getattr(self._CFG.env, "max_history_relation", 3) or 3
        filtered_logs = filter_logs_by_human_count(
            logs, max_history_relation, exclude_robot=False
        )

        # 人間発話のみをカウントして評価可否を判定
        human_only_logs = [log for log in filtered_logs if log.get("speaker") != "ロボット"]
        if len(human_only_logs) < 3:
            # 人間発話が3未満は評価不可
            return {}, False, False

        # 関係性スコアを取得
        scores = analyzer.estimate_relation(filtered_logs)
        # GPTの生スコアを表示(1桁丸め) とEMAを表示（有効時）- ソート順で表示
        try:
            gpt_scores = analyzer.last_gpt_scores
            ema_scores = analyzer.last_ema_scores
            if gpt_scores:
                # テーブル表示: A-B: GPT +0.7 -> EMA +0.6（ソート順）
                for k in sorted(gpt_scores.keys()):
                    a, b = k
                    g = gpt_scores.get(k)
                    e = ema_scores.get(k) if ema_scores else None
                    if getattr(analyzer, "use_ema", False) and e is not None:
                        print(f"  {a}-{b}: GPT {g:+.1f} -> EMA {e:+.1f}")
                    else:
                        print(f"  {a}-{b}: GPT {g:+.1f}")
        except Exception:
            pass

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

        # 疎外ノードを検出（全てのエッジが負の値）
        for node in self.personas:
            edges = []
            for other in self.personas:
                if other != node:
                    # スコア辞書のキーは常にソート済み (a, b) where a < b
                    key = tuple(sorted([node, other]))
                    edges.append(scores.get(key, 0.0))
            # 全てのエッジが負の値の場合のみ疎外ノードと判定（0.0は含まない）
            if edges and all(edge < 0.0 for edge in edges):
                metrics["isolated_nodes"].append(node)

        # 安定判定: 不安定三角形数が0 かつ 疎外ノードが存在しない
        is_stable = (
            metrics["unstable_triads"] == 0 and len(metrics["isolated_nodes"]) == 0
        )
        has_isolated = len(metrics["isolated_nodes"]) > 0

        return metrics, is_stable, has_isolated

    def should_intervene(
        self, logs: List[Dict], metrics: Dict, scores: Dict = None, graph: nx.Graph = None
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        介入判定

        Args:
            logs: 会話ログ
            metrics: 関係性メトリクス
            scores: 関係性スコア（evaluate_relationshipsから渡される）
            graph: 関係性グラフ（evaluate_relationshipsから渡される）

        Returns:
            (should_intervene, plan, robot_utterance): 介入判定、介入プラン、ロボット発話
        """
        # 不安定三角形または疎外ノードがある場合に介入
        unstable_triads = metrics.get("unstable_triads", 0)
        isolated_nodes = metrics.get("isolated_nodes", [])

        if unstable_triads == 0 and len(isolated_nodes) == 0:
            # 安定状態 → 介入しない
            return False, None, None

        # scoresとgraphが渡されていない場合は構築（後方互換性のため）
        if scores is None or graph is None:
            max_history_relation = getattr(self._CFG.env, "max_history_relation", 3) or 3
            filtered_logs = filter_logs_by_human_count(
                logs, max_history_relation, exclude_robot=True
            )
            scores = self.analyzer.estimate_relation(filtered_logs)
            graph = nx.Graph()
            for (a, b), score in scores.items():
                graph.add_edge(a, b, score=score)

        triangle_scores = self.analyzer.analyze_triangles(graph)

        # config.yamlから介入設定を読み込み
        intervention_cfg = getattr(self._CFG, "intervention", None)
        isolation_threshold = (
            getattr(intervention_cfg, "isolation_threshold", 0.0)
            if intervention_cfg
            else 0.0
        )
        intervention_mode = (
            getattr(intervention_cfg, "mode", "proposal")
            if intervention_cfg
            else "proposal"
        )

        # InterventionPlannerで介入計画
        planner = InterventionPlanner(
            graph=graph,
            triangle_scores=triangle_scores,
            isolation_threshold=isolation_threshold,
            mode=intervention_mode,
        )

        # config.yamlから介入判定用の会話履歴数を読み込み
        # 直近 intervention_max_history 個の人間発話 + その間のロボット発話を取得
        intervention_max_history = (
            getattr(self._CFG.env, "intervention_max_history", 3) or 10
        )
        session_logs = filter_logs_by_human_count(
            logs, intervention_max_history, exclude_robot=False
        )
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
            stats: エピソード統計情報
        """
        print(f"\n{'='*80}")
        print(f"� エピソード {episode_id}")
        print(f"{'='*80}")

        # EMA履歴をリセット（新しいエピソード）
        self.analyzer.reset_ema()

        # 開始時刻
        start_time = datetime.now()

        # 話題を生成
        topic, topic_trigger = self.generate_topic()
        print(f"� 話題: {topic}")
        if topic_trigger:
            print(f"  (トリガー: {topic_trigger})")

        # 会話ログ
        logs = []
        robot_utterances = []

        # カウンター
        human_utterance_count = 0
        intervention_count = 0

        # 統計用変数
        stability_checks = []
        isolation_checks = []
        edge_score_history = []
        positive_ratio_history = []
        intervention_improvements = []
        consecutive_stable_count = 0
        consecutive_unstable_count = 0
        consecutive_unstable_max = 0
        first_stable_utterance = None
        last_stable_state = None
        oscillation_count = 0
        early_termination = False

        print("\n")
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
                print(f"[{reply['speaker']}] {reply['utterance']}")

            # stability_check_interval ごとに関係性評価
            if human_utterance_count % self.stability_check_interval == 0:
                print(f"📊 関係性評価 ({human_utterance_count}発話時点)")
                
                # 関係性推定用の会話履歴を取得
                max_history_relation = getattr(self._CFG.env, "max_history_relation", 3) or 3
                relation_logs = filter_logs_by_human_count(logs, max_history_relation, exclude_robot=True)
                if relation_logs:
                    print(f"    → {len(relation_logs)}件の発話を使用して関係性を推定")

                metrics, is_stable, has_isolated = self.evaluate_relationships(logs)
                
                # 直前のロボット介入の効果を測定
                if robot_utterances:
                    last_robot = robot_utterances[-1]
                    if "pre_score" in last_robot and last_robot["pre_score"] is not None:
                        target_edge = last_robot.get("target_edge")
                        pre_score = last_robot["pre_score"]
                        edges = metrics.get("edges", {})
                        post_score = edges.get(target_edge) or edges.get((target_edge[1], target_edge[0]))
                        if post_score is not None:
                            improvement = post_score - pre_score
                            intervention_improvements.append(improvement)
                            print(f"  📈 介入効果: {target_edge[0]}-{target_edge[1]} = {pre_score:+.1f} → {post_score:+.1f} (変化: {improvement:+.1f})")
                            # 測定済みなのでフラグを削除
                            del last_robot["pre_score"]
                            del last_robot["target_edge"]

                print(f"  不安定三角形数: {metrics.get('unstable_triads', 0)}")
                print(f"  疎外ノード: {metrics.get('isolated_nodes', [])}")
                print(f"  安定状態: {'✅ はい' if is_stable else '❌ いいえ'}")

                # エッジスコアを表示と記録
                edges = metrics.get("edges", {})
                if edges:
                    print(f"  関係性スコア:")
                    for (a, b), score in sorted(edges.items()):
                        print(f"    {a}-{b}: {score:+.1f}")

                    # 全エッジ平均スコアを計算
                    avg_edge_score = sum(edges.values()) / len(edges)
                    edge_score_history.append(avg_edge_score)

                    # 正エッジ割合を計算
                    positive_edges = sum(1 for score in edges.values() if score > 0)
                    positive_ratio = positive_edges / len(edges)
                    positive_ratio_history.append(positive_ratio)

                # 安定性と疎外ノード有無を記録
                stability_checks.append(is_stable)
                isolation_checks.append(has_isolated)

                # 初回安定時の発話数を記録
                if is_stable and first_stable_utterance is None:
                    first_stable_utterance = human_utterance_count
                    print(f"  🎯 初回安定達成: {human_utterance_count}発話")

                # 安定⇔不安定の切り替わりを検出
                if last_stable_state is not None and last_stable_state != is_stable:
                    oscillation_count += 1
                last_stable_state = is_stable

                if is_stable:
                    consecutive_stable_count += 1
                    consecutive_unstable_count = 0
                    print(
                        f"  連続安定回数: {consecutive_stable_count}/{self.consecutive_stable_threshold}\n"
                    )

                    if consecutive_stable_count >= self.consecutive_stable_threshold:
                        print(
                            f"\n🎉 {self.consecutive_stable_threshold}回連続で安定 → エピソード終了"
                        )
                        early_termination = True
                        break
                else:
                    consecutive_stable_count = 0
                    consecutive_unstable_count += 1
                    # 連続不安定回数の最大値を更新
                    consecutive_unstable_max = max(
                        consecutive_unstable_max, consecutive_unstable_count
                    )

                    # 介入判定（graphとscoresを渡して重複計算を防止）
                    graph = nx.Graph()
                    for (a, b), score in edges.items():
                        graph.add_edge(a, b, score=score)
                    should_intervene, plan, robot_utterance = self.should_intervene(
                        logs, metrics, scores=edges, graph=graph
                    )

                    if should_intervene and robot_utterance:
                        intervention_count += 1
                        print(f"\n🤖 ロボット介入 ({intervention_count}回目)")
                        print(f"  介入タイプ: {plan.get('type', '不明')}")
                        print(f"  発話: {robot_utterance}")

                        # 介入対象エッジの介入前スコアを記録
                        target_edge = plan.get("edge")
                        pre_intervention_score = None
                        if target_edge and edges:
                            # エッジは (A, B) または (B, A) の可能性があるので両方チェック
                            pre_intervention_score = edges.get(
                                target_edge
                            ) or edges.get((target_edge[1], target_edge[0]))
                            if pre_intervention_score is not None:
                                print(f"  介入対象エッジ: {target_edge[0]}-{target_edge[1]} (介入前: {pre_intervention_score:+.1f})")

                        robot_entry = {
                            "speaker": "ロボット",
                            "utterance": robot_utterance,
                            "plan": plan,
                            "pre_score": pre_intervention_score,  # 次の関係性推定時に使用
                            "target_edge": target_edge,
                        }
                        logs.append(robot_entry)
                        robot_utterances.append(robot_entry)

        # 終了時刻
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 最終評価（早期終了でない場合のみ）
        if not early_termination:
            final_metrics, final_stable, final_has_isolated = self.evaluate_relationships(
                logs
            )
        else:
            # 早期終了の場合は最後の評価結果を使用
            final_metrics = metrics
            final_stable = is_stable
            final_has_isolated = has_isolated

        print(f"\n📈 エピソード終了")
        print(f"  総人間発話数: {human_utterance_count}")
        print(f"  ロボット介入回数: {intervention_count}")
        print(
            f"  早期終了: {'✅ はい (2連続安定)' if early_termination else '❌ いいえ'}"
        )
        print(f"  最終状態: {'✅ 安定' if final_stable else '❌ 不安定'}")
        print(f"  最終不安定三角形数: {final_metrics.get('unstable_triads', 0)}")
        print(f"  最終疎外ノード: {final_metrics.get('isolated_nodes', [])}")
        print(f"  所要時間: {duration:.1f}秒")

        # 新規指標の計算
        num_checks = len(stability_checks)
        stable_count_in_checks = sum(stability_checks)  # 安定評価回数
        stability_rate = stable_count_in_checks / num_checks if num_checks > 0 else 0.0
        isolation_occurrence_rate = (
            sum(isolation_checks) / num_checks if num_checks > 0 else 0.0
        )
        avg_edge_score = (
            sum(edge_score_history) / len(edge_score_history)
            if edge_score_history
            else 0.0
        )
        avg_positive_ratio = (
            sum(positive_ratio_history) / len(positive_ratio_history)
            if positive_ratio_history
            else 0.0
        )
        intervention_success_rate = 0.0
        avg_improvement_per_intervention = 0.0
        if intervention_improvements:
            successful_interventions = sum(
                1 for imp in intervention_improvements if imp > 0
            )
            intervention_success_rate = successful_interventions / len(
                intervention_improvements
            )
            avg_improvement_per_intervention = sum(intervention_improvements) / len(
                intervention_improvements
            )
        intervention_frequency = (
            intervention_count / human_utterance_count
            if human_utterance_count > 0
            else 0.0
        )

        # 新規指標: 1介入あたりの安定率と1安定あたりのロボット介入回数
        stable_rate_per_intervention = (
            stable_count_in_checks / intervention_count
            if intervention_count > 0
            else 0.0
        )
        interventions_per_stable = (
            intervention_count / stable_count_in_checks
            if stable_count_in_checks > 0
            else 0.0
        )

        print(f"  安定率: {stability_rate:.2%}")
        print(f"  疎外発生率: {isolation_occurrence_rate:.2%}")
        if first_stable_utterance:
            print(f"  初回安定達成: {first_stable_utterance}発話")
        print(f"  切り替わり回数: {oscillation_count}回")
        print(f"  連続不安定最大: {consecutive_unstable_max}回")

        # 統計を返す
        stats = {
            "episode_id": episode_id,
            "topic": topic,
            "topic_trigger": topic_trigger,
            "human_utterance_count": human_utterance_count,
            "robot_utterance_count": intervention_count,
            "early_termination": early_termination,
            "final_stable": final_stable,
            "final_unstable_triads": final_metrics.get("unstable_triads", 0),
            "final_isolated_nodes": final_metrics.get("isolated_nodes", []),
            "duration_seconds": duration,
            "logs": logs,
            "robot_utterances": robot_utterances,
            # 新規指標
            "stability_rate": stability_rate,
            "isolation_occurrence_rate": isolation_occurrence_rate,
            "first_stable_utterance": first_stable_utterance,
            "oscillation_count": oscillation_count,
            "consecutive_unstable_max": consecutive_unstable_max,
            "avg_edge_score": avg_edge_score,
            "avg_positive_ratio": avg_positive_ratio,
            "intervention_success_rate": intervention_success_rate,
            "avg_improvement_per_intervention": avg_improvement_per_intervention,
            "intervention_frequency": intervention_frequency,
            "stable_rate_per_intervention": stable_rate_per_intervention,
            "interventions_per_stable": interventions_per_stable,
        }

        return stats

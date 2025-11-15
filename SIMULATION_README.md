# 会話シミュレーションシステム

人間 LLM（Azure OpenAI）を使用した 3 者会話シミュレーションとロボット介入効果の評価システム

## 概要

このシミュレーションシステムは、以下の機能を提供します：

- **人間 LLM**: Azure OpenAI の HUMAN_MODEL を使用して 3 名（A, B, C）の会話を生成
- **話題自動生成**: 地雷キーワードに基づいた話題の自動生成
- **関係性評価**: 参加者数ごとに関係性を推定し、不安定三角形と疎外ノードを検出
- **ロボット介入**: 不安定状態を検出した場合の自動介入
- **安定判定**: 2 回連続で安定したらエピソード早期終了
- **詳細な評価指標**: 安定率、介入効果、関係性スコアなど 15+ の評価指標

## ファイル構成

```
human_llm_simulator.py      # 人間LLM会話生成
simulation_environment.py   # シミュレーション環境
intervention_planner.py     # ロボット介入戦略
community_analyzer.py       # 関係性分析
log_filtering.py            # 会話履歴フィルタリングユーティリティ
run_simulation.py           # 実行スクリプト
config.local.yaml           # 設定ファイル
```

## 設定

`config.local.yaml` の主要セクション:

### シミュレーション設定

```yaml
simulation:
  max_human_utterances: 45 # 最大人間発話数（1エピソードあたり）
  consecutive_stable_threshold: 2 # 何回連続で安定したらエピソード終了とするか
  num_episodes: 10 # 実行するエピソード数
```

### 履歴管理設定

```yaml
env:
  intervention_max_history: 9 # 介入判定・ロボット発話LLMへの会話履歴（人間発話数。その間のロボット発話も含む）
  max_history_human: 12 # 人間LLMへの会話履歴（人間発話数。その間のロボット発話も含む）
  max_history_relation: 3 # 関係性LLMへの会話履歴（人間発話数のみ。ロボット発話は除外）
  evaluation_horizon: 3 # ロボット介入後に生成する人間発話数
```

### 参加者設定

```yaml
participants:
  num_participants: 3 # 参加者数（3 or 4）
```

**重要**: 関係性評価の周期は `num_participants` に自動的に設定されます（3人なら3発話ごと、4人なら4発話ごと）

### 介入設定

```yaml
intervention:
  mode: "proposal" # 介入モード ("proposal", "few_utterances", "random_target")
  isolation_threshold: 0.0 # 孤立判定のスコア閾値
  temperature: 1.0 # 介入発話生成時のLLM温度パラメータ
```

### 安定判定条件

以下の条件を **両方** 満たすと安定と判定:

1. `unstable_triads == 0` (不安定三角形数が 0)
2. `isolated_nodes == 0` (疎外ノードが 0)

### 疎外ノードの定義

話者が **全ての他者との関係性スコアが 0.0 以下** の場合、疎外ノードと判定

## 使用方法

### 基本的な実行

```bash
# デフォルト設定で実行（config.yamlのnum_episodes設定を使用）
python run_simulation.py

# エピソード数を指定
python run_simulation.py --num-episodes 10

# 出力ディレクトリを指定
python run_simulation.py --num-episodes 20 --output results/sim_20250115
```

### 実行例

```bash
# 10エピソード実行
python run_simulation.py --num-episodes 10
```

実行中の出力例:

```
================================================================================
🚀 会話シミュレーション開始
================================================================================
エピソード数: 10
出力ディレクトリ: results/simulation_20250115_153045

================================================================================
📺 エピソード 1
================================================================================
📌 話題: 最近の物価高について
   (トリガー: お金)
  [A] 最近本当に物価が上がりましたね。
  [B] そうですね、買い物に行くたびに驚きます。
  [C] 特に食品が高くなった気がします。

📊 関係性評価 (3発話時点)
  不安定三角形数: 0
  疎外ノード: []
  安定状態: ✅ はい
  連続安定回数: 1/2
  関係性スコア:
    A-B: +0.85
    A-C: +0.72
    B-C: +0.68
  ...
```

## 出力ファイル

シミュレーション終了後、以下のファイルが生成されます：

```
results/simulation_YYYYMMDD_HHMMSS/
├── summary.json                      # 全体統計（全エピソードの平均）
├── episode_1.json                    # エピソード1の統計
├── episode_1_conversation.txt        # エピソード1の会話ログ
├── episode_2.json
├── episode_2_conversation.txt
...
```

### summary.json の内容（全エピソードの平均）

```json
{
  "total_episodes": 10,
  "stable_completion_rate": 0.7,
  "stable_completion_episodes": 7,
  "stability_achieved_rate": 0.8,
  "stability_achieved_episodes": 8,
  "avg_stability_rate": 0.65,
  "avg_isolation_occurrence_rate": 0.15,
  "avg_human_utterances": 28.5,
  "avg_robot_utterances": 3.2,
  "avg_human_utterances_to_stable": 22.3,
  "avg_robot_utterances_to_stable": 2.1,
  "avg_first_stable_utterance": 18.5,
  "avg_final_unstable_triads": 0.4,
  "avg_oscillation_count": 2.3,
  "avg_consecutive_unstable_max": 3.1,
  "avg_edge_score": 0.42,
  "avg_positive_ratio": 0.68,
  "avg_intervention_success_rate": 0.72,
  "avg_improvement_per_intervention": 0.15,
  "avg_intervention_frequency": 0.11,
  "avg_stable_rate_per_intervention": 2.1,
  "avg_interventions_per_stable": 0.48,
  "avg_duration_seconds": 245.6
}
```

### episode_N.json の内容（エピソードごとの詳細）

```json
{
  "episode_id": 1,
  "topic": "最近の物価高について",
  "topic_trigger": "お金",
  "human_utterance_count": 24,
  "robot_utterance_count": 3,
  "early_termination": true,
  "final_stable": true,
  "final_unstable_triads": 0,
  "final_isolated_nodes": [],
  "duration_seconds": 235.4,
  "stability_rate": 0.75,
  "isolation_occurrence_rate": 0.125,
  "first_stable_utterance": 18,
  "oscillation_count": 2,
  "consecutive_unstable_max": 3,
  "avg_edge_score": 0.55,
  "avg_positive_ratio": 0.72,
  "intervention_success_rate": 0.67,
  "avg_improvement_per_intervention": 0.22,
  "intervention_frequency": 0.125,
  "stable_rate_per_intervention": 2.5,
  "interventions_per_stable": 0.4
}
```

## 評価指標

シミュレーションでは以下の 17 の指標を評価します：

### 1. エピソード達成率

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **安定終了エピソード数** | 2連続安定で早期終了したエピソード数 | 多いほど良い |
| **安定終了率** (`stable_completion_rate`) | 安定終了エピソード / 全エピソード | 70%以上 |
| **一度でも安定達成エピソード数** | 一度でも安定状態を達成したエピソード数 | - |
| **一度でも安定達成率** (`stability_achieved_rate`) | 一度でも安定達成 / 全エピソード | - |

### 2. 安定性指標

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **平均安定率** (`avg_stability_rate`) | 全評価回数に対する安定評価の割合（累積） | 高いほど良い |
| **平均疎外発生率** (`avg_isolation_occurrence_rate`) | 全評価回数に対する疎外発生の割合（累積） | 低いほど良い |

### 3. 発話数指標

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **平均人間発話数** (`avg_human_utterances`) | 全エピソードの平均 | - |
| **平均ロボット介入回数** (`avg_robot_utterances`) | 全エピソードの平均 | - |
| **平均人間発話数（安定終了エピソードのみ）** | 安定終了までの平均発話数 | 少ないほど効率的 |
| **平均ロボット介入回数（安定終了エピソードのみ）** | 安定終了までの平均介入回数 | 少ないほど効率的 |
| **平均初回安定達成発話数** (`avg_first_stable_utterance`) | 初めて安定状態を達成した発話数の平均 | 少ないほど早い |

### 4. 構造指標

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **平均最終不安定三角形数** (`avg_final_unstable_triads`) | エピソード終了時の不安定三角形数 | 0に近いほど良い |
| **平均切り替わり回数** (`avg_oscillation_count`) | 安定⇔不安定の切り替わり回数 | 少ないほど安定 |
| **平均最大連続不安定** (`avg_consecutive_unstable_max`) | 連続不安定評価の最大回数 | 少ないほど良い |

### 5. 関係性スコア指標

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **平均エッジスコア** (`avg_edge_score`) | 全評価時点での全エッジの平均スコア（累積平均） | 正の値が望ましい |
| **平均正エッジ割合** (`avg_positive_ratio`) | 全評価時点での正エッジの割合（累積平均） | 高いほど良い |

### 6. 介入効果指標

| 指標 | 説明 | 目標値 |
|------|------|--------|
| **介入成功率** (`avg_intervention_success_rate`) | 介入後に対象エッジが改善した割合 | 高いほど良い |
| **介入あたり平均改善度** (`avg_improvement_per_intervention`) | 介入による対象エッジのスコア変化の平均 | 正の値が望ましい |
| **介入頻度** (`avg_intervention_frequency`) | 人間発話あたりの介入回数 | - |
| **1介入あたりの安定評価回数** (`avg_stable_rate_per_intervention`) | ロボット介入1回あたりの安定評価回数 | 高いほど効率的 |
| **1安定あたりのロボット介入回数** (`avg_interventions_per_stable`) | 安定評価1回あたりの介入回数 | 低いほど効率的 |

### 7. その他

| 指標 | 説明 |
|------|------|
| **平均所要時間** (`avg_duration_seconds`) | 1エピソードあたりの実行時間 |

## 評価結果の見方

### コンソール出力

シミュレーション実行後、以下の2つのセクションが表示されます：

1. **エピソードごとの詳細**: 各エピソードの個別統計
2. **全エピソードの平均サマリー**: 全エピソードを集計した統計

```
================================================================================
📋 エピソードごとの詳細
================================================================================

--- エピソード 1 ---
話題: 最近の物価高について
トリガー: お金
人間発話数: 24
ロボット介入回数: 3
早期終了: ✅ はい
安定率: 75.0%
初回安定達成: 18発話
介入成功率: 66.7%
1介入あたりの安定評価回数: 2.50
1安定あたりのロボット介入回数: 0.40
所要時間: 235.4秒

--- エピソード 2 ---
...

================================================================================
📊 シミュレーション結果サマリー（全エピソードの平均）
================================================================================
総エピソード数: 10

【エピソード達成率】
安定終了エピソード数: 7 (70.0%)
一度でも安定達成: 8 (80.0%)

【安定性指標】
平均安定率: 65.5%
平均疎外発生率: 14.2%

【発話数指標】
平均人間発話数: 28.5
平均ロボット介入回数: 3.2

【安定終了エピソードのみ】
  平均人間発話数（終了まで）: 22.3
  平均ロボット介入回数（終了まで）: 2.1
  平均初回安定達成: 18.5発話

【構造指標】
平均最終不安定三角形数: 0.40
平均切り替わり回数: 2.3
平均最大連続不安定: 3.1

【関係性スコア】
平均エッジスコア: +0.42
平均正エッジ割合: 68.0%

【介入効果】
介入成功率: 72.0%
介入あたり平均改善度: +0.150
介入頻度: 0.11
1介入あたりの安定評価回数: 2.10
1安定あたりのロボット介入回数: 0.48

【その他】
平均所要時間: 245.6秒
```

### 理想的なシミュレーション結果

以下の値を目標とします：

- **安定終了率**: 70%以上
- **平均安定率**: 60%以上
- **平均疎外発生率**: 20%以下
- **介入成功率**: 70%以上
- **平均正エッジ割合**: 60%以上
- **1介入あたりの安定評価回数**: 2.0以上（少ない介入で多くの安定を達成）

## トラブルシューティング

### エラー: Azure OpenAI client could not be initialized

環境変数が正しく設定されているか確認:

```bash
# .env ファイルを確認
AZURE_ENDPOINT=https://...
AZURE_API_KEY=...
HUMAN_MODEL=gpt-4.1
RELATION_MODEL=gpt-4.1
ROBOT_MODEL=gpt-5-chat
```

### エラー: 話題生成エラー

`config.local.yaml` の `env.personas` に地雷（triggers）が設定されているか確認

### 会話が生成されない

- `max_history_human` の値を確認（デフォルト: 12）
- Azure OpenAI のクォータを確認

## カスタマイズ

### 安定判定条件の変更

`simulation_environment.py` の `evaluate_relationships()` メソッドを修正:

```python
# 例: 不安定三角形が1個以下でも安定とする
is_stable = (metrics["unstable_triads"] <= 1 and len(metrics["isolated_nodes"]) == 0)
```

### 介入条件の変更

`simulation_environment.py` の `should_intervene()` メソッドを修正

### 評価指標の追加

1. `simulation_environment.py` の `run_episode()` で新しい指標を計算
2. `run_simulation.py` の `calculate_statistics()` で平均を計算
3. `run_simulation.py` の `print_summary()` で表示を追加

## 履歴フィルタリング

各LLMへの入力履歴は以下のように管理されています：

| LLM | 履歴の範囲 | ロボット発話 |
|-----|-----------|------------|
| **関係性LLM** | 直近 `max_history_relation`個の人間発話 | ❌ 除外 |
| **人間LLM** | 直近 `max_history_human`個の人間発話 + その間のロボット発話 | ✅ 含む |
| **ロボット発話LLM** | 直近 `intervention_max_history`個の人間発話 + その間のロボット発話 | ✅ 含む |

参考実装: https://github.com/yoshinari0312/policyRL_relation_robot/blob/main/app/utils/log_filtering.py

## 参考

- 参考実装: https://github.com/yoshinari0312/policyRL_relation_robot
- Azure OpenAI ドキュメント: https://learn.microsoft.com/azure/ai-services/openai/

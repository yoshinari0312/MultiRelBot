# 会話シミュレーションシステム

人間 LLM（Azure OpenAI）を使用した 3 者会話シミュレーションとロボット介入効果の評価システム

## 概要

このシミュレーションシステムは、以下の機能を提供します：

- **人間 LLM**: Azure OpenAI の HUMAN_MODEL を使用して 3 名（A, B, C）の会話を生成
- **話題自動生成**: 地雷キーワードに基づいた話題の自動生成
- **関係性評価**: 3 発話ごとに関係性を推定し、不安定三角形と疎外ノードを検出
- **ロボット介入**: 不安定状態を検出した場合の自動介入
- **安定判定**: 2 回連続で安定したらエピソード終了

## ファイル構成

```
human_llm_simulator.py      # 人間LLM会話生成
simulation_environment.py   # シミュレーション環境
run_simulation.py           # 実行スクリプト
config.local.yaml           # 設定ファイル（simulation セクション追加）
```

## 設定

`config.local.yaml` の `simulation` セクション:

```yaml
simulation:
  max_human_utterances: 45 # 最大人間発話数（1エピソードあたり）
  stability_check_interval: 3 # 何人間発話ごとに関係性を評価するか
  consecutive_stable_threshold: 2 # 何回連続で安定したらエピソード終了とするか
  num_episodes: 10 # 実行するエピソード数
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
python run_simulation.py --num-episodes 20 --output results/sim_20250113
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
出力ディレクトリ: results/simulation_20250113_153045

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
├── summary.json                      # 全体統計
├── episode_1.json                    # エピソード1の統計
├── episode_1_conversation.txt        # エピソード1の会話ログ
├── episode_2.json
├── episode_2_conversation.txt
...
```

### summary.json の内容

```json
{
  "total_episodes": 10,
  "stable_episodes": 7,
  "stable_rate": 0.7,
  "isolated_episodes": 2,
  "isolated_rate": 0.2,
  "avg_human_utterances": 28.5,
  "avg_robot_utterances": 3.2,
  "avg_human_utterances_to_stable": 22.3,
  "avg_robot_utterances_to_stable": 2.1,
  "avg_unstable_triads": 0.4,
  "avg_duration_seconds": 245.6
}
```

## 評価指標

シミュレーションでは以下の指標を評価します：

### 主要指標

1. **安定率 (stable_rate)**

   - 安定状態で終了したエピソードの割合
   - 目標: 70%以上

2. **安定までのロボット発話数 (avg_robot_utterances_to_stable)**

   - 安定達成までの平均ロボット介入回数
   - 少ないほど効率的

3. **安定までの人間発話数 (avg_human_utterances_to_stable)**

   - 安定達成までの平均人間発話数
   - 少ないほど早く安定

4. **疎外ノード率 (isolated_rate)**
   - 疎外ノードが発生したエピソードの割合
   - 低いほど良好

### 補助指標

5. **平均不安定三角形数 (avg_unstable_triads)**

   - エピソード終了時の平均不安定三角形数
   - 0 に近いほど良好

6. **平均所要時間 (avg_duration_seconds)**
   - 1 エピソードあたりの平均実行時間

## 評価結果の見方

### 理想的なシミュレーション結果

```
📊 シミュレーション結果サマリー
総エピソード数: 10
安定達成エピソード数: 8 (80.0%)        ← 70%以上が目標
疎外ノード発生エピソード数: 1 (10.0%)  ← 低いほど良い

平均人間発話数: 25.3
平均ロボット介入回数: 2.8

【安定達成エピソードのみ】
  平均人間発話数（安定まで）: 20.5    ← 少ないほど効率的
  平均ロボット介入回数（安定まで）: 2.0 ← 少ないほど効率的

平均不安定三角形数: 0.20              ← 0に近いほど良い
平均所要時間: 235.4秒
```

## トラブルシューティング

### エラー: Azure OpenAI client could not be initialized

環境変数が正しく設定されているか確認:

```bash
# .env ファイルを確認
AZURE_ENDPOINT=https://...
AZURE_API_KEY=...
HUMAN_MODEL=gpt-4.1
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

`run_simulation.py` の `calculate_statistics()` メソッドに統計を追加

## 今後の拡張案

### 追加評価指標の候補

1. **介入成功率**

   - 介入後に関係性が改善した割合
   - 実装: 介入前後の不安定三角形数を比較

2. **平均関係性スコア**

   - 全エッジの平均スコア
   - 実装: `metrics["edges"]` の平均値を計算

3. **関係性改善速度**

   - 不安定 → 安定への遷移速度
   - 実装: 各評価ポイントでの不安定三角形数の変化を追跡

4. **話題別成功率**
   - 各地雷トリガーごとの安定達成率
   - 実装: `topic_trigger` ごとに統計を集計

### 実装例: 介入成功率

```python
def calculate_intervention_success_rate(all_stats: List[Dict]) -> float:
    """介入成功率を計算"""
    successful_interventions = 0
    total_interventions = 0

    for stats in all_stats:
        robot_utterances = stats.get("robot_utterances", [])
        total_interventions += len(robot_utterances)

        # ここで介入前後の関係性変化を評価
        # （実装には logs の詳細な追跡が必要）

    if total_interventions == 0:
        return 0.0

    return successful_interventions / total_interventions
```

## 参考

- 参考実装: https://github.com/yoshinari0312/policyRL_relation_robot
- Azure OpenAI ドキュメント: https://learn.microsoft.com/azure/ai-services/openai/

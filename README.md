# Multi-Party Conversation Facilitation System

## 概要
本プロジェクトは、複数人の対話をリアルタイムに録音・文字起こしし、Pyannote と SpeakerRecognition による話者分離・話者識別を行います。さらに、NetworkX と OpenAI GPT API を使って「関係性グラフ」を可視化し、よりバランスの取れた関係性を目指したロボット介入プランを自動生成・実行します。Pepper 連携に対応。

## 主な機能
1. **リアルタイム録音 & VAD（realtime_communicator.py）**  
   - WebRTC-VAD (感度モード 0–3) で沈黙検知  
   - `SILENCE_DURATION` 秒以上の無音で区切ってバッファ化

2. **話者分離・話者識別（realtime_communicator.py）**  
   - Pyannote Audio でダイアライゼーション  
   - SpeechBrain ECAPA-VOXCELEB で埋め込み比較による話者ラベリング

3. **音声文字起こし（realtime_communicator.py）**  
   - Google Cloud Speech-to-Text v1 ストリーミング

4. **関係性グラフ & 介入ロジック**  
   - **community_analyzer.py**  
     - NetworkX でエッジに GPT+EMA 評価スコア (-1.0〜+1.0) を付与  
     - 力学モデルを用いてレイアウトし、連番で画像出力  
     - Pepper への介入指示（音声＆アニメーション）を管理  
   - **intervention_planner.py**  
     - “孤立検出”“不安定三角形検出”“弱リンク促進” など複数戦略を実装  
     - 過去発話回避／モード切替（少発話者フォロー／ランダム呼びかけ）対応

5. **Web サーバ & UI（graph_realtime_web.py）**  
   - Flask + Flask-SocketIO でリアルタイム制御・更新中継  
   - **index.html**（`templates/`）と JS で、録音制御ボタン・グラフ・会話ログを表示  
   - Socket.IO イベント：  
     - `control`（録音開始／停止）  
     - `conversation_update`（発話ログ更新）  
     - `robot_speak` → `conversation_update` と `robot_log` 中継  
     - `graph_updated` → `refresh_graph` 中継（画像キャッシュ回避付き）

6. **Pepper 連携（community_analyzer.py）**  
   - TCP ソケット経由で音声合成指示を送信

7. **ロギング & 可視化**  
   - `LOG_ROOT` 配下にタイムスタンプ付きディレクトリを自動生成  
   - 会話ログ（`conversation.txt`）、ターミナルログ（`terminal.txt`）、グラフ画像（`relation_graph0.png`, `relation_graph1.png`, …）を保存  
   - Web UI でログディレクトリ名を元に画像リフレッシュ対応

## 実行手順
1. **サーバ起動**  
   python graph_realtime_web.py
   
2. **ブラウザで UI を開く**  
   URL: http://localhost:8888

3. **リアルタイム録音ワーカー起動**  
   python realtime_communicator.py

4. **対話開始**

## パラメータ。括弧内はデフォルト値
### realtime_communicator.py
- **SILENCE_DURATION**: 無音時間（0.5秒）
- **N_BATCH**: セッション分割判定の頻度（5発話）
- **USE_GOOGLE_STT**: 音声認識モデルの設定 (gcp v1)
- **USE_DIRECT_STREAM**: マイクから直接ストリーミング (False)
- **DIARIZATION_THRESHOLD**: この秒数以上なら話者分離する（5秒）
- **SKIP_THRESHOLD_BYTES**: このバイト数以下なら処理スキップ (30000バイト←大体「うん」以上)
- **register_reference_speaker**: 話者を登録
- **process_audio** か **process_audio_batch** か: セッション判定の頻度（5発話に1回）
- **add_utterance** か **add_utterance_count** か: セッションを意味判定か、固定発話数（10）か

### session_manager.py
- **utterances_per_session**: 何会話分を分析するか（10発話）
- **analyze_every**: 何発話ごとに関係性推定するか（5発話）

### community_analyzer.py
- **pepper_ip**: IPアドレス
- **use_robot**: Pepperを使用するか（True）
- **robot_included**: ロボットを関係性学習に組み込むか（機能しないかも）（False）
- **mode**: 条件切り替え
- **rc.set_robot_count(5)**: 介入決定してから何発話、ロボットを話者識別に組み込むか（5）

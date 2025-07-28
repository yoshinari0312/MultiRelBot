import threading
from openai import OpenAI
from dotenv import load_dotenv
import os
from community_analyzer import CommunityAnalyzer
from typing import List, Dict
import queue

# 環境変数からOpenAI APIキーを読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SessionManager:
    """
    音声認識による発話ログを受け取り、時間と話題に基づいて会話セッションを自動的に分割するクラス。
    セッションが確定したらQueueに送ることで、後段の分析処理と非同期に連携できる。
    """

    def __init__(self, time_threshold_sec=90, utterances_per_session: int = 10, analyze_every: int = 5):
        """
        :param time_threshold_sec: 発話間の時間差がこの秒数を超えるとセッションが切り替わる
        """
        self.time_threshold_sec = time_threshold_sec
        self.utterances_per_session = utterances_per_session
        self.analyze_every = analyze_every
        self.lock = threading.Lock()  # スレッドセーフに処理するためのロック
        # 最新10発話を保持するスライディングウィンドウ
        from collections import deque
        self.current_session = deque(maxlen=self.utterances_per_session)
        self.total_utterance_count = 0
        self.history = []             # 現在のセッションの発話履歴（話題判定に使う）
        self.analyzer = CommunityAnalyzer()

        self.topic_queue = queue.Queue()
        self.topic_thread = threading.Thread(target=self._topic_worker, daemon=True)
        self.topic_thread.start()

        # 新：バッチ判定用キュー
        self.batch_queue = queue.Queue()
        self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.batch_thread.start()

    def add_utterance_count(self, log: Dict):
        """
        固定発話数でセッションを切るメソッド。
        呼ばれるたびに current_session に追加し、
        5 発話ごとに analyzer.update(最新10発話) を実行。
        """
        with self.lock:
            self.total_utterance_count += 1
            # 最新10発話を保持
            self.current_session.append(log)
            # 5 発話ごとに分析
            if self.total_utterance_count % self.analyze_every == 0:
                window = list(self.current_session)
                self.analyzer.update(window)
                print(f"分析実行（{self.analyze_every}発話ごと）: 最新{len(window)}発話を使用")

    def add_utterance(self, log):
        """
        新しい発話ログを受け取り、現在のセッションに追加するか、新しいセッションを開始するかを判定する。

        :param log: dict形式の発話ログ（{"time": datetime, "speaker": str, "utterance": str}）
        """
        with self.lock:
            if not self.current_session:
                # 最初の発話 → セッションを開始
                self.current_session.append(log)
                self.history.append(log["utterance"])
                return

            # 現在のセッションの最後の発話との時間差を計算
            last_time = self.current_session[-1]["time"]
            time_diff = (log["time"] - last_time).total_seconds()

            if time_diff > self.time_threshold_sec:
                if len(self.current_session) >= 2:
                    self.analyzer.update(self.current_session)
                    print("セッション確定（時間による切替）")
                self._finalize_and_start_new_session(log)
            else:
                # GPT判定が必要 → 非同期処理に渡す
                self.topic_queue.put(log)

    def add_utterance_batch(self, logs: List[dict]):
        """
        logs: 新しく溜めた n 件の発話ログ
        これらをまとめて「話題継続 or セッション切替」を判定する
        """
        with self.lock:
            # まだセッションが空なら普通に追加
            if not self.current_session:
                self.current_session.extend(logs)
                self.history.extend([log["utterance"] for log in logs])
                return

            # 時間差判定は最初のログだけ見れば OK
            last_time = self.current_session[-1]["time"]
            time_diff = (logs[0]["time"] - last_time).total_seconds()
            if time_diff > self.time_threshold_sec:
                # 時間切れでセッション確定
                if len(self.current_session) >= 2:
                    self.analyzer.update(self.current_session)
                # 全部新セッション
                self.current_session = logs.copy()
                self.history = [log["utterance"] for log in logs]
                return

            # —— 時間内なら、まとめて話題判定 —— #
            sentences = [log["utterance"] for log in logs]
            same = self._are_same_topic(self.history, sentences)

            if same:
                # 全部同じ話題と見なして一気に追加
                self.current_session.extend(logs)
                self.history.extend(sentences)
                print("セッション継続（バッチ）")
            else:
                # どこかで話題切替と判定
                if len(self.current_session) >= 2:
                    self.analyzer.update(self.current_session)
                print("セッション確定（バッチ）")
                # logs 全体を新セッションに
                self.current_session = logs.copy()
                self.history = sentences.copy()

    def _topic_worker(self):
        while True:
            log = self.topic_queue.get()
            with self.lock:
                if not self.current_session:
                    self.current_session.append(log)
                    self.history.append(log["utterance"])
                    continue

                same_topic = self._is_same_topic(self.history, log["utterance"])

                if same_topic:
                    self.current_session.append(log)
                    self.history.append(log["utterance"])
                    print("セッション継続")
                else:
                    if len(self.current_session) >= 2:
                        self.analyzer.update(self.current_session)
                        print("セッション確定")
                    self._finalize_and_start_new_session(log)
            self.topic_queue.task_done()

    def _batch_worker(self):
        """
        バッチごとに話題継続判定＆セッション分割を非同期で行う
        """
        while True:
            logs: List[Dict] = self.batch_queue.get()
            try:
                self.add_utterance_batch(logs)
            finally:
                self.batch_queue.task_done()

    def _finalize_and_start_new_session(self, log):
        self.current_session = [log]
        self.history = [log["utterance"]]

    def finalize(self):
        """
        現在保持しているセッションを強制的に確定してキューに送る。
        スクリプト終了時などに呼び出すことを想定。
        """
        with self.lock:
            if len(self.current_session) >= 2:
                self.analyzer.update(self.current_session)
            self.current_session = []
            self.history = []

    def _is_same_topic(self, history_utterances, current_utterance):
        """
        GPT-4oを使って、これまでの発話と新しい発話が同じ話題かを判定する。

        :param history_utterances: 過去の発話リスト
        :param current_utterance: 今回追加される発話
        :return: 同じ話題ならTrue、違う話題ならFalse
        """
        history_text = "\n".join([f"- {u}" for u in history_utterances])
        prompt = f"""以下は、ある会話のログの一部です。
この流れの中で、最後の発話が【同じ話題の続きかどうか】をYes/Noで判定してください。

【これまでの会話】
{history_text}

【最後の発話】
- {current_utterance}

この発話は、上の会話と同じ話題の続きですか？ Yes か No で答えてください。
"""
        res = client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "あなたは会話分析の専門家です。話題の変化に敏感です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return "yes" in res.choices[0].message.content.strip().lower()

    def _are_same_topic(self, history_utterances, current_utterances):
        """
        GPT-4oを使って、これまでの発話と新しい発話群が同じ話題かを判定する。

        :param history_utterances: 過去の発話リスト
        :param current_utterances: 今回追加される発話リスト
        :return: 同じ話題ならTrue、違う話題ならFalse
        """
        history_text = "\n".join([f"- {u}" for u in history_utterances])
        current_text = "\n".join([f"- {u}" for u in current_utterances])
        prompt = f"""以下は、ある会話のログの一部です。
この流れの中で、最近の会話が【同じ話題の続きかどうか】をYes/Noで判定してください。

【これまでの会話】
{history_text}

【最近の会話】
{current_text}

最近の会話は、これまでの会話と同じ話題の続きですか？ Yes か No で答えてください。最近の会話の途中で話題が変わっている場合は、No としてください。
"""
        res = client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "あなたは会話分析の専門家です。話題の変化に敏感です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return "yes" in res.choices[0].message.content.strip().lower()

    def get_current_session_logs(self) -> List[Dict]:
        """
        現在のセッションログ（時刻・話者・発話）を返す
        """
        return self.current_session

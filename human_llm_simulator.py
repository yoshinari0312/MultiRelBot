"""
人間LLM会話シミュレーター
Azure OpenAI を使用して3名の人間(A, B, C)の会話を生成
"""

import os
import random
from typing import List, Dict, Optional
from dotenv import load_dotenv
import config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from log_filtering import filter_logs_by_human_count

load_dotenv()
_CFG = config.get_config()


class HumanLLMSimulator:
    """Azure OpenAI (HUMAN_MODEL) を使用して人間の会話を生成"""

    def __init__(self, personas: List[str], persona_triggers: Dict[str, List[str]]):
        """
        Args:
            personas: 話者のリスト (例: ["A", "B", "C"])
            persona_triggers: 各話者の地雷キーワード辞書
        """
        self.personas = personas
        self.persona_triggers = persona_triggers

    def build_human_prompt(
        self,
        speaker: str,
        logs: List[Dict],
        topic: str = None,
        topic_trigger: str = None,
        max_history: int = 12,
    ) -> str:
        """
        人間LLM用のプロンプトを構築

        Args:
            speaker: 話者名 (A, B, C)
            logs: 会話履歴
            topic: 会話のトピック
            topic_trigger: トピックの地雷（不機嫌になるトリガー）
            max_history: 会話履歴の最大数

        Returns:
            プロンプト文字列
        """
        # 直近 max_history 個の人間発話 + その間のロボット発話を取得
        filtered_logs = filter_logs_by_human_count(logs, max_history, exclude_robot=False)

        # 会話履歴をテキスト化
        history = "\n".join(
            [f"[{entry['speaker']}] {entry['utterance']}" for entry in filtered_logs]
        )
        if not history:
            history = "(まだ発言なし)"

        # 話者の地雷を取得
        triggers = self.persona_triggers.get(speaker, [])

        lines = [
            f"あなたは会話参加者の一人「{speaker}」です。実在の人のように、状況に応じて態度や言い方が微妙に変わります。\n",
            f"会話は A/B/C(/ロボット) で進みます。あなたは {speaker} として、会話履歴に続けて、1–2文で自然に応答してください。\n",
            f"{speaker}の自然な会話の発話のみを生成してください。\n",
        ]

        if topic:
            lines.append(f"会話のテーマ：{topic}\n")

        lines.append(
            "会話の開始時点では、他の参加者がまだ発言していない場合もあります。その場合は話題に沿って発言してください\n"
        )

        # 地雷システム：topic_triggerが自分の地雷リストに含まれている場合のみ不機嫌に
        if triggers and topic_trigger and topic_trigger in triggers:
            # 他の参加者を取得
            other_speakers = [p for p in self.personas if p != speaker]
            other_names = "、".join(other_speakers) if other_speakers else "他の参加者"

            # 誰が同じ地雷を持っているか確認
            who_has_same_trigger = [
                s
                for s in self.personas
                if topic_trigger in self.persona_triggers.get(s, [])
            ]

            if len(who_has_same_trigger) == 3:  # 3人全員が持っている
                other_angry = [s for s in who_has_same_trigger if s != speaker]
                angry_names = "、".join(other_angry)
                lines.append(
                    f"最初はあなた（{speaker}）も{angry_names}も全員が不機嫌な状態です。"
                    f"全員がイライラしているため、お互いに強い口調で不満を表現し、意見が対立します。\n"
                )
            elif len(who_has_same_trigger) == 2:  # 2人が持っている
                other_angry = [s for s in who_has_same_trigger if s != speaker]
                if other_angry:
                    angry_name = other_angry[0]
                    calm_person = [
                        s for s in self.personas if s not in who_has_same_trigger
                    ]
                    calm_name = calm_person[0] if calm_person else "他の参加者"
                    lines.append(
                        f"最初はあなた（{speaker}）も{angry_name}も不機嫌な状態です。"
                        f"{angry_name}に対して強い口調で不満を表現し、意見が対立します。{calm_name}には穏やかに接します。\n"
                    )
                else:
                    # 自分だけが地雷を持っている
                    lines.append(
                        f"最初はあなただけが不機嫌で、{other_names}など他の参加者に対して強い口調で不満を表現し、意見が対立します。\n"
                    )
            else:
                # 自分だけが地雷を持っている
                lines.append(
                    f"最初は不機嫌で、{other_names}など他の参加者に対して少し不満を含んだ言い方になることがあります。\n"
                )

            lines.append("あなたの態度（機嫌や協力度）は、他者やロボットの発言内容によって変化します。\n")
            lines.append(
                "他者やロボットから共感・橋渡し・建設的な助言を受けた場合、だんだん機嫌が良くなります。\n"
                "逆に、的確で納得できる介入を受け取れなかった場合は、だんだん機嫌が悪くなります。\n"
                "一般的な人間もそうですがいきなり態度を変えるのではなく、徐々に変化します。\n"
            )

        lines.extend(
            [
                f"{speaker}として、次の発話を1-2文で自然な日本語で生成してください。\n",
                "",
                "--- 会話履歴 ---",
                history,
                "",
                f"[{speaker}の返答]",
            ]
        )

        return "\n".join(lines)

    def generate_utterance(
        self,
        speaker: str,
        logs: List[Dict],
        topic: str = None,
        topic_trigger: str = None,
    ) -> str:
        """
        1人の話者の発話を生成

        Args:
            speaker: 話者名
            logs: 会話履歴
            topic: 会話のトピック
            topic_trigger: トピックの地雷

        Returns:
            生成された発話テキスト
        """
        # HUMAN_MODEL を使用するためのクライアント取得
        client, deployment = get_azure_chat_completion_client(
            _CFG.llm, model_type="human"
        )
        if not client or not deployment:
            raise RuntimeError(
                "Failed to obtain Azure OpenAI client for human utterance generation."
            )

        # プロンプト構築
        max_history = getattr(_CFG.env, "max_history_human", 12) or 12
        persona_prompt = self.build_human_prompt(
            speaker, logs, topic, topic_trigger, max_history
        )

        messages = [
            {
                "role": "system",
                "content": "あなたは会話参加者の一人として、自然で短い日本語の返答を行います。",
            },
            {"role": "user", "content": persona_prompt},
        ]

        # Azure OpenAI で発話生成
        max_attempts = getattr(_CFG.llm, "max_attempts", 5) or 5
        base_backoff = getattr(_CFG.llm, "base_backoff", 0.5) or 0.5

        for attempt in range(1, max_attempts + 1):
            try:
                params = build_chat_completion_params(
                    deployment, messages, _CFG.llm, temperature=0.8
                )
                res = client.chat.completions.create(**params)

                if res and getattr(res, "choices", None):
                    choice = res.choices[0]
                    message = getattr(choice, "message", None)

                    if isinstance(message, dict):
                        text = message.get("content", "")
                    else:
                        text = getattr(message, "content", "")

                    text = (text or "").strip()

                    if text:
                        return text

            except Exception as exc:
                print(f"[HumanLLM] attempt {attempt} failed: {exc}")
                if attempt < max_attempts:
                    import time

                    time.sleep(base_backoff * (2 ** (attempt - 1)))

        # フォールバック
        return f"はい、{topic}について考えてみましょう。"

    def generate_human_reply(
        self,
        logs: List[Dict],
        topic: str = None,
        topic_trigger: str = None,
        num_speakers: int = 1,
    ) -> List[Dict]:
        """
        人間の発話を生成（1人または複数人）

        Args:
            logs: 会話履歴
            topic: 会話のトピック
            topic_trigger: トピックの地雷
            num_speakers: 生成する発話数

        Returns:
            発話のリスト [{"speaker": "A", "utterance": "..."},...]
        """
        replies = []
        working_logs = list(logs)  # コピー

        # 最後の人間発話者を取得し、次の話者から開始
        last_human_speaker = None
        for log in reversed(logs):
            if log.get("speaker") != "ロボット":
                last_human_speaker = log.get("speaker")
                break

        # ローテーション開始位置を決定
        start_idx = 0
        if last_human_speaker and last_human_speaker in self.personas:
            try:
                last_idx = self.personas.index(last_human_speaker)
                start_idx = (last_idx + 1) % len(self.personas)
            except ValueError:
                pass

        # speakersをローテート
        speakers = self.personas[start_idx:] + self.personas[:start_idx]

        for speaker in speakers:
            if len(replies) >= num_speakers:
                break

            # 発話生成
            utterance = self.generate_utterance(
                speaker, working_logs, topic, topic_trigger
            )

            reply = {"speaker": speaker, "utterance": utterance}
            replies.append(reply)
            working_logs.append(reply)

        return replies

import os
import re
import pyaudio
import wave
import time
import queue
import threading
from io import BytesIO
from openai import OpenAI
import webrtcvad
import torch
import numpy as np
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition
from huggingface_hub import login
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import torchaudio
import soundfile as sf
from datetime import datetime
from session_manager import SessionManager
import socketio
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from typing import List
from google.cloud import speech
import sys
import config

load_dotenv()

socketio_cli = socketio.Client()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Hugging Face にログイン
login(token=os.getenv("HUGGINGFACE_TOKEN"))
# Google Cloud Speech-to-Text のプロジェクトID
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
RECOGNIZER = f"projects/{PROJECT_ID}/locations/eu/recognizers/my-recognizer"

# 話者識別モデルのロード
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

# GPU を使用できるなら使用
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}, GPUを使うか? {torch.backends.mps.is_available()}")
diarization_pipeline.to(device)

embedding_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# 日本語文字（ひらがな・カタカナ・CJK統合漢字）の正規表現
_jp_regex = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')

# 録音設定
SAMPLE_RATE = 16000  # Whisperの推奨サンプルレート
CHUNK = 160  # 16-bit PCM (1サンプル=2バイト) → 160サンプル = 320バイト
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # モノラル
VAD_MODE = 3  # 0(敏感) - 3(鈍感) (webrtcVADで沈黙検知)
SILENCE_DURATION = 0.5  # 無音が続いたら送信する秒数
MIN_SPEAKERS = 1  # 最小の話者数
MAX_SPEAKERS = 4  # 最大の話者数
# NUM_SPEAKERS = 3  # 確定話者数
SIMILARITY_THRESHOLD = -10  # 既存の話者との類似度の閾値
N_BATCH = 5  # セッション分割判定のバッチサイズ
ROBOT_UTTERANCE_REMAIN = 0  # ロボット発話後にロボット識別を有効にする残り発話数
USE_GOOGLE_STT = "v1"  # "v1": v1の非ストリーミング、"v1-streaming": v1のストリーミング、 "v2-streaming": v2のストリーミング、False: OpenAI
USE_DIRECT_STREAM = False  # True にするとマイクチャンクを直接STTへ流し込む
DIARIZATION_THRESHOLD = 5  # 話者分離するかどうかの閾値
SKIP_THRESHOLD_BYTES = 30000  # 音声データのバイト数がこの値以下なら処理をスキップ

# 音声アクティビティ検出 (VAD)
vad = webrtcvad.Vad(VAD_MODE)

# 既存の話者情報 (話者名: embedding)
known_speakers = {}

# 音声データのキュー（録音スレッド → 処理スレッドへ渡す）
audio_queue = queue.Queue()

# 会話履歴ログ
conversation_log = []

stt_requests_queue = queue.Queue()
stream_buffer: List[bytes] = []

# === セッション管理の初期化 ===
session_manager = SessionManager()

recording_enabled = True

# テキスト結合用の変数
buffer_speaker = None
buffer_text = ""
buffer_time = None

# 標準出力・エラーをログファイルにリダイレクト
original_stdout = sys.stdout
original_stderr = sys.stderr

# ログファイルを開く
log_path = os.path.join(config.LOG_ROOT, "terminal.txt")
terminal_log = open(log_path, "w", encoding="utf-8")


# 両方のストリームに書き込む Tee クラス
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(original_stdout, terminal_log)
sys.stderr = Tee(original_stderr, terminal_log)


# recording_enabledをスタートボタン押したらTrue、ストップボタン押したらFalseにする
@socketio_cli.on("control")
def on_control(data):
    global recording_enabled
    recording_enabled = (data.get("action") == "start")
    print(f"🔔 control: recording_enabled = {recording_enabled}")


def set_robot_count(n):
    global ROBOT_UTTERANCE_REMAIN
    ROBOT_UTTERANCE_REMAIN = n


def dec_robot_count():
    global ROBOT_UTTERANCE_REMAIN
    if ROBOT_UTTERANCE_REMAIN > 0:
        ROBOT_UTTERANCE_REMAIN -= 1


def get_robot_count():
    return ROBOT_UTTERANCE_REMAIN


def is_japanese(text: str) -> bool:
    return bool(_jp_regex.search(text))


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


def transcribe_streaming_v2(stream_file: str) -> cloud_speech.StreamingRecognizeResponse:
    """オーディオファイルストリームから Google Cloud Speech-to-Text API を使用して音声を文字起こし
    Args:
        stream_file (str): 文字起こし対象のローカルオーディオファイルへのパス。
        例: "resources/audio.wav"
    Returns:
        list[cloud_speech.StreamingRecognizeResponse]:
        各オーディオセグメントに対応する文字起こし結果を含むオブジェクトのリスト。
    """
    # クライアントのインスタンス
    client = SpeechClient(client_options=ClientOptions(api_endpoint="eu-speech.googleapis.com"))

    # ファイルをバイト列として読み込む
    with open(stream_file, "rb") as f:
        audio_content = f.read()

    # ⚠️ v2 の audio chunk は最大 25600 bytes まで
    MAX_BYTES = 25600
    stream = [
        audio_content[i : i + MAX_BYTES]
        for i in range(0, len(audio_content), MAX_BYTES)
    ]

    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["ja-JP"],
        model="long",
        features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),  # 句読点挿入
    )
    streaming_config = cloud_speech.StreamingRecognitionConfig(config=recognition_config)
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=RECOGNIZER,
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        yield config
        yield from audio

    # 音声をテキストに文字起こし
    try:
        responses_iterator = client.streaming_recognize(
            requests=requests(config_request, audio_requests)
        )
    except Exception as e:
        print(f"⚠️ Google Cloud Speech-to-Text エラー: {e}")
        return []
    responses = []
    for response in responses_iterator:
        responses.append(response)

    return responses


def transcribe_streaming_v1(stream_file: str) -> list[speech.StreamingRecognizeResponse]:
    """v1 API でファイルを読み込んでストリーミング認識を行い、StreamingRecognizeResponse のリストを返す"""
    # クライアントのインスタンス
    client = speech.SpeechClient()

    # ファイルをバイト列として読み込む
    with open(stream_file, "rb") as f:
        audio_content = f.read()

    # v1 では大きすぎるチャンクを避ける程度に約32KBごとに切り出し
    CHUNK_BYTES = 32000
    chunks = [
        audio_content[i : i + CHUNK_BYTES]
        for i in range(0, len(audio_content), CHUNK_BYTES)
    ]
    audio_requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk)
        for chunk in chunks
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ja-JP",
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
    )

    # ストリーミング認識ジェネレータ
    def requests_gen():
        yield from audio_requests

    # 認識実行＆結果収集
    responses: list[speech.StreamingRecognizeResponse] = []
    try:
        for resp in client.streaming_recognize(streaming_config, requests_gen()):
            responses.append(resp)
    except Exception as e:
        print(f"⚠️ v1 ストリーミング認識エラー: {e}")
    return responses


def transcribe_v1(stream_file: str) -> speech.RecognizeResponse:
    """オーディオファイルを Google Cloud Speech-to-Text v1 の 非ストリーミング Recognize API で文字起こし"""
    # クライアントのインスタンス
    client = speech.SpeechClient()

    # ファイルをバイト列として読み込む
    with open(stream_file, "rb") as f:
        content = f.read()

    # 認識設定 (LINEAR16, 16kHz, 日本語)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ja-JP",
        enable_automatic_punctuation=True,
    )
    audio = speech.RecognitionAudio(content=content)

    # 非ストリーミング認識を実行して結果を返す
    return client.recognize(config=config, audio=audio)


def is_speech(frame, sample_rate):
    """VAD を使って音声か無音かを判定"""
    if len(frame) not in [160, 320, 480]:  # フレームサイズをチェック
        print(f"⚠️ フレームサイズが異常: {len(frame)} バイト (許容値: 160, 320, 480)")
        return False
    return vad.is_speech(frame, sample_rate)


def extract_embedding(audio_input):
    """音声データから話者の特徴量（embedding）を取得"""

    if isinstance(audio_input, str):  # ファイルパスの場合
        with open(audio_input, "rb") as f:
            audio_buffer = BytesIO(f.read())
    else:  # 既に `BytesIO` の場合
        audio_buffer = audio_input

    audio_buffer.seek(0)

    with wave.open(audio_buffer, "rb") as wf:
        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    # 入力が極端に短い場合は処理をスキップ
    if len(audio_data) < 1600 * 6:  # 約0.6秒未満のデータをスキップ
        print("⚠️ 音声データが短い：話者識別スキップ")
        return None

    # モデルに入力できる形式に変換
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0) / 32768.0

    # Embedding（特徴量）を取得
    with torch.no_grad():
        embedding = embedding_model.encode_batch(audio_tensor).squeeze(0)

    return embedding


def register_reference_speaker(speaker_name, reference_audio_path):
    """ 参考音声から話者埋め込みを作成し登録 """
    reference_embedding = extract_embedding(reference_audio_path)
    known_speakers[speaker_name] = reference_embedding


def identify_speaker(audio_buffer):
    """話者識別：新しい話者なら登録、既存の話者ならラベル付け"""
    import realtime_communicator as rc
    global known_speakers
    robot_utterance_remain = rc.get_robot_count()
    print(f"ロボット発話残り: {robot_utterance_remain}")

    new_embedding = extract_embedding(audio_buffer)
    if new_embedding is None:
        return "未識別"

    new_embedding = new_embedding.flatten()

    best_match = None
    best_score = -1.0

    # 参考音声を使って話者を特定
    for speaker, emb in known_speakers.items():
        # カウンターが 0 ならロボットモデルはスキップ
        if speaker == "ロボット" and robot_utterance_remain <= 0:
            continue

        similarity = 1 - cosine(new_embedding, emb.flatten())
        print(f"🔍 類似度（{speaker}）: {similarity:.2f}")
        if similarity > SIMILARITY_THRESHOLD and similarity > best_score:  # SIMILARITY_THRESHOLD 以下なら新規登録。以上なら既存話者から類似度が最も大きいものを選択
            best_score = similarity
            best_match = speaker

    if best_match:
        result = best_match
    else:
        # 新しい話者として登録
        new_speaker_id = f"話者_{len(known_speakers) + 1}"
        known_speakers[new_speaker_id] = new_embedding
        result = new_speaker_id

    if robot_utterance_remain > 0:
        rc.dec_robot_count()  # ロボット発話残りをデクリメント

    return result


def diarize_audio(audio_buffer):
    """話者分離を適用して、話者ごとの音声を分離"""
    print(f"音声ファイルの話者ごと分離開始{datetime.now()}")
    audio_buffer.seek(0)
    waveform, sample_rate = torchaudio.load(audio_buffer)
    diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
    # diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=NUM_SPEAKERS)

    # 🔹 発話順を保持するため、リストを使用
    speaker_timeline = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append((speaker, turn.start, turn.end))

    print(f"音声ファイルの話者ごと分離終了{datetime.now()}")
    return speaker_timeline, waveform, sample_rate


def record_audio():
    """
    音声録音スレッド
    音声が無音になったら、音声をキューに追加
    """
    print("🔴 録音開始...")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    silence_start_time = None
    has_voice = False

    try:
        while True:
            if not recording_enabled:
                time.sleep(0.1)
                continue
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError:
                print("⚠️ 音声データが取得できませんでした。無音として処理します。")
                data = b"\x00" * CHUNK

            # VAD でほぼ無音なら完全スキップ
            # if not is_speech(data, SAMPLE_RATE):
            #     continue

            frames.append(data)

            if USE_DIRECT_STREAM:
                stt_requests_queue.put(data)
                stream_buffer.append(data)
            else:
                if is_speech(data, SAMPLE_RATE):
                    has_voice = True
                    silence_start_time = None
                else:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= SILENCE_DURATION:
                        if has_voice:
                            # print("沈黙を検知！音声をキューに追加します...")
                            audio_queue.put(frames)
                        frames = []
                        has_voice = False
    except KeyboardInterrupt:
        print("録音終了")

    stream.stop_stream()
    stream.close()
    p.terminate()


def process_audio():
    """
    音声処理スレッド
    音声がキューに追加されるたびに、話者分離と文字起こしを実行
    """
    global buffer_speaker, buffer_text, buffer_time
    while True:
        frames = audio_queue.get()
        if not frames:
            continue

        print(f"音声処理開始：{datetime.now()}")
        audio_buffer = BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        audio_buffer.seek(0)

        # ── DIARIZATION_THRESHOLD 秒未満は話者分離せず「1 人」と見なす ──
        with wave.open(audio_buffer, 'rb') as wf2:
            nframes = wf2.getnframes()
            framerate = wf2.getframerate()
            duration = nframes / framerate
        audio_buffer.seek(0)
        if duration >= DIARIZATION_THRESHOLD:
            speaker_timeline, waveform, sample_rate = diarize_audio(audio_buffer)
        else:
            # 短い音声は「1 人」扱い
            waveform, sample_rate = torchaudio.load(audio_buffer)
            speaker_timeline = [('single', 0.0, duration)]

        # 音声があるバイト数よりも小さければ、処理を全てスキップ
        # print(f"音声データのバイト数: {audio_buffer.getbuffer().nbytes}")
        if audio_buffer.getbuffer().nbytes < SKIP_THRESHOLD_BYTES:
            print("⚠️ 音声データが短い：処理を全てスキップ")
            continue

        prev_speaker = None
        combined_audio_list = []
        current_audio = BytesIO()

        for speaker, start, end in speaker_timeline:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            # 🔹 同じ話者の発話なら音声を結合
            if prev_speaker == speaker:
                combined_audio_list.append(segment_waveform.numpy())
            else:
                # 🔹 話者が変わったら、直前の話者の音声を処理。1人の場合は入らない
                if combined_audio_list:
                    combined_waveform = np.concatenate(combined_audio_list, axis=1)
                    try:
                        sf.write(current_audio, combined_waveform.T, sample_rate, format="WAV")
                        current_audio.seek(0)
                    except Exception as e:
                        print(f"⚠️ WAV 書き込みエラー（複数セグメント）：{e}")
                        # このセグメントの処理をスキップ
                        combined_audio_list = []
                        continue

                    # print(f"話者識別開始（複数）：{datetime.now()}")
                    recognized_speaker = identify_speaker(current_audio)
                    # print(f"話者識別終了（複数）：{datetime.now()}")
                    if recognized_speaker == "未識別":
                        print("⚠️ 話者が識別できなかったため、文字起こしをスキップします。")
                    elif recognized_speaker == "ロボット":
                        print("🤖 ロボットの発話のため、スキップします。")
                    else:
                        print(f"音声認識開始（複数）：{datetime.now()}")
                        if USE_GOOGLE_STT:
                            # Google Cloud Speech-to-Text v2 を使う
                            # --- バッファを書き出してファイル化 ---
                            tmp_path = "audio_segment.wav"
                            with open(tmp_path, "wb") as wf:
                                wf.write(current_audio.getbuffer())
                            if USE_GOOGLE_STT == "v2-streaming":
                                responses = transcribe_streaming_v2(tmp_path)
                            elif USE_GOOGLE_STT == "v1-streaming":
                                responses = transcribe_streaming_v1(tmp_path)
                            elif USE_GOOGLE_STT == "v1":
                                response = transcribe_v1(tmp_path)
                                transcript_text = "".join(
                                    result.alternatives[0].transcript
                                    for result in response.results
                                )
                            if USE_GOOGLE_STT == "v1-streaming" or USE_GOOGLE_STT == "v2-streaming":
                                # 各レスポンスの最初の代替候補を取り出して結合
                                transcript_text = "".join(
                                    res.results[0].alternatives[0].transcript
                                    for res in responses
                                    if res.results
                                )
                        else:
                            # gpt-4o-transcribe を使う
                            resp = client.audio.transcriptions.create(
                                # model="whisper-1",
                                # model="gpt-4o-mini-transcribe",  # 速度重視
                                model="gpt-4o-transcribe",
                                file=("audio_segment.wav", current_audio, "audio/wav"),
                                language="ja"
                            )
                            transcript_text = resp.text
                        print(f"音声認識終了（複数）：{datetime.now()}")
                        if not is_japanese(transcript_text):
                            print(f"⚠️ 話者識別結果が日本語ではありません: {transcript_text}")
                            continue
                        print(f"🧑[{recognized_speaker}] {transcript_text}")
                        timestamp = datetime.now()
                        # 音声認識後に、一つ前と話者が同じなら結合して、発話数をカウント
                        if buffer_speaker == recognized_speaker:
                            # 同一話者なら追記
                            buffer_text += " " + transcript_text
                        else:
                            # 話者が変わったら、まず前のバッファをフラッシュ
                            if buffer_speaker is not None:
                                send_conversation(buffer_speaker, buffer_text)  # 発話ログをブラウザへ送信
                                session_manager.add_utterance_count({
                                    "time": buffer_time,
                                    "speaker": buffer_speaker,
                                    "utterance": buffer_text
                                })
                                log_line = f"[{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}] [{buffer_speaker}] {buffer_text}"
                                conversation_log.append(log_line)
                            # 新しいバッファを開始
                            buffer_speaker = recognized_speaker
                            buffer_text = transcript_text
                            buffer_time = timestamp

                # 🔹 新しい話者のためにリセット
                combined_audio_list = [segment_waveform.numpy()]
                current_audio = BytesIO()

            prev_speaker = speaker

        # 🔹 最後の話者の発話も処理。1人の場合はこっちに入る
        if combined_audio_list:
            combined_waveform = np.concatenate(combined_audio_list, axis=1)
            try:
                sf.write(current_audio, combined_waveform.T, sample_rate, format="WAV")
                current_audio.seek(0)
            except Exception as e:
                print(f"⚠️ WAV 書き込みエラー（最終セグメント）：{e}")
                # 最終セグメントだけスキップして終了
                continue

            # print(f"話者識別開始（1人）：{datetime.now()}")
            recognized_speaker = identify_speaker(current_audio)
            # print(f"話者識別終了（1人）：{datetime.now()}")
            if recognized_speaker == "未識別":
                print("⚠️ 話者が識別できなかったため、文字起こしをスキップします。")
            elif recognized_speaker == "ロボット":
                print("🤖 ロボットの発話のため、スキップします。")
            else:
                print(f"音声認識開始（1人）：{datetime.now()}")
                if USE_GOOGLE_STT:
                    # Google Cloud Speech-to-Text v2 を使う
                    # --- バッファを書き出してファイル化 ---
                    tmp_path = "audio_segment.wav"
                    with open(tmp_path, "wb") as wf:
                        wf.write(current_audio.getbuffer())
                    if USE_GOOGLE_STT == "v2-streaming":
                        responses = transcribe_streaming_v2(tmp_path)
                    elif USE_GOOGLE_STT == "v1-streaming":
                        responses = transcribe_streaming_v1(tmp_path)
                    elif USE_GOOGLE_STT == "v1":
                        response = transcribe_v1(tmp_path)
                        transcript_text = "".join(
                            result.alternatives[0].transcript
                            for result in response.results
                        )
                    if USE_GOOGLE_STT == "v1-streaming" or USE_GOOGLE_STT == "v2-streaming":
                        # 各レスポンスの最初の代替候補を取り出して結合
                        transcript_text = "".join(
                            res.results[0].alternatives[0].transcript
                            for res in responses
                            if res.results
                        )
                else:
                    # gpt-4o-transcribe を使う
                    resp = client.audio.transcriptions.create(
                        # model="whisper-1",
                        # model="gpt-4o-mini-transcribe",  # 速度重視
                        model="gpt-4o-transcribe",
                        file=("audio_segment.wav", current_audio, "audio/wav"),
                        language="ja"
                    )
                    transcript_text = resp.text
                print(f"音声認識終了（1人）：{datetime.now()}")
                if not is_japanese(transcript_text):
                    print(f"⚠️ 音声認識結果が日本語ではありません: {transcript_text}")
                    continue
                print(f"🧑[{recognized_speaker}] {transcript_text}")
                timestamp = datetime.now()
                # 音声認識後に、一つ前と話者が同じなら結合して、発話数をカウント
                if buffer_speaker == recognized_speaker:
                    print("同一話者の発話を検出")
                    # 同一話者なら追記
                    buffer_text += " " + transcript_text
                else:
                    # 話者が変わったら、まず前のバッファをフラッシュ
                    if buffer_speaker is not None:
                        print(f"フラッシュ: {buffer_speaker} - {buffer_text}")
                        send_conversation(buffer_speaker, buffer_text)  # 発話ログをブラウザへ送信
                        session_manager.add_utterance_count({
                            "time": buffer_time,
                            "speaker": buffer_speaker,
                            "utterance": buffer_text
                        })
                        log_line = f"[{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}] [{buffer_speaker}] {buffer_text}"
                        conversation_log.append(log_line)
                    # 新しいバッファを開始
                    print(f"新しいバッファを開始: {recognized_speaker}")
                    buffer_speaker = recognized_speaker
                    buffer_text = transcript_text
                    buffer_time = timestamp
                print(f"音声処理終了：{datetime.now()}")


def process_audio_batch():
    """
    音声処理スレッド
    音声がキューに追加されるたびに、話者分離と文字起こしを実行
    """
    global buffer_speaker, buffer_text, buffer_time
    utterance_buffer = []
    while True:
        frames = audio_queue.get()
        if not frames:
            continue

        print("音声処理中...")
        audio_buffer = BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        audio_buffer.seek(0)

        speaker_timeline, waveform, sample_rate = diarize_audio(audio_buffer)

        prev_speaker = None
        combined_audio_list = []
        current_audio = BytesIO()

        for speaker, start, end in speaker_timeline:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            # 🔹 同じ話者の発話なら音声を結合
            if prev_speaker == speaker:
                combined_audio_list.append(segment_waveform.numpy())
            else:
                # 🔹 話者が変わったら、直前の話者の音声を処理
                if combined_audio_list:
                    combined_waveform = np.concatenate(combined_audio_list, axis=1)
                    sf.write(current_audio, combined_waveform.T, sample_rate, format="WAV")
                    current_audio.seek(0)

                    recognized_speaker = identify_speaker(current_audio)
                    if recognized_speaker == "未識別":
                        print("⚠️ 話者が識別できなかったため、文字起こしをスキップします。")
                    elif recognized_speaker == "ロボット":
                        print("🤖 ロボットの発話のため、スキップします。")
                    else:
                        transcript = client.audio.transcriptions.create(
                            # model="whisper-1",
                            # model="gpt-4o-mini-transcribe",  # 速度重視
                            model="gpt-4o-transcribe",
                            file=("audio_segment.wav", current_audio, "audio/wav"),
                            language="ja"
                        )
                        if not is_japanese(transcript.text):
                            print(f"⚠️ 話者識別結果が日本語ではありません: {transcript.text}")
                            continue
                        print(f"[{recognized_speaker}] {transcript.text}")
                        timestamp = datetime.now()
                        # 音声認識後に、一つ前と話者が同じなら結合して、発話数をカウント
                        if buffer_speaker == recognized_speaker:
                            # 同一話者なら追記
                            buffer_text += " " + transcript.text
                        else:
                            # 話者が変わったら、まず前のバッファをフラッシュ
                            if buffer_speaker is not None:
                                send_conversation(buffer_speaker, buffer_text)  # 発話ログをブラウザへ送信
                                utterance_buffer.append({
                                    "time": buffer_time,
                                    "speaker": buffer_speaker,
                                    "utterance": buffer_text
                                })
                                log_line = f"[{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}] [{buffer_speaker}] {buffer_text}"
                                conversation_log.append(log_line)
                            # 新しいバッファを開始
                            buffer_speaker = recognized_speaker
                            buffer_text = transcript.text
                            buffer_time = timestamp
                        # N_BATCHたまったら非同期キューへ投入
                        if len(utterance_buffer) >= N_BATCH:
                            session_manager.batch_queue.put(utterance_buffer.copy())
                            utterance_buffer.clear()
                        log_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{recognized_speaker}] {transcript.text}"
                        conversation_log.append(log_line)

                # 🔹 新しい話者のためにリセット
                combined_audio_list = [segment_waveform.numpy()]
                current_audio = BytesIO()

            prev_speaker = speaker

        # 🔹 最後の話者の発話も処理
        if combined_audio_list:
            combined_waveform = np.concatenate(combined_audio_list, axis=1)
            sf.write(current_audio, combined_waveform.T, sample_rate, format="WAV")
            current_audio.seek(0)

            recognized_speaker = identify_speaker(current_audio)
            if recognized_speaker == "未識別":
                print("⚠️ 話者が識別できなかったため、文字起こしをスキップします。")
            elif recognized_speaker == "ロボット":
                print("🤖 ロボットの発話のため、スキップします。")
            else:
                transcript = client.audio.transcriptions.create(
                    # model="whisper-1",
                    # model="gpt-4o-mini-transcribe",  # 速度重視
                    model="gpt-4o-transcribe",
                    file=("audio_segment.wav", current_audio, "audio/wav"),
                    language="ja"
                )
                if not is_japanese(transcript.text):
                    print(f"⚠️ 話者識別結果が日本語ではありません: {transcript.text}")
                    continue
                print(f"[{recognized_speaker}] {transcript.text}")
                timestamp = datetime.now()
                # 音声認識後に、一つ前と話者が同じなら結合して、発話数をカウント
                if buffer_speaker == recognized_speaker:
                    # 同一話者なら追記
                    buffer_text += " " + transcript.text
                else:
                    # 話者が変わったら、まず前のバッファをフラッシュ
                    if buffer_speaker is not None:
                        send_conversation(buffer_speaker, buffer_text)  # 発話ログをブラウザへ送信
                        utterance_buffer.append({
                            "time": buffer_time,
                            "speaker": buffer_speaker,
                            "utterance": buffer_text
                        })
                        log_line = f"[{buffer_time.strftime('%Y-%m-%d %H:%M:%S')}] [{buffer_speaker}] {buffer_text}"
                        conversation_log.append(log_line)
                    # 新しいバッファを開始
                    buffer_speaker = recognized_speaker
                    buffer_text = transcript.text
                    buffer_time = timestamp
                # N_BATCHたまったら非同期キューへ投入
                if len(utterance_buffer) >= N_BATCH:
                    session_manager.batch_queue.put(utterance_buffer.copy())
                    utterance_buffer.clear()
                log_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{recognized_speaker}] {transcript.text}"
                conversation_log.append(log_line)


def pcm_to_wav_bytesio(pcm_bytes: bytes,
                       sample_rate: int = 16000,
                       nchannels: int = 1,
                       sampwidth: int = 2) -> BytesIO:
    """ヘッダ無し PCM を WAV バイト列に包む"""
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)   # 16-bit PCM → 2 byte
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_buf.seek(0)
    return wav_buf


def streaming_stt_worker1():
    """マイク入力チャンクを直接 Google STT v1 へ流し込み、is_final ごとに話者分離をトリガー"""
    client = speech.SpeechClient()       # v1 クライアント
    # 最初の config リクエスト
    recog_conf = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ja-JP",
        enable_automatic_punctuation=True,
    )
    streaming_conf = speech.StreamingRecognitionConfig(
        config=recog_conf,
        interim_results=True,
        single_utterance=True,
    )

    SILENCE_FRAMES = 50

    def requests_gen():
        silence_cnt = 0
        while True:
            chunk = stt_requests_queue.get()
            if chunk is None:
                break
            # --- 無音判定（B: クライアント VAD） ---
            if not chunk or not is_speech(chunk, SAMPLE_RATE):
                silence_cnt += 1
                if silence_cnt >= SILENCE_FRAMES:
                    break                           # 無音しきい値超え → ストリーム終了
                chunk = b"\x00" * 320     # 16 kHz × 1ch × 2B × 0.01s
            else:
                silence_cnt = 0                     # リセット
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    while True:
        stream_start = time.time()
        responses = client.streaming_recognize(
            config=streaming_conf,
            requests=requests_gen(),
            timeout=900,
        )
        # ストリーミング認識ループ
        for resp in responses:
            # A: END_OF_SINGLE_UTTERANCE を確定トリガに
            if resp.speech_event_type == speech.StreamingRecognizeResponse.SpeechEventType.END_OF_SINGLE_UTTERANCE:
                if resp.results:
                    result = resp.results[-1]        # 直近の結果
                    result.is_final = True           # フラグ擬似的に立てる
                else:
                    break                            # 結果なしで終了
            for result in resp.results:
                print(result)
                if result.is_final:
                    print("is_finalに入りました")
                    # 確定区間のテキストとタイムスタンプを取得
                    text = result.alternatives[0].transcript
                    elapsed = result.result_end_time.total_seconds()
                    end = stream_start + elapsed
                    # バッファから確定区間の音声を切り出し
                    segment = b"".join(stream_buffer)  # バッファ内の音声データを結合
                    full_buf = pcm_to_wav_bytesio(segment)
                    # 話者分離＆話者認識
                    speaker_tl, _, _ = diarize_audio(full_buf)  # 要検討。絶対2人入らないならコメント
                    speaker = identify_speaker(full_buf)
                    # ログ＆セッション管理
                    timestamp = datetime.fromtimestamp(end)
                    send_conversation(speaker, text)
                    session_manager.add_utterance({
                        "time": timestamp,
                        "speaker": speaker,
                        "utterance": text
                    })
                    print(f"🧑[{speaker}] {text}")
                    log_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{speaker}] {text}"
                    conversation_log.append(log_line)

                    # 次区間用にバッファクリア
                    stream_buffer.clear()


def streaming_stt_worker2():
    """マイク入力チャンクを直接 Google STT v2 へ流し込み、is_final ごとに話者分離をトリガー"""
    client = SpeechClient(client_options=ClientOptions(api_endpoint="eu-speech.googleapis.com"))
    # 最初の config リクエスト
    recog_conf = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["ja-JP"], model="latest_short",
        features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
    )
    streaming_conf = cloud_speech.StreamingRecognitionConfig(
        config=recog_conf,
        # ───── ストリーム維持用の各種フラグ ──────
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True,                  # 部分結果を早めに返す
            enable_voice_activity_events=True,     # VAD イベントも受信
            # voice_activity_timeout=cloud_speech.VoiceActivityTimeout(
            #     start_timeout=duration_pb2.Duration(seconds=5),   # 発話開始待ち
            #     end_timeout=duration_pb2.Duration(seconds=30),    # 発話後の無音
            # ),
        ),
    )

    def requests_gen():
        # 初回: 設定 + recognizer を送信
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=RECOGNIZER,
            streaming_config=streaming_conf,
        )
        # ② 以降: 音声 + recognizer
        while True:
            # 100 ms（約 10 チャンク）分まとめて送信
            buf = []
            try:
                for _ in range(10):
                    buf.append(stt_requests_queue.get(timeout=0.02))
            except queue.Empty:
                pass
            chunk = b"".join(buf)
            if not chunk:                          # 送るものが無ければスキップ
                continue
            # 以降のリクエストにも recognizer を含める
            yield cloud_speech.StreamingRecognizeRequest(
                audio=chunk,
            )

    # ストリーミング認識ループ
    for resp in client.streaming_recognize(requests=requests_gen()):
        print(f"🔄 音声認識結果を受信: {len(resp.results)} 件")
        for result in resp.results:
            print(result)
            if result.is_final:
                print("is_finalに入りました")
                # 確定区間のテキストとタイムスタンプを取得
                text = result.alternatives[0].transcript
                int_sec, dec_sec = result.result_end_time.seconds, result.result_end_time.nanos / 1e9
                end = int_sec + dec_sec
                # バッファから確定区間の音声を切り出し
                segment = b"".join(stream_buffer)  # バッファ内の音声データを結合
                full_buf = BytesIO(segment)
                # 話者分離＆話者認識
                speaker_tl, _, _ = diarize_audio(full_buf)  # 要検討。絶対2人入らないならコメント
                speaker = identify_speaker(full_buf)
                # ログ＆セッション管理
                timestamp = datetime.fromtimestamp(end)
                send_conversation(speaker, text)
                session_manager.add_utterance({
                    "time": timestamp,
                    "speaker": speaker,
                    "utterance": text
                })
                print(f"🧑[{speaker}] {text}")
                log_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{speaker}] {text}"
                conversation_log.append(log_line)

                # 次区間用にバッファクリア
                stream_buffer.clear()


def save_conversation_log():
    if not conversation_log:
        print("🔕 会話ログは空です。")
        return

    filename = os.path.join(config.LOG_ROOT, "conversation.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for line in conversation_log:
            f.write(line + "\n")
    print(f"💾 会話ログを保存しました: {filename}")


# 発話ログをブラウザへ送信
def send_conversation(speaker, utterance):
    if socketio_cli.connected:
        socketio_cli.emit("conversation_update", {
            "speaker": speaker,
            "utterance": utterance
        })


@socketio_cli.on("robot_log")
def on_conversation_update(data):
    speaker = data.get("speaker")
    utterance = data.get("utterance")
    timestamp = datetime.now()
    log_line = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{speaker}] {utterance}"
    conversation_log.append(log_line)
    print("🤖 ロボットのログ記録")


try_connect_socketio()
if __name__ == "__main__":
    register_reference_speaker("ロボット", "static/audio/robot_sample.wav")
    # register_reference_speaker("小野寺", "static/audio/onodera_sample.wav")
    # register_reference_speaker("佐藤", "static/audio/sato_sample.wav")
    # register_reference_speaker("田中", "static/audio/tanaka_sample.wav")
    # register_reference_speaker("今井", "static/audio/imai_sample.wav")
    # register_reference_speaker("大場", "static/audio/oba_sample.wav")
    # register_reference_speaker("馬場", "static/audio/hibiki_sample.wav")
    register_reference_speaker("三宅", "static/audio/serina_sample.wav")
    # register_reference_speaker("けんしん", "static/audio/kenshin_sample.wav")
    register_reference_speaker("立川", "static/audio/kanta_sample.wav")
    register_reference_speaker("松崎", "static/audio/matsuzaki_sample.wav")
    # register_reference_speaker("けいじろう", "static/audio/keijiro_sample.wav")
    # register_reference_speaker("ゆうき", "static/audio/yuki_sample.wav")
    # register_reference_speaker("なかそう", "static/audio/nakasou_sample.wav")
    # register_reference_speaker("なおき", "static/audio/naoki_sample.wav")
    # while True:
    #     record_and_transcribe()
    threading.Thread(target=record_audio, daemon=True).start()
    if USE_DIRECT_STREAM:
        threading.Thread(target=streaming_stt_worker2, daemon=True).start()  # マイク入力チャンクを直接STTへ流し込
    else:
        threading.Thread(target=process_audio, daemon=True).start()  # 0.5秒ごとに音声を処理する
        # threading.Thread(target=process_audio_batch, daemon=True).start()  # バッチ処理で音声を処理する（会話数ごとにセッション分割判定している場合は使わない）

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 終了信号を受け取りました。ログを保存します...")
        session_manager.finalize()  # 最後のセッションをキューに送る
        save_conversation_log()

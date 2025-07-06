import os
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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Hugging Face にログイン
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# 話者識別モデルのロード
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

# 必要なら GPU を使用
if torch.cuda.is_available():
    diarization_pipeline.to(torch.device("cuda"))

embedding_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# 録音設定
SAMPLE_RATE = 16000  # Whisperの推奨サンプルレート
CHUNK = 160  # 16-bit PCM (1サンプル=2バイト) → 160サンプル = 320バイト
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # モノラル
VAD_MODE = 3  # 0(敏感) - 3(鈍感) (webrtcVADで沈黙検知)
SILENCE_DURATION = 1  # 無音が続いたら送信する秒数
MIN_SPEAKERS = 1  # 最小の話者数
MAX_SPEAKERS = 3  # 最大の話者数
NUM_SPEAKERS = 2  # 確定話者数
SIMILARITY_THRESHOLD = 0.0  # 既存の話者との類似度の閾値

# 音声アクティビティ検出 (VAD)
vad = webrtcvad.Vad(VAD_MODE)

# 既存の話者情報 (話者名: embedding)
known_speakers = {}

# 音声データのキュー（録音スレッド → 処理スレッドへ渡す）
audio_queue = queue.Queue()

# 会話履歴ログ
conversation_log = []


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
    if len(audio_data) < 1600:  # 約0.1秒未満のデータをスキップ
        print("⚠️ 音声データが短すぎるため、スキップします。")
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
    global known_speakers

    new_embedding = extract_embedding(audio_buffer)
    if new_embedding is None:
        return "未識別"

    new_embedding = new_embedding.flatten()

    best_match = None
    best_score = -1.0

    # 参考音声を使って話者を特定
    for speaker, emb in known_speakers.items():
        similarity = 1 - cosine(new_embedding, emb.flatten())
        print(f"🔍 類似度（{speaker}）: {similarity:.2f}")
        if similarity > SIMILARITY_THRESHOLD and similarity > best_score:  # SIMILARITY_THRESHOLD 以下なら新規登録。以上なら既存話者から類似度が最も大きいものを選択
            best_score = similarity
            best_match = speaker

    if best_match:
        return best_match

    # 新しい話者として登録
    new_speaker_id = f"話者_{len(known_speakers) + 1}"
    known_speakers[new_speaker_id] = new_embedding
    return new_speaker_id


def diarize_audio(audio_buffer):
    """話者分離を適用して、話者ごとの音声を分離"""
    audio_buffer.seek(0)
    waveform, sample_rate = torchaudio.load(audio_buffer)
    diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
    # diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=NUM_SPEAKERS)

    # 🔹 発話順を保持するため、リストを使用
    speaker_timeline = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append((speaker, turn.start, turn.end))

    return speaker_timeline, waveform, sample_rate


def record_audio():
    """
    音声録音スレッド
    音声が無音になったら、音声をキューに追加
    """
    print("録音中...")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    silence_start_time = None
    has_voice = False

    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError:
                print("⚠️ 音声データが取得できませんでした。無音として処理します。")
                data = b"\x00" * CHUNK

            frames.append(data)

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
                    else:
                        transcript = client.audio.transcriptions.create(
                            # model="whisper-1",
                            # model="gpt-4o-mini-transcribe",  # 速度重視
                            model="gpt-4o-transcribe",
                            file=("audio_segment.wav", current_audio, "audio/wav"),
                            language="ja"
                        )
                        print(f"[{recognized_speaker}] {transcript.text}")
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_line = f"[{timestamp}] [{recognized_speaker}] {transcript.text}"
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
            else:
                transcript = client.audio.transcriptions.create(
                    # model="whisper-1",
                    # model="gpt-4o-mini-transcribe",  # 速度重視
                    model="gpt-4o-transcribe",
                    file=("audio_segment.wav", current_audio, "audio/wav"),
                    language="ja"
                )
                print(f"[{recognized_speaker}] {transcript.text}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"[{timestamp}] [{recognized_speaker}] {transcript.text}"
                conversation_log.append(log_line)


def save_conversation_log():
    if not conversation_log:
        print("🔕 会話ログは空です。")
        return

    os.makedirs("logs", exist_ok=True)
    filename = datetime.now().strftime("logs/conversation6.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for line in conversation_log:
            f.write(line + "\n")
    print(f"💾 会話ログを保存しました: {filename}")


if __name__ == "__main__":
    register_reference_speaker("小野寺", "static/audio/onodera_sample.wav")
    register_reference_speaker("佐藤", "static/audio/sato_sample.wav")
    register_reference_speaker("田中", "static/audio/tanaka_sample.wav")
    # while True:
    #     record_and_transcribe()
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=process_audio, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 終了信号を受け取りました。ログを保存します...")
        save_conversation_log()

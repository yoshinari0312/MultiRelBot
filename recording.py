import os
import pyaudio
import wave
import numpy as np
import time
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 録音設定
SAMPLE_RATE = 16000  # サンプルレート
CHUNK = 160  # フレームサイズ
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # モノラル
THRESHOLD = 500  # 無音判定の閾値
SILENCE_DURATION = 2  # 無音が続いたら停止する秒数
OUTPUT_FILE = "static/audio/nakasou_sample.wav"
SAVE_AUDIO = True  # 音声ファイルを保存するかどうか


def record_audio(sample_rate, chunk, threshold, silence_duration):
    """無音が3秒続くまで録音し、WAVファイルに保存"""
    print("録音を開始します...")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    frames = []
    silence_start_time = None  # 無音が始まった時間

    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)

        # 音量レベルを計算
        audio_array = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_array).mean()

        if volume < threshold:
            if silence_start_time is None:
                silence_start_time = time.time()  # 無音開始時間を記録
            elif time.time() - silence_start_time >= silence_duration:
                print("無音が続いたため録音を停止します。")
                break
        else:
            silence_start_time = None  # 音があればリセット

    # 録音終了
    stream.stop_stream()
    stream.close()
    p.terminate()

    if SAVE_AUDIO:
        # 保存する場合
        with wave.open(OUTPUT_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return OUTPUT_FILE  # ファイル名を返す
    else:
        # 保存せずメモリ内で処理
        audio_buffer = BytesIO()
        with wave.open(audio_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        audio_buffer.seek(0)  # メモリの先頭に移動
        return audio_buffer  # メモリバッファを返す


# 録音実行（ファイル or メモリバッファを取得）
audio_source = record_audio(SAMPLE_RATE, CHUNK, THRESHOLD, SILENCE_DURATION)

if SAVE_AUDIO:
    # 保存した音声ファイルを Whisper に渡す
    with open(audio_source, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language="ja"
        )
else:
    # メモリ内の音声データを Whisper に渡す
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("speech.wav", audio_source, "audio/wav"),
        language="ja"
    )

# 結果を表示
print("音声認識結果:")
print(transcript.text)

def google_stt_worker():
    global vad_speaking_event, in_turn_event, cb_speech_frames, last_text, dialogue_history, playing_event
    global last_greet_time, greeted_this_turn, last_greeted_transcript, last_speech_end_time, last_was_short, AIZUCHI_COUNT
    global ONLY_BC
    import time
    print("[GOOGLE_STT] ストリーミングワーカー開始", flush=True)
    last_chunk_for_bc = ""
    last_greet_time = 0
    greeted_this_turn = False        # このターンで1回だけ挨拶返す用
    last_greeted_transcript = None   # この発話に対して挨拶返答したかどうか記録
    try:
        from google.cloud import speech
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="ja",
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )
        def audio_generator():
            while True:
                try:
                    chunk = stt_buffer.get()
                    if chunk is None:
                        print("[STT_DEBUG] Received termination signal", flush=True)
                        break
                    yield chunk
                except Exception as e:
                    print(f"[STT ERROR] Audio generator error: {e}", flush=True)
                    break
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator()
        )
        print("[STT_DEBUG] before streaming_recognize...", flush=True)
        responses = client.streaming_recognize(streaming_config, requests)
        last_start_time = time.time()  # 最初の応答を受信した時刻
        last_speech_end_time = time.time()  # 最後の発話時刻を初期化
        for response in responses:
            try:
                # if playing_event.is_set():
                #     print("[STT_DEBUG] 応答中のためSTT応答をスキップ", flush=True)
                #     time.sleep(0.1)
                #     continue
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript.strip()
                if not transcript:
                    continue
                # --- ターン開始時 ---
                if not in_turn_event.is_set():
                    last_start_time = time.time()  # ターン開始時刻を更新
                    if not greeted_this_turn:
                        in_turn_event.set()
                        stop_agent_response_event.set()
                    # greeted_this_turn = False
                    last_greeted_transcript = None
                    last_chunk_for_bc = ""  # 最初の発話を記録
                    print(f"[TURN_START] ターン開始: '{transcript}'", flush=True)
                # ---------- interim（途中経過）：相槌のみ ----------
                if not result.is_final:
                    last_speech_end_time = time.time()  # 最後の発話時刻を更新
                    incremental = get_incremental_text(last_chunk_for_bc, transcript)
                    print(f"prev={last_chunk_for_bc}, curr='{transcript}', incremental='{incremental}'", flush=True)
                    # print(f"len(incremental)={len(incremental)}, MAX_BC_LEN={MIN_BC_LEN}, incremental={incremental}'", flush=True)
                    if (contains_punctuation(incremental) and len(incremental) >= MIN_BC_LEN):
                        if playing_event.is_set():
                            stop_agent_response_event.set()  # 相槌検知で応答停止
                        stop_agent_response_event.clear()
                        # print(f"[INTERIM] 相槌検知: '{incremental}'")
                        segments = re.split(r"[、。？！]", transcript)
                        prefix = "、".join(segments[:-1]) if len(segments) > 1 else transcript
                        last_text += prefix.strip()
                        print(f"[INTERIM] 相槌検知: '{incremental}', last text = {last_text}", flush=True)
                        enqueue_backchannel()
                        last_chunk_for_bc = transcript
                    # --- interimで挨拶検知（同じtranscriptで1回だけ） ---
                    gkey = contains_greeting(transcript)
                    if gkey and not greeted_this_turn and transcript != last_greeted_transcript:
                        last_speech_end_time = time.time()  # 挨拶検知時も更新
                        greeted_this_turn = True
                        in_turn_event.clear()  # 挨拶検知でターン終了
                        stop_agent_response_event.clear()  # 相槌検知で応答停止解除
                        last_greeted_transcript = transcript
                        resp = GREETINGS_RESPONSE_MAP[gkey]
                        print(f"[GREETING_TRIGGER] 挨拶検出: '{gkey}' → '{resp}'", flush=True)
                        wav_bytes = TTS_CACHE.get(resp)
                        if wav_bytes:
                            threading.Thread(target=play_wav_bytes, args=(wav_bytes,)).start()
                        last_greet_time = time.time()
                        # 履歴追加・ターン終了
                        user_entry = f"User:{transcript}"
                        dialogue_history.append(user_entry)
                        agent_entry = f"Assistant:{resp}"
                        dialogue_history.append(agent_entry)
                        # in_turn_event.clear()
                        continue  # このターンはここで終わり
                    continue  # interimはここで終わり
                # ---------- final（確定）時 ----------
                if result.is_final:
                    stop_agent_response_event.clear()  # 相槌検知で応答停止解除
                    # print(f"[FINAL] 確定発話: '{transcript}'", flush=True)
                    last_text = transcript
                    vad_speaking_event.clear()
                    print(f"[SPEECH_END] {now_str()} - '{transcript}'", flush=True)
                    if greeted_this_turn:
                        in_turn_event.clear()  # 挨拶後はターン終了
                    user_entry = f"User:{transcript}"
                    dialogue_history.append(user_entry)
                    print(f"[DIALOGUE_HISTORY] ユーザ発話を追加: '{transcript}' (total: {len(dialogue_history)} entries)", flush=True)
                    if greeted_this_turn:
                        dialogue_history.append(f"Assistant:{GREETINGS_RESPONSE_MAP.get(contains_greeting(transcript), '')}")
                        print(f"[GREETING_ADDED] 挨拶応答を履歴に追加: '{GREETINGS_RESPONSE_MAP.get(contains_greeting(transcript), '')}'", flush=True)
                        greeted_this_turn = False  # このターンで挨拶は1回だけ
                        last_speech_end_time = time.time()
                        last_was_short = False  # 挨拶後は短い発話ではない
                    else:
                        if USE_VAP:
                            enqueue_backchannel()
                        else:
                            last_speech_end_time = time.time()
                            last_was_short = len(transcript) < MIN_RESPONSE_LEN
                            if len(transcript) >= MIN_RESPONSE_LEN:
                                if not ONLY_BC:
                                    enqueue_agent_response()
                                else:
                                    enqueue_backchannel()  # 相槌のみ
                            else:
                                print(f"[STT_DEBUG] 発話が短すぎるため相槌をスキップ: '{transcript}'", flush=True)
                                AIZUCHI_COUNT += 1
                                # enqueue_backchannel()
                                # 相槌はスキップ
                    SPEECH_START_END_LIST.append((last_start_time, last_speech_end_time, transcript))
            except Exception as e:
                print(f"[STT ERROR] Response processing error: {e}", flush=True)
                import traceback
                traceback.print_exc(file=sys.stdout)
                continue
    except Exception as e:
        print(f"[GOOGLE_STT ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)


# Google STTワーカースレッド起動
    stt_thread = threading.Thread(target=google_stt_worker, daemon=True)
    stt_thread.start()
    print("[OK] Google STTスレッド起動")
    # コールバック関数の初期化（簡略化）
    print("[OK] コールバック関数初期化完了")
    stream = sd.InputStream(
        samplerate=RATE,
        channels=1,
        dtype="float32",
        blocksize=int(RATE * CHUNK_SEC),
        callback=cb,
    )
    stream.start()
    print(":マイク: READY - speak!")
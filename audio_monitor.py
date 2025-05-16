import speech_recognition as sr
import threading
import numpy as np
import pyaudio
import wave
import time
from datetime import datetime
import os
import sys

def monitor_audio(log_callback, cheating_detect_dir="cheating_detect"):
    try:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        THRESHOLD = 0.1
        SILENCE_LIMIT = 1
        
        log_callback("Initializing audio system...")
        
        p = pyaudio.PyAudio()
        
        log_callback("Available audio devices:")
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            log_callback(f"Device {i}: {dev_info['name']}")
        
        try:
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            log_callback("Successfully opened audio stream")
        except Exception as e:
            log_callback(f"Error opening audio stream: {str(e)}")
            return
        
        try:
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            log_callback("Speech recognition initialized")
        except Exception as e:
            log_callback(f"Error initializing speech recognition: {str(e)}")
            return
        
        last_sound_time = time.time()
        sound_detected = False
        last_save_time = 0
        save_cooldown = 2
        
        def listen_loop():
            nonlocal sound_detected, last_sound_time, last_save_time
            
            try:
                with mic as source:
                    log_callback("Adjusting for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=2)
                    log_callback("Ambient noise adjustment complete")
                    
                    while True:
                        try:
                            audio = recognizer.listen(source, timeout=3)
                            text = recognizer.recognize_google(audio)
                            log_callback(f"Speech detected: {text}")
                            sound_detected = True
                            last_sound_time = time.time()
                        except sr.WaitTimeoutError:
                            pass
                        except sr.UnknownValueError:
                            pass
                        except sr.RequestError as e:
                            log_callback(f"Speech recognition error: {str(e)}")
                        except Exception as e:
                            log_callback(f"Unexpected error in speech recognition: {str(e)}")
            except Exception as e:
                log_callback(f"Error in speech recognition loop: {str(e)}")
        
        speech_thread = threading.Thread(target=listen_loop)
        speech_thread.daemon = True
        speech_thread.start()
        
        log_callback("Audio monitoring started")
        
        try:
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    sound_level = np.abs(audio_data).mean() / 32768.0
                    
                    if sound_level > THRESHOLD:
                        sound_detected = True
                        last_sound_time = time.time()
                        log_callback(f"Sound detected! Level: {sound_level:.3f}")
                        
                        current_time = time.time()
                        if (current_time - last_save_time) >= save_cooldown:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"sound_{timestamp}_level{sound_level:.2f}.wav"
                            filepath = os.path.join(cheating_detect_dir, filename)
                            
                            with wave.open(filepath, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(p.get_sample_size(FORMAT))
                                wf.setframerate(RATE)
                                wf.writeframes(data)
                            
                            log_callback(f"Saved audio sample: {filename}")
                            last_save_time = current_time
                    
                    elif sound_detected and (time.time() - last_sound_time) > SILENCE_LIMIT:
                        sound_detected = False
                        log_callback("Silence detected")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    log_callback(f"Error in main audio loop: {str(e)}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            log_callback("Audio monitoring stopped by user")
        finally:
            log_callback("Cleaning up audio resources...")
            stream.stop_stream()
            stream.close()
            p.terminate()
            log_callback("Audio monitoring cleanup complete")
            
    except Exception as e:
        log_callback(f"Critical error in audio monitoring: {str(e)}")
        sys.exit(1)

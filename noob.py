import sys
import os
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.beat
import threading
import time
import logging
import contextlib
import tracemalloc
from numba import jit
from scipy.signal import butter, lfilter
from pedalboard import Reverb, Compressor, PitchShift
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QGroupBox, QDockWidget, QPushButton, QSlider, QComboBox, QLabel, QStatusBar,
                             QListWidget, QGraphicsView, QGraphicsScene, QFileDialog, QMessageBox, QDialog,
                             QLineEdit, QMenu, QShortcut)
from PyQt5.QtGui import QPainter, QPainterPath, QPen, QBrush, QColor, QLinearGradient, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, QPropertyAnimation

# Loglama
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sabitler
EFFECTS_LIST = [
    "Flanger", "Reverb", "Delay", "Compressor", "PitchShift", "Echo", "Phaser", "Chorus",
    "Distortion", "BitCrusher", "Tremolo", "Vibrato", "Wah", "HighPass", "LowPass", "BandPass",
    "Notch", "Gate", "Limiter", "AutoTune", "Reverse", "Scratch", "Glitch", "Stutter", "Vinyl", "TapeStop"
]
SHORTCUTS = {
    "play_deck1": "Q", "play_deck2": "W", "play_deck3": "E", "play_deck4": "R",
    "cue_deck1": "A", "cue_deck2": "S", "cue_deck3": "D", "cue_deck4": "F",
    "load_deck": "Ctrl+O", "mic_toggle": "T", "media_scan": "Ctrl+S",
    "master_volume_up": "+", "master_volume_down": "-",
    "crossfader_left": "Left", "crossfader_right": "Right",
    "pad_1": "1", "pad_2": "2", "pad_3": "3", "pad_4": "4",
    "pad_5": "5", "pad_6": "6", "pad_7": "7", "pad_8": "8",
    "pad_9": "Z", "pad_10": "X", "pad_11": "C", "pad_12": "V",
    "pad_13": "B", "pad_14": "N", "pad_15": "M", "pad_16": ",",
    "effect_select": "F1-F12",
    "play_pause_player": "Space", "next_track": "Right", "prev_track": "Left",
    "midi_note": "Q", "midi_copy": "C", "midi_paste": "V",
    "metronome_toggle": "M"
}

# Config Okuma
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {
        "devices_dir": "devices",
        "default_profile": {
            "device_id": "realtek_alc1220",
            "name": "Realtek ALC1220",
            "manufacturer": "Realtek",
            "type": "Integrated Audio",
            "driver_type": ["WASAPI"],
            "bit_depth": [16, 24],
            "sample_rates": [44100, 48000, 96000],
            "recommended_sample_rate": 44100,
            "default_buffer_size": 256,
            "min_buffer_size": 128,
            "max_buffer_size": 1024,
            "latency_tolerance": "medium",
            "dsd_support": False,
            "dac_chip": "Realtek ALC1220",
            "phantom_power": False
        }
    }

# Ses Kartı Yöneticisi
class AudioDeviceManager:
    """Sistemdeki ses kartlarını algılar, JSON profillerini yükler ve ses motorunu yapılandırır."""
    
    def __init__(self):
        self.devices_dir = CONFIG.get("devices_dir", "devices")
        self.default_profile = CONFIG.get("default_profile")
        self.profiles = {}
        self.current_device = None
        self.load_profiles()
    
    def load_profiles(self):
        """JSON profillerini belirtilen dizinden yükler."""
        try:
            os.makedirs(self.devices_dir, exist_ok=True)
            for file in os.listdir(self.devices_dir):
                if file.endswith(".json"):
                    with open(os.path.join(self.devices_dir, file), 'r') as f:
                        profile = json.load(f)
                        self.profiles[profile["device_id"]] = profile
            self.profiles[self.default_profile["device_id"]] = self.default_profile
            logging.info(f"{len(self.profiles)} ses kartı profili yüklendi")
        except FileNotFoundError:
            logging.error(f"Dizin bulunamadı: {self.devices_dir}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse hatası: {e}")
    
    def detect_device(self):
        """Sistemdeki ses kartını algılar ve profili eşleştirir."""
        try:
            devices = sd.query_devices()
            detected = devices[sd.default.device["output"]]["name"].lower()
            for device_id, profile in self.profiles.items():
                if device_id in detected or profile["name"].lower() in detected:
                    self.current_device = profile
                    return profile
            self.current_device = self.default_profile
            return self.default_profile
        except sd.PortAudioError as e:
            logging.error(f"Cihaz algılama hatası: {e}")
            return self.default_profile
        except Exception as e:
            logging.error(f"Beklenmeyen hata: {e}")
            return self.default_profile
    
    def configure_audio(self, fs=None, buffer_size=None):
        """
        Ses motorunu belirtilen frekans ve buffer boyutuna göre yapılandırır.
        
        Args:
            fs (int, optional): Örnekleme frekansı (Hz).
            buffer_size (int, optional): Buffer boyutu (örnek sayısı).
        
        Returns:
            tuple: (bool, str) Başarı durumu ve hata mesajı.
        """
        if not self.current_device:
            self.detect_device()
        profile = self.current_device
        fs = fs or profile["recommended_sample_rate"]
        buffer_size = buffer_size or profile["default_buffer_size"]
        try:
            # Platforma göre sürücü seçimi
            if sys.platform == "win32":
                sd.default.extra_settings = {"backend": "wasapi"}
            elif sys.platform == "linux":
                sd.default.extra_settings = {"backend": "alsa"}
            elif sys.platform == "darwin":
                sd.default.extra_settings = {"backend": "coreaudio"}
            sd.default.samplerate = fs
            sd.default.blocksize = buffer_size
            sd.default.device = sd.default.device["output"]
            logging.info(f"Ses yapılandırıldı: {profile['name']}, {fs}Hz, {buffer_size} buffer")
            return True, ""
        except sd.PortAudioError as e:
            logging.error(f"PortAudio hatası: {e}")
            return False, f"Ses kartı yapılandırılamadı: {e}"
        except ValueError as e:
            logging.error(f"Değer hatası: {e}")
            return False, f"Geçersiz ayarlar: {fs}Hz veya {buffer_size} desteklenmiyor"
        except Exception as e:
            logging.error(f"Beklenmeyen hata: {e}")
            return False, "Bilinmeyen bir hata oluştu"

# Neve 1073 Simülatörü
class Neve1073Sim:
    """Neve 1073 preamp simülatörü: düşük, orta ve yüksek frekanslar için EQ."""
    
    def __init__(self, low_gain=0.0, mid_gain=0.0, high_gain=0.0):
        self.low_gain = low_gain / 15.0
        self.mid_gain = mid_gain / 15.0
        self.high_gain = high_gain / 15.0
    
    def process(self, audio, fs):
        """Sinyali EQ ile işler."""
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=1)
        b_low, a_low = butter(2, 200 / (fs / 2), btype='low')
        b_mid, a_mid = butter(2, [500 / (fs / 2), 3000 / (fs / 2)], btype='band')
        b_high, a_high = butter(2, 8000 / (fs / 2), btype='high')
        low = lfilter(b_low, a_low, audio) * (1 + self.low_gain)
        mid = lfilter(b_mid, a_mid, audio) * (1 + self.mid_gain)
        high = lfilter(b_high, a_high, audio) * (1 + self.high_gain)
        return np.clip(low + mid + high, -1.0, 1.0)

# Efekt Rafı
class EffectRack:
    """Gerçek zamanlı ses efektlerini yönetir ve uygular."""
    
    def __init__(self):
        self.effects = {name: {"instance": None, "params": {}} for name in EFFECTS_LIST}
        self.active_effects = []
    
    def add_effect(self, name, params=None):
        """Belirtilen efekti aktif listeye ekler."""
        params = params or {"wet": 0.3}
        try:
            if name == "Reverb":
                effect = Reverb(room_size=params.get("room_size", 0.5), damping=params.get("damping", 0.5), wet_level=params["wet"])
            elif name == "Compressor":
                effect = Compressor(threshold_db=params.get("threshold", -24), ratio=params.get("ratio", 4), wet_level=params["wet"])
            elif name == "PitchShift":
                effect = PitchShift(semitones=params.get("semitones", 0), wet_level=params["wet"])
            elif name == "Flanger":
                effect = lambda audio, fs: fast_flanger(audio, fs, **params)
            else:
                return
            self.active_effects.append({"name": name, "instance": effect, "params": params})
            logging.info(f"Efekt eklendi: {name}")
        except Exception as e:
            logging.error(f"Efekt ekleme hatası: {e}")
    
    def process(self, audio, fs, bpm=None):
        """Sinyali aktif efekt zinciriyle işler."""
        processed = audio.copy()
        try:
            for fx in self.active_effects:
                if fx["name"] == "Delay" and bpm:
                    fx["params"]["time"] = 60 / bpm / 4  # 1/4 beat
                processed = fx["instance"](processed, fs)
            return np.clip(processed, -1.0, 1.0)
        except Exception as e:
            logging.error(f"Efekt işleme hatası: {e}")
            return processed

@jit(nopython=True)
def fast_flanger(audio, fs, depth=0.005, rate=0.5, wet=0.3):
    """Hızlı flanger efekti (Numba optimizasyonlu)."""
    output = audio.copy()
    delay = depth * np.sin(2 * np.pi * rate * np.arange(len(audio)) / fs)
    indices = np.arange(len(audio)) - delay * fs
    indices = np.clip(indices, 0, len(audio) - 1).astype(np.int32)
    output += audio[indices] * wet
    return output

# AI Remix Modülü
class AIRemixer:
    """AI tabanlı remix üretimi: basit sample katmanlama ve BPM uyumu."""
    
    def __init__(self, fs=44100):
        self.fs = fs
        self.samples = {}
    
    def load_samples(self, sample_dir):
        """Sample dizininden WAV/FLAC dosyalarını yükler."""
        try:
            for file in os.listdir(sample_dir):
                if file.endswith(('.wav', '.flac')):
                    name = os.path.splitext(file)[0]
                    with sf.SoundFile(os.path.join(sample_dir, file)) as f:
                        self.samples[name] = f.read()
            logging.info(f"{len(self.samples)} sample yüklendi")
        except FileNotFoundError:
            logging.error(f"Sample dizini bulunamadı: {sample_dir}")
    
    def generate_remix(self, description, bpm=120, duration=60, key="C"):
        """Tarife göre remix üretir."""
        try:
            tokens = description.lower().split()
            style = "trap" if "trap" in tokens else "house"
            elements = []
            if "bass" in tokens:
                elements.append("kick")
                elements.append("bass")
            if "vocal" in tokens:
                elements.append("vocal")
            output = np.zeros((int(duration * self.fs), 2))
            beat_samples = int(self.fs * 60 / bpm / 4)
            for i in range(0, len(output), beat_samples):
                for elem in elements:
                    if elem in self.samples and np.random.random() > 0.3:
                        sample = self.samples[elem]
                        start = i % len(output)
                        end = min(start + len(sample), len(output))
                        output[start:end] += sample[:end-start]
            return output
        except Exception as e:
            logging.error(f"AI Remix hatası: {e}")
            return np.zeros((int(duration * self.fs), 2))

# DJ Deck
class DJDeck:
    """DJ deck: parça oynatma, cue, loop, metronom ve waveform cache."""
    
    def __init__(self, deck_id, fs=44100):
        self.deck_id = deck_id
        self.fs = fs
        self.audio = None
        self.position = 0
        self.playing = False
        self.pitch = 1.0
        self.cue_points = []
        self.hot_cues = [None] * 8
        self.loop = None
        self.volume = 1.0
        self.bpm = None
        self.waveform_cache = None
        self.metronome = False
        self.metronome_sound = np.sin(2 * np.pi * 440 * np.arange(int(fs * 0.1)) / fs) * 0.2
    
    def load_track(self, file_path):
        """Parçayı yükler ve waveform cache oluşturur."""
        tracemalloc.start()
        try:
            with contextlib.suppress(Exception):
                if file_path.endswith(('.flac', '.m4a', '.alac', '.wav')):
                    with sf.SoundFile(file_path) as f:
                        self.audio = f.read()
                        self.fs = f.samplerate
                    self.bpm, _ = librosa.beat.beat_track(y=self.audio.mean(axis=0), sr=self.fs)
                    self.waveform_cache = self.audio.mean(axis=0)[::10]
                    logging.info(f"Deck {self.deck_id}: {file_path} yüklendi, BPM: {self.bpm}")
                    return True
                raise ValueError("Desteklenmeyen format")
        except FileNotFoundError:
            logging.error(f"Dosya bulunamadı: {file_path}")
            return False
        except Exception as e:
            logging.error(f"Yükleme hatası: {e}")
            return False
        finally:
            tracemalloc.stop()
    
    def preview(self, duration=5.0):
        """Parçanın ilk 5 saniyesini döndürür."""
        if self.audio is not None:
            return self.audio[:int(duration * self.fs)]
        return None
    
    def play(self):
        self.playing = True
        logging.info(f"Deck {self.deck_id} oynatılıyor")
    
    def pause(self):
        self.playing = False
        logging.info(f"Deck {self.deck_id} durduruldu")
    
    def toggle_metronome(self):
        self.metronome = not self.metronome
        logging.info(f"Deck {self.deck_id} metronom: {'Açık' if self.metronome else 'Kapalı'}")
    
    def set_cue(self, position):
        self.cue_points.append(position)
        logging.info(f"Deck {self.deck_id} cue noktası: {position}")
    
    def set_hot_cue(self, index, position):
        self.hot_cues[index] = position
        logging.info(f"Deck {self.deck_id} hot cue {index}: {position}")
    
    def trigger_hot_cue(self, index):
        if self.hot_cues[index] is not None:
            self.position = self.hot_cues[index]
            self.play()
            logging.info(f"Deck {self.deck_id} hot cue {index} tetiklendi")
    
    def set_loop(self, start, end):
        if self.bpm:
            beat_samples = self.fs * 60 / self.bpm
            end = start + beat_samples * round((end - start) / beat_samples)  # Beat snap
        self.loop = (start, end)
        logging.info(f"Deck {self.deck_id} loop: {start}-{end}")
    
    def process(self, block_size):
        """Sinyali işler, metronom ekler."""
        if not self.playing or self.audio is None:
            return np.zeros((block_size, 2))
        start = int(self.position)
        end = min(start + int(block_size * self.pitch), len(self.audio))
        block = self.audio[start:end] * self.volume
        if self.loop:
            loop_start, loop_end = self.loop
            if self.position >= loop_end:
                self.position = loop_start
        self.position += block_size * self.pitch
        if self.metronome and self.bpm:
            beat_interval = int(self.fs * 60 / self.bpm)
            if int(self.position) % beat_interval < block_size:
                block[:len(self.metronome_sound)] += self.metronome_sound
        return librosa.resample(block, orig_sr=self.fs, target_sr=int(self.fs / self.pitch))

# DJ Mikser
class DJMixer:
    """DJ mikser: kanal gain, EQ, filtre ve crossfader."""
    
    def __init__(self, num_channels=4):
        self.channels = [
            {"gain": 1.0, "eq_low": 0.0, "eq_mid": 0.0, "eq_high": 0.0, "filter": 0.0, "volume": 1.0}
            for _ in range(num_channels)
        ]
        self.crossfader = 0.5
        self.master_volume = 1.0
        self.mic = {"gain": 1.0, "talkover": False}
    
    def process(self, deck_outputs, fs=44100):
        """Kanal sinyallerini karıştırır ve EQ/filtre uygular."""
        output = np.zeros_like(deck_outputs[0])
        try:
            for i, (deck_out, ch) in enumerate(zip(deck_outputs, self.channels)):
                eq = Neve1073Sim(low_gain=ch["eq_low"], mid_gain=ch["eq_mid"], high_gain=ch["eq_high"])
                processed = eq.process(deck_out, fs)
                if ch["filter"] != 0:
                    cutoff = 200 + (abs(ch["filter"]) * 19800)
                    b, a = butter(2, cutoff / (fs / 2), btype='low' if ch["filter"] < 0 else 'high')
                    processed = lfilter(b, a, processed)
                processed *= ch["gain"] * ch["volume"]
                if i < 2 and self.crossfader < 0.5:
                    processed *= (1 - self.crossfader * 2)
                elif i >= 2 and self.crossfader > 0.5:
                    processed *= (self.crossfader * 2 - 1)
                output += processed
            output *= self.master_volume
            if self.mic["talkover"]:
                output *= 0.7
            return output
        except Exception as e:
            logging.error(f"Mikser işleme hatası: {e}")
            return output

# Player Modülü
class MediaPlayer:
    """Medya oynatıcı: playlist yönetimi ve yerel dosya oynatma."""
    
    def __init__(self, fs=44100):
        self.fs = fs
        self.playlist = []
        self.current_track = None
        self.position = 0
        self.playing = False
        self.volume = 1.0
    
    def load_playlist(self, directory):
        """Klasörden desteklenen dosyaları playlist’e ekler."""
        try:
            self.playlist = []
            for file in os.listdir(directory):
                if file.endswith(('.flac', '.m4a', '.alac', '.wav', '.mp3')):
                    self.playlist.append(os.path.join(directory, file))
            logging.info(f"Playlist yüklendi: {len(self.playlist)} parça")
        except FileNotFoundError:
            logging.error(f"Playlist dizini bulunamadı: {directory}")
    
    def play(self, track_index=None):
        """Belirtilen parçayı oynatır."""
        try:
            if track_index is not None and 0 <= track_index < len(self.playlist):
                self.current_track = self.playlist[track_index]
                with sf.SoundFile(self.current_track) as f:
                    self.audio = f.read()
                    self.fs = f.samplerate
                self.position = 0
                self.playing = True
                logging.info(f"Player: {self.current_track} oynatılıyor")
        except FileNotFoundError:
            logging.error(f"Parça bulunamadı: {self.current_track}")
        except Exception as e:
            logging.error(f"Player oynatma hatası: {e}")
    
    def pause(self):
        self.playing = False
        logging.info("Player durduruldu")
    
    def next_track(self):
        current_idx = self.playlist.index(self.current_track) if self.current_track in self.playlist else -1
        if current_idx + 1 < len(self.playlist):
            self.play(current_idx + 1)
    
    def prev_track(self):
        current_idx = self.playlist.index(self.current_track) if self.current_track in self.playlist else -1
        if current_idx > 0:
            self.play(current_idx - 1)
    
    def process(self, block_size):
        """Sinyali oynatır ve track geçişlerini yönetir."""
        if not self.playing or self.audio is None:
            return np.zeros((block_size, 2))
        start = int(self.position)
        end = min(start + block_size, len(self.audio))
        block = self.audio[start:end] * self.volume
        self.position += block_size
        if self.position >= len(self.audio):
            self.next_track()
        return block

# Prodüksiyon Modülü
class ProductionTrack:
    """Prodüksiyon track: clip ve MIDI nota yönetimi."""
    
    def __init__(self, fs=44100):
        self.fs = fs
        self.clips = []
        self.midi_notes = []
        self.volume = 1.0
    
    def add_clip(self, audio, start_time):
        self.clips.append({"audio": audio, "start": start_time})
        logging.info(f"Clip eklendi: {start_time}s")
    
    def add_midi_note(self, note, velocity, start_time, duration):
        self.midi_notes.append({"note": note, "velocity": velocity, "start": start_time, "duration": duration})
        logging.info(f"MIDI nota eklendi: {note}")
    
    def process(self, block_size, position):
        """Track sinyalini işler."""
        output = np.zeros((block_size, 2))
        try:
            for clip in self.clips:
                clip_start = int(clip["start"] * self.fs)
                if clip_start <= position < clip_start + len(clip["audio"]):
                    start = position - clip_start
                    end = min(start + block_size, len(clip["audio"]))
                    output[:end-start] += clip["audio"][start:end] * self.volume
            return output
        except Exception as e:
            logging.error(f"Track işleme hatası: {e}")
            return output

# Ses İşleme Zinciri
class AudioChain:
    """Tüm ses modüllerini birleştirir ve gerçek zamanlı işleme yapar."""
    
    def __init__(self):
        self.device_manager = AudioDeviceManager()
        self.fs = self.device_manager.current_device["recommended_sample_rate"] if self.device_manager.current_device else 44100
        self.dj_decks = [DJDeck(i, self.fs) for i in range(4)]
        self.dj_mixer = DJMixer()
        self.effect_rack = EffectRack()
        self.ai_remixer = AIRemixer(self.fs)
        self.player = MediaPlayer(self.fs)
        self.production_tracks = [ProductionTrack(self.fs) for _ in range(8)]
        self.mic_active = False
        self.master_volume = 1.0
        self.stream = None
        self.configure()
    
    def configure(self):
        """Ses kartına ve platforma göre yapılandırma."""
        try:
            if self.stream:
                self.stream.close()
            self.device_manager.detect_device()
            success, message = self.device_manager.configure_audio()
            if not success:
                raise RuntimeError(message)
            self.fs = sd.default.samplerate
            for deck in self.dj_decks:
                deck.fs = self.fs
            self.player.fs = self.fs
            self.ai_remixer.fs = self.fs
            for track in self.production_tracks:
                track.fs = self.fs
            self.stream = sd.OutputStream(samplerate=self.fs, blocksize=self.device_manager.current_device["default_buffer_size"], channels=2)
            self.stream.start()
        except Exception as e:
            logging.error(f"Ses yapılandırma hatası: {e}")
    
    def process(self, block_size=128):
        """Tüm modüllerden sinyali karıştırır."""
        output = np.zeros((block_size, 2))
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                deck_futures = [executor.submit(deck.process, block_size) for deck in self.dj_decks]
                deck_outputs = [f.result() for f in deck_futures]
            dj_out = self.dj_mixer.process(deck_outputs, self.fs)
            bpm = self.dj_decks[0].bpm if self.dj_decks[0].bpm else None
            dj_out = self.effect_rack.process(dj_out, self.fs, bpm)
            output += dj_out
            player_out = self.player.process(block_size)
            output += player_out
            for track in self.production_tracks:
                prod_out = track.process(block_size, self.player.position if self.player.playing else 0)
                output += prod_out
            if self.mic_active:
                mic_input = np.random.randn(block_size, 2) * 0.1  # Placeholder
                output += mic_input * self.dj_mixer.mic["gain"]
            return output * self.master_volume
        except Exception as e:
            logging.error(f"Ses işleme hatası: {e}")
            return output

# Jog Wheel Widget
class JogWheelWidget(QWidget):
    """DJ jog wheel: scratch ve pitch bend."""
    
    jog_moved = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(100, 100)
        self.angle = 0
        self.last_pos = None
        self.setToolTip("Mouse ile scratch (sol-sağ) veya pitch bend (yukarı-aşağı)")
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(5, 5, 90, 90)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#003087"))
        gradient.setColorAt(1, QColor("#007bff"))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(rect)
        painter.translate(50, 50)
        painter.rotate(self.angle)
        painter.setBrush(QBrush(Qt.white))
        painter.drawRect(-3, -45, 6, 10)
    
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        if self.last_pos:
            delta_x = event.pos().x() - self.last_pos.x()
            delta_y = event.pos().y() - self.last_pos.y()
            self.angle += delta_x
            value = delta_x / 100.0 if abs(delta_x) > abs(delta_y) else delta_y / 100.0
            self.jog_moved.emit(value)
            self.last_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        self.last_pos = None
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.jog_moved.emit(delta * 0.05)
        self.update()

# GUI
class MusicaProOmnibusGUI(QMainWindow):
    """Ana kullanıcı arayüzü: DJ, prodüksiyon ve player kontrolleri."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Musica Pro Omnibus")
        self.audio_chain = AudioChain()
        self.deck_widgets = []
        self.pad_modes = ["HotCue", "Loop", "Effect", "Sample"]
        self.current_pad_mode = "HotCue"
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.waveform_scale = 1.0
        self.midi_clipboard = []
        self.init_gui()
        self.setup_shortcuts()
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()
    
    def apply_theme(self, theme="Prusya Mavisi"):
        """Arayüz temasını uygular."""
        if theme == "Açık Tema":
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; }
                QGroupBox { background-color: #ffffff; border: 1px solid #cccccc; color: black; }
                QPushButton { background-color: #007bff; color: white; }
                QPushButton:hover { background-color: #0055a4; }
                QComboBox { background-color: #ffffff; color: black; }
                QPushButton.pad { background-color: #007bff; border-radius: 6px; }
                QPushButton.pad:pressed { background-color: #ff5555; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #001a3d; }
                QGroupBox { 
                    background-color: #002b5c; 
                    border: 1px solid #004080; 
                    border-radius: 5px; 
                    margin-top: 10px; 
                    color: white; 
                }
                QGroupBox::title { 
                    subcontrol-origin: margin; 
                    subcontrol-position: top left; 
                    padding: 0 3px; 
                    color: white; 
                }
                QPushButton { 
                    background-color: #007bff; 
                    color: white; 
                    border-radius: 5px; 
                    padding: 5px; 
                    font-size: 12px; 
                }
                QPushButton:hover { background-color: #0055a4; }
                QPushButton:pressed { background-color: #003087; }
                QSlider::groove:vertical {
                    background: #002b5c;
                    width: 4px;
                    border-radius: 2px;
                }
                QSlider::handle:vertical {
                    background: #007bff;
                    height: 15px;
                    margin: 0 -3px;
                    border-radius: 7px;
                }
                QSlider::handle:vertical:hover { background: #0055a4; }
                QPushButton.pad { background-color: #007bff; border-radius: 6px; padding: 8px; }
                QPushButton.pad:pressed { background-color: #ff5555; }
                QDockWidget { background-color: #002b5c; border: 1px solid #004080; }
                QListWidget, QComboBox, QLineEdit { 
                    background-color: #003087; 
                    color: white; 
                    border: 1px solid #004080; 
                    border-radius: 3px; 
                }
            """)
    
    def setup_shortcuts(self):
        """Klavye kısayollarını tanımlar."""
        for i in range(4):
            QShortcut(QKeySequence(SHORTCUTS[f"play_deck{i+1}"]), self, lambda d=i: self.toggle_deck_play(d))
            QShortcut(QKeySequence(SHORTCUTS[f"cue_deck{i+1}"]), self, lambda d=i: self.audio_chain.dj_decks[d].set_cue(self.audio_chain.dj_decks[d].position))
        for i in range(16):
            QShortcut(QKeySequence(SHORTCUTS[f"pad_{i+1}"]), self, lambda p=i: self.trigger_pad(p))
        QShortcut(QKeySequence(SHORTCUTS["load_deck"]), self, lambda: self.load_dj_track())
        QShortcut(QKeySequence(SHORTCUTS["mic_toggle"]), self, self.toggle_mic)
        QShortcut(QKeySequence(SHORTCUTS["media_scan"]), self, self.scan_usb_sd)
        QShortcut(QKeySequence(SHORTCUTS["master_volume_up"]), self, lambda: self.adjust_master_volume(5))
        QShortcut(QKeySequence(SHORTCUTS["master_volume_down"]), self, lambda: self.adjust_master_volume(-5))
        QShortcut(QKeySequence(SHORTCUTS["crossfader_left"]), self, lambda: self.adjust_crossfader(-0.05))
        QShortcut(QKeySequence(SHORTCUTS["crossfader_right"]), self, lambda: self.adjust_crossfader(0.05))
        QShortcut(QKeySequence(SHORTCUTS["play_pause_player"]), self, self.toggle_player)
        QShortcut(QKeySequence(SHORTCUTS["next_track"]), self, self.audio_chain.player.next_track)
        QShortcut(QKeySequence(SHORTCUTS["prev_track"]), self, self.audio_chain.player.prev_track)
        QShortcut(QKeySequence(SHORTCUTS["midi_note"]), self, lambda: self.add_midi_note(60, 100, 0, 0.5))
        QShortcut(QKeySequence(SHORTCUTS["midi_copy"]), self, self.copy_midi)
        QShortcut(QKeySequence(SHORTCUTS["midi_paste"]), self, self.paste_midi)
        QShortcut(QKeySequence(SHORTCUTS["metronome_toggle"]), self, lambda: self.audio_chain.dj_decks[0].toggle_metronome())
    
    def audio_loop(self):
        """Sürekli ses işleme döngüsü."""
        block_size = self.audio_chain.device_manager.current_device["default_buffer_size"]
        try:
            with sd.OutputStream(samplerate=self.fs, blocksize=block_size, channels=2) as stream:
                while True:
                    output = self.audio_chain.process(block_size)
                    stream.write(output)
        except Exception as e:
            logging.error(f"Ses döngüsü hatası: {e}")
            QMessageBox.critical(self, "Hata", "Ses döngüsü çalıştırılamadı")
    
    def init_gui(self):
        """GUI bileşenlerini başlatır."""
        self.apply_theme()
        central = QWidget()
        main_layout = QHBoxLayout(central)
        
        # Sol Panel: DJ ve Ses Ayarları
        left_dock = QDockWidget("DJ & Ses")
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Tema Seçici
        theme_selector = QComboBox()
        theme_selector.addItems(["Prusya Mavisi", "Açık Tema"])
        theme_selector.currentTextChanged.connect(self.apply_theme)
        theme_selector.setToolTip("Arayüz temasını seç")
        left_layout.addWidget(theme_selector)
        
        # Ses Ayarları
        audio_group = QGroupBox("Ses Ayarları")
        audio_layout = QHBoxLayout(audio_group)
        self.device_label = QLabel(f"Cihaz: {self.audio_chain.device_manager.current_device['name']}")
        self.device_label.setToolTip("Algılanan ses kartı")
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([str(rate) for rate in self.audio_chain.device_manager.current_device["sample_rates"]])
        self.sample_rate_combo.setCurrentText(str(self.audio_chain.device_manager.current_device["recommended_sample_rate"]))
        self.sample_rate_combo.currentTextChanged.connect(self.update_audio_config)
        self.sample_rate_combo.setToolTip("Örnekleme hızı (Hz)")
        self.buffer_combo = QComboBox()
        buffers = list(range(self.audio_chain.device_manager.current_device["min_buffer_size"],
                            self.audio_chain.device_manager.current_device["max_buffer_size"] + 1, 32))
        self.buffer_combo.addItems([str(b) for b in buffers])
        self.buffer_combo.setCurrentText(str(self.audio_chain.device_manager.current_device["default_buffer_size"]))
        self.buffer_combo.currentTextChanged.connect(self.update_audio_config)
        self.buffer_combo.setToolTip("Buffer boyutu (düşük = az gecikme, yüksek = stabilite)")
        audio_layout.addWidget(self.device_label)
        audio_layout.addWidget(QLabel("Sample Rate:"))
        audio_layout.addWidget(self.sample_rate_combo)
        audio_layout.addWidget(QLabel("Buffer:"))
        audio_layout.addWidget(self.buffer_combo)
        left_layout.addWidget(audio_group)
        
        # DJ Paneli
        dj_group = QGroupBox("DJ Stüdyosu")
        dj_layout = QVBoxLayout(dj_group)
        deck_group = QGroupBox("Decks")
        deck_layout = QHBoxLayout(deck_group)
        self.deck_widgets = []
        for i in range(4):
            deck_widget = QWidget()
            deck_inner = QVBoxLayout(deck_widget)
            waveform = QGraphicsView()
            waveform.setFixedHeight(80)
            waveform.setStyleSheet("background-color: #002b5c; border: 1px solid #004080;")
            waveform_scene = QGraphicsScene()
            waveform.setScene(waveform_scene)
            waveform.wheelEvent = lambda e, d=i: self.zoom_waveform(e, d)
            waveform.mouseMoveEvent = lambda e, d=i: self.scroll_waveform(e, d)
            deck_inner.addWidget(waveform)
            controls = QHBoxLayout()
            play_btn = QPushButton("▶")
            play_btn.setToolTip(f"Play/Pause ({SHORTCUTS[f'play_deck{i+1}']})")
            play_btn.clicked.connect(lambda _, d=i: self.toggle_deck_play(d))
            cue_btn = QPushButton("Cue")
            cue_btn.setToolTip(f"Cue noktası ekle ({SHORTCUTS[f'cue_deck{i+1}']})")
            cue_btn.clicked.connect(lambda _, d=i: self.audio_chain.dj_decks[d].set_cue(self.audio_chain.dj_decks[d].position))
            load_btn = QPushButton("Yükle")
            load_btn.setToolTip(f"Parça yükle ({SHORTCUTS['load_deck']})")
            load_btn.clicked.connect(lambda _, d=i: self.load_dj_track(deck_id=d))
            preview_btn = QPushButton("Önizle")
            preview_btn.setToolTip("5 saniyelik önizleme")
            preview_btn.clicked.connect(lambda _, d=i: self.preview_track(d))
            pitch_slider = QSlider(Qt.Vertical)
            pitch_slider.setRange(-20, 20)
            pitch_slider.setValue(0)
            pitch_slider.setToolTip("Pitch: ±20% (Mouse tekerleği ile ince ayar)")
            pitch_slider.valueChanged.connect(lambda v, d=i: setattr(self.audio_chain.dj_decks[d], 'pitch', 1.0 + v / 100))
            pitch_slider.wheelEvent = lambda e, s=pitch_slider: s.setValue(s.value() + (1 if e.angleDelta().y() > 0 else -1))
            controls.addWidget(play_btn)
            controls.addWidget(cue_btn)
            controls.addWidget(load_btn)
            controls.addWidget(preview_btn)
            controls.addWidget(pitch_slider)
            deck_inner.addLayout(controls)
            jog = JogWheelWidget()
            jog.jog_moved.connect(lambda v, d=i: self.handle_jog(d, v))
            deck_inner.addWidget(jog)
            deck_layout.addWidget(deck_widget)
            self.deck_widgets.append({"waveform": waveform_scene, "jog": jog, "play_btn": play_btn})
        dj_layout.addWidget(deck_group)
        
        pad_group = QGroupBox("Pad Kontrolleri")
        pad_layout = QGridLayout(pad_group)
        self.pads = []
        for i in range(4):
            for j in range(4):
                pad = QPushButton()
                pad.setObjectName("pad")
                pad.setFixedSize(50, 50)
                idx = i * 4 + j
                pad.setToolTip(f"Pad {idx+1}: Hot cue, loop, efekt veya sample ({SHORTCUTS[f'pad_{idx+1}']})")
                pad.clicked.connect(lambda _, x=idx: self.trigger_pad(x))
                pad.setContextMenuPolicy(Qt.CustomContextMenu)
                pad.customContextMenuRequested.connect(lambda _, x=idx: self.show_pad_menu(x))
                pad.setAcceptDrops(True)
                pad.dragEnterEvent = lambda e: e.accept()
                pad.dropEvent = lambda e, x=idx: self.handle_drop(e, x)
                pad_layout.addWidget(pad, i, j)
                self.pads.append(pad)
        pad_mode_selector = QComboBox()
        pad_mode_selector.addItems(self.pad_modes)
        pad_mode_selector.currentTextChanged.connect(self.change_pad_mode)
        pad_mode_selector.setToolTip("Pad modunu seç: HotCue, Loop, Effect, Sample")
        pad_layout.addWidget(pad_mode_selector, 4, 0, 1, 4)
        dj_layout.addWidget(pad_group)
        
        mixer_group = QGroupBox("Mikser")
        mixer_layout = QHBoxLayout(mixer_group)
        self.channel_sliders = []
        for i in range(4):
            ch_widget = QWidget()
            ch_layout = QVBoxLayout(ch_widget)
            gain = QSlider(Qt.Vertical)
            gain.setRange(0, 200)
            gain.setValue(100)
            gain.setToolTip("Kanal gain: 0-200%")
            gain.valueChanged.connect(lambda v, d=i: self.audio_chain.dj_mixer.channels[d].update({"gain": v / 100}))
            gain.wheelEvent = lambda e, s=gain: s.setValue(s.value() + (5 if e.angleDelta().y() > 0 else -5))
            eq_low = QSlider(Qt.Vertical)
            eq_low.setRange(-15, 15)
            eq_low.setToolTip("Düşük frekanslar: ±15dB")
            eq_low.valueChanged.connect(lambda v, d=i: self.audio_chain.dj_mixer.channels[d].update({"eq_low": v}))
            eq_low.wheelEvent = lambda e, s=eq_low: s.setValue(s.value() + (1 if e.angleDelta().y() > 0 else -1))
            eq_mid = QSlider(Qt.Vertical)
            eq_mid.setRange(-15, 15)
            eq_mid.setToolTip("Orta frekanslar: ±15dB")
            eq_mid.valueChanged.connect(lambda v, d=i: self.audio_chain.dj_mixer.channels[d].update({"eq_mid": v}))
            eq_mid.wheelEvent = lambda e, s=eq_mid: s.setValue(s.value() + (1 if e.angleDelta().y() > 0 else -1))
            eq_high = QSlider(Qt.Vertical)
            eq_high.setRange(-15, 15)
            eq_high.setToolTip("Yüksek frekanslar: ±15dB")
            eq_high.valueChanged.connect(lambda v, d=i: self.audio_chain.dj_mixer.channels[d].update({"eq_high": v}))
            eq_high.wheelEvent = lambda e, s=eq_high: s.setValue(s.value() + (1 if e.angleDelta().y() > 0 else -1))
            filter = QSlider(Qt.Vertical)
            filter.setRange(-100, 100)
            filter.setToolTip("Filtre: -100 (LP) to +100 (HP)")
            filter.valueChanged.connect(lambda v, d=i: self.audio_chain.dj_mixer.channels[d].update({"filter": v / 100}))
            filter.wheelEvent = lambda e, s=filter: s.setValue(s.value() + (5 if e.angleDelta().y() > 0 else -5))
            volume = QSlider(Qt.Vertical)
            volume.setRange(0, 100)
            volume.setValue(100)
            volume.setToolTip("Kanal ses seviyesi")
            volume.valueChanged.connect(lambda v, d=i: self.audio_chain.dj_mixer.channels[d].update({"volume": v / 100}))
            volume.wheelEvent = lambda e, s=volume: s.setValue(s.value() + (5 if e.angleDelta().y() > 0 else -5))
            ch_layout.addWidget(QLabel(f"Kanal {i+1}"))
            ch_layout.addWidget(gain)
            ch_layout.addWidget(eq_low)
            ch_layout.addWidget(eq_mid)
            ch_layout.addWidget(eq_high)
            ch_layout.addWidget(filter)
            ch_layout.addWidget(volume)
            mixer_layout.addWidget(ch_widget)
            self.channel_sliders.append({"gain": gain, "eq_low": eq_low, "eq_mid": eq_mid, "eq_high": eq_high, "filter": filter, "volume": volume})
        crossfader = QSlider(Qt.Horizontal)
        crossfader.setRange(0, 100)
        crossfader.setValue(50)
        crossfader.setToolTip(f"Crossfader: Deck 1-2 ↔ Deck 3-4 ({SHORTCUTS['crossfader_left']}/{SHORTCUTS['crossfader_right']})")
        crossfader.valueChanged.connect(lambda v: setattr(self.audio_chain.dj_mixer, 'crossfader', v / 100))
        crossfader.wheelEvent = lambda e, s=crossfader: s.setValue(s.value() + (5 if e.angleDelta().y() > 0 else -5))
        master_volume = QSlider(Qt.Vertical)
        master_volume.setRange(0, 100)
        master_volume.setValue(100)
        master_volume.setToolTip(f"Master çıkış seviyesi ({SHORTCUTS['master_volume_up']}/{SHORTCUTS['master_volume_down']})")
        master_volume.valueChanged.connect(lambda v: setattr(self.audio_chain, 'master_volume', v / 100))
        master_volume.wheelEvent = lambda e, s=master_volume: s.setValue(s.value() + (5 if e.angleDelta().y() > 0 else -5))
        mixer_layout.addWidget(QLabel("Crossfader"))
        mixer_layout.addWidget(crossfader)
        mixer_layout.addWidget(QLabel("Master"))
        mixer_layout.addWidget(master_volume)
        dj_layout.addWidget(mixer_group)
        
        fx_group = QGroupBox("Efektler")
        fx_layout = QVBoxLayout(fx_group)
        fx_selector = QComboBox()
        fx_selector.addItems(EFFECTS_LIST)
        fx_selector.setToolTip("Efekt seç (F1-F12 ile hızlı seçim)")
        fx_add = QPushButton("Ekle")
        fx_add.setToolTip("Seçili efekti ekle")
        fx_add.clicked.connect(lambda: self.add_effect(fx_selector.currentText()))
        self.fx_params = QListWidget()
        self.fx_params.setToolTip("Aktif efektlerin parametrelerini düzenle (çift tık)")
        self.fx_params.itemDoubleClicked.connect(self.edit_effect_params)
        fx_controls = QHBoxLayout()
        fx_controls.addWidget(fx_selector)
        fx_controls.addWidget(fx_add)
        fx_layout.addLayout(fx_controls)
        fx_layout.addWidget(self.fx_params)
        dj_layout.addWidget(fx_group)
        
        mic_group = QGroupBox("Mikrofon")
        mic_layout = QHBoxLayout(mic_group)
        mic_toggle = QPushButton("Aç/Kapat")
        mic_toggle.setToolTip(f"Mikrofonu aç/kapat ({SHORTCUTS['mic_toggle']})")
        mic_toggle.clicked.connect(self.toggle_mic)
        mic_gain = QSlider(Qt.Horizontal)
        mic_gain.setRange(0, 200)
        mic_gain.setValue(100)
        mic_gain.setToolTip("Mikrofon gain: 0-200%")
        mic_gain.valueChanged.connect(lambda v: self.audio_chain.dj_mixer.mic.update({"gain": v / 100}))
        mic_gain.wheelEvent = lambda e, s=mic_gain: s.setValue(s.value() + (5 if e.angleDelta().y() > 0 else -5))
        mic_layout.addWidget(mic_toggle)
        mic_layout.addWidget(QLabel("Gain"))
        mic_layout.addWidget(mic_gain)
        dj_layout.addWidget(mic_group)
        
        metronome_btn = QPushButton("Metronom")
        metronome_btn.setToolTip(f"Metronomu aç/kapat ({SHORTCUTS['metronome_toggle']})")
        metronome_btn.clicked.connect(lambda: self.audio_chain.dj_decks[0].toggle_metronome())
        dj_layout.addWidget(metronome_btn)
        
        left_layout.addWidget(dj_group)
        left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)
        
        # Sağ Panel: Player ve Prodüksiyon
        right_dock = QDockWidget("Player & Prodüksiyon")
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Player Paneli
        player_group = QGroupBox("Medya Oynatıcı")
        player_layout = QVBoxLayout(player_group)
        player_controls = QHBoxLayout()
        play_btn = QPushButton("▶/⏸")
        play_btn.setToolTip(f"Play/Pause ({SHORTCUTS['play_pause_player']})")
        play_btn.clicked.connect(self.toggle_player)
        next_btn = QPushButton("▶▶")
        next_btn.setToolTip(f"Sonraki parça ({SHORTCUTS['next_track']})")
        next_btn.clicked.connect(self.audio_chain.player.next_track)
        prev_btn = QPushButton("◄◄")
        prev_btn.setToolTip(f"Önceki parça ({SHORTCUTS['prev_track']})")
        prev_btn.clicked.connect(self.audio_chain.player.prev_track)
        load_playlist_btn = QPushButton("Playlist Yükle")
        load_playlist_btn.setToolTip("Klasörden playlist oluştur")
        load_playlist_btn.clicked.connect(self.load_playlist)
        player_controls.addWidget(play_btn)
        player_controls.addWidget(prev_btn)
        player_controls.addWidget(next_btn)
        player_controls.addWidget(load_playlist_btn)
        self.player_list = QListWidget()
        self.player_list.setToolTip("Parçaları seç (yön tuşları) ve oynat (Enter)")
        self.player_list.itemDoubleClicked.connect(lambda item: self.audio_chain.player.play(self.player_list.row(item)))
        player_layout.addLayout(player_controls)
        player_layout.addWidget(self.player_list)
        right_layout.addWidget(player_group)
        
        # Prodüksiyon Paneli
        prod_group = QGroupBox("Prodüksiyon Stüdyosu")
        prod_layout = QVBoxLayout(prod_group)
        track_group = QGroupBox("Track’ler")
        track_layout = QVBoxLayout(track_group)
        self.track_widgets = []
        for i in range(8):
            track_widget = QWidget()
            track_inner = QHBoxLayout(track_widget)
            track_label = QLabel(f"Track {i+1}")
            clip_btn = QPushButton("Clip Ekle")
            clip_btn.setToolTip("Ses klibi ekle")
            clip_btn.clicked.connect(lambda _, t=i: self.add_clip(t))
            midi_btn = QPushButton("MIDI Ekle")
            midi_btn.setToolTip(f"MIDI nota ekle ({SHORTCUTS['midi_note']})")
            midi_btn.clicked.connect(lambda _, t=i: self.add_midi_note(60, 100, 0, 0.5, track_id=t))
            volume = QSlider(Qt.Horizontal)
            volume.setRange(0, 100)
            volume.setValue(100)
            volume.setToolTip("Track ses seviyesi")
            volume.valueChanged.connect(lambda v, t=i: setattr(self.audio_chain.production_tracks[t], 'volume', v / 100))
            track_inner.addWidget(track_label)
            track_inner.addWidget(clip_btn)
            track_inner.addWidget(midi_btn)
            track_inner.addWidget(volume)
            track_layout.addWidget(track_widget)
            self.track_widgets.append({"label": track_label, "volume": volume})
        prod_layout.addWidget(track_group)
        
        piano_roll = QGraphicsView()
        piano_roll.setFixedHeight(200)
        piano_roll.setStyleSheet("background-color: #002b5c; border: 1px solid #004080;")
        piano_roll_scene = QGraphicsScene()
        piano_roll.setScene(piano_roll_scene)
        piano_roll.mousePressEvent = lambda e: self.add_midi_note_from_click(e)
        prod_layout.addWidget(piano_roll)
        self.piano_roll_scene = piano_roll_scene
        
        ai_group = QGroupBox("AI Remix")
        ai_layout = QGridLayout(ai_group)
        self.ai_description = QLineEdit()
        self.ai_description.setPlaceholderText("Örn: 80 BPM trap drop, ağır bas, glitch vokaller")
        self.ai_description.setToolTip("Remix tarifini yaz")
        self.ai_bpm = QComboBox()
        self.ai_bpm.addItems(["80", "100", "120", "140"])
        self.ai_bpm.setToolTip("BPM seç")
        self.ai_key = QComboBox()
        self.ai_key.addItems(["C", "D", "E", "F", "G", "A", "B"])
        self.ai_key.setToolTip("Ton seç")
        ai_generate = QPushButton("Remix Üret")
        ai_generate.setToolTip("AI remix’i üret ve Deck 1’e yükle")
        ai_generate.clicked.connect(self.generate_ai_remix)
        ai_layout.addWidget(self.ai_description, 0, 0, 1, 2)
        ai_layout.addWidget(QLabel("BPM"), 1, 0)
        ai_layout.addWidget(self.ai_bpm, 1, 1)
        ai_layout.addWidget(QLabel("Key"), 2, 0)
        ai_layout.addWidget(self.ai_key, 2, 1)
        ai_layout.addWidget(ai_generate, 3, 0, 1, 2)
        prod_layout.addWidget(ai_group)
        right_layout.addWidget(prod_group)
        
        right_dock.setWidget(right_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)
        
        # Medya Yöneticisi
        media_dock = QDockWidget("Medya")
        media_widget = QWidget()
        media_layout = QHBoxLayout(media_widget)
        usb_btn = QPushButton("USB/SD Tara")
        usb_btn.setToolTip(f"Bağlı medya cihazlarını tara ({SHORTCUTS['media_scan']})")
        usb_btn.clicked.connect(self.scan_usb_sd)
        self.media_list = QListWidget()
        self.media_list.setToolTip("Parçaları seç (yön tuşları) ve önizle (Enter)")
        self.media_list.itemDoubleClicked.connect(self.load_dj_track)
        self.media_list.itemClicked.connect(self.preview_track)
        self.media_list.keyPressEvent = lambda e: self.handle_media_keys(e)
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Parça ara...")
        search_bar.setToolTip("Parça adına göre filtrele")
        search_bar.textChanged.connect(lambda t: self.filter_media(t))
        media_layout.addWidget(usb_btn)
        media_layout.addWidget(search_bar)
        media_layout.addWidget(self.media_list)
        media_dock.setWidget(media_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, media_dock)
        
        self.setCentralWidget(central)
        self.menuBar().addMenu("Yardım").addAction("Kısayollar", self.show_shortcuts)
        QTimer.singleShot(1000, lambda: QMessageBox.information(self, "Hoş Geldin!",
            "Musica Pro Omnibus’a hoş geldin!\n"
            f"- DJ: {SHORTCUTS['play_deck1']}-{SHORTCUTS['play_deck4']} play, {SHORTCUTS['pad_1']}-{SHORTCUTS['pad_16']} pad’ler\n"
            f"- Player: {SHORTCUTS['play_pause_player']} play/pause, {SHORTCUTS['next_track']}/{SHORTCUTS['prev_track']} track\n"
            f"- Prodüksiyon: {SHORTCUTS['midi_note']} MIDI, {SHORTCUTS['midi_copy']}/{SHORTCUTS['midi_paste']} kopyala/yapıştır\n"
            f"- Medya: {SHORTCUTS['load_deck']} yükle, {SHORTCUTS['media_scan']} tara\n"
            f"- Metronom: {SHORTCUTS['metronome_toggle']} aç/kapat\n"
            "Yardım > Kısayollar menüsüne bak!"
        ))
    
    def show_shortcuts(self):
        """Kısayol listesini gösterir."""
        msg = "\n".join([f"{k}: {v}" for k, v in SHORTCUTS.items()])
        QMessageBox.information(self, "Klavye Kısayolları", msg)
    
    def update_audio_config(self):
        """Ses ayarlarını günceller."""
        fs = int(self.sample_rate_combo.currentText())
        buffer_size = int(self.buffer_combo.currentText())
        success, message = self.audio_chain.device_manager.configure_audio(fs, buffer_size)
        if success:
            self.audio_chain.configure()
            self.status_bar.showMessage(f"Ayarlar güncellendi: {fs}Hz, {buffer_size} buffer")
        else:
            QMessageBox.critical(self, "Hata", message)
            self.status_bar.showMessage("Ses yapılandırılamadı")
    
    def toggle_deck_play(self, deck_id):
        """Deck oynatma/durdurma."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.playing:
            deck.pause()
            self.deck_widgets[deck_id]["play_btn"].setText("▶")
        else:
            deck.play()
            self.deck_widgets[deck_id]["play_btn"].setText("⏸")
        self.status_bar.showMessage(f"Deck {deck_id+1}: {'Oynatılıyor' if deck.playing else 'Durduruldu'}")
    
    def load_dj_track(self, item=None, deck_id=None):
        """DJ deck’ine parça yükler."""
        if deck_id is None:
            deck_id = 0
        file = item.text() if item else QFileDialog.getOpenFileName(self, "Parça Yükle", "", "Audio Files (*.flac *.m4a *.alac *.wav)")[0]
        if file:
            if self.audio_chain.dj_decks[deck_id].load_track(file):
                self.update_waveform(deck_id)
                self.status_bar.showMessage(f"Deck {deck_id+1}: {os.path.basename(file)} yüklendi")
            else:
                QMessageBox.critical(self, "Hata", f"Parça yüklenemedi: {file}")
    
    def preview_track(self, item=None):
        """Parçayı önizler."""
        if isinstance(item, int):
            deck_id = item
            preview = self.audio_chain.dj_decks[deck_id].preview()
            if preview is not None:
                sd.play(preview, self.audio_chain.fs)
                self.status_bar.showMessage(f"Deck {deck_id+1} önizlemesi oynatılıyor")
        elif item:
            file = item.text()
            try:
                with sf.SoundFile(file) as f:
                    audio = f.read(int(5 * f.samplerate))
                sd.play(audio, self.audio_chain.fs)
                self.status_bar.showMessage(f"Önizleme: {os.path.basename(file)}")
            except FileNotFoundError:
                QMessageBox.critical(self, "Hata", f"Önizleme dosyası bulunamadı: {file}")
    
    def zoom_waveform(self, event, deck_id):
        """Waveform’u yakınlaştırır/uzaklaştırır."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.waveform_cache is not None:
            delta = event.angleDelta().y() / 120
            self.waveform_scale = getattr(self, 'waveform_scale', 1.0) * (1.1 if delta > 0 else 0.9)
            self.waveform_scale = max(0.5, min(self.waveform_scale, 5.0))
            self.update_waveform(deck_id)
    
    def scroll_waveform(self, event, deck_id):
        """Waveform’da gezinir."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.waveform_cache is not None:
            delta_x = event.pos().x() - getattr(self, f'last_x_{deck_id}', event.pos().x())
            deck.position += delta_x * 100
            if deck.position < 0:
                deck.position = 0
            self.update_waveform(deck_id)
            setattr(self, f'last_x_{deck_id}', event.pos().x())
    
    def update_waveform(self, deck_id):
        """Waveform’u günceller, çalma pozisyonu ve beat marker ekler."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.waveform_cache is not None:
            scene = self.deck_widgets[deck_id]["waveform"]
            scene.clear()
            scale = getattr(self, 'waveform_scale', 1.0)
            samples = deck.waveform_cache[::int(10 / scale)]
            path = QPainterPath()
            for i, v in enumerate(samples[:500]):
                path.lineTo(i * 200 / 500, -v * 40 / scale)
            scene.addPath(path, QPen(QColor("#007bff")))
            # Çalma pozisyonu
            pos = deck.position / len(deck.audio) * 200 if deck.audio is not None else 0
            scene.addLine(pos, -40, pos, 40, QPen(QColor("#ff5555"), 2))
            # Beat marker’lar
            if deck.bpm:
                beat_interval = self.audio_chain.fs * 60 / deck.bpm
                for i in range(0, len(deck.audio), int(beat_interval)):
                    x = i / len(deck.audio) * 200
                    scene.addLine(x, -20, x, 20, QPen(QColor("#ffffff"), 1))
    
    def handle_jog(self, deck_id, value):
        """Jog wheel hareketini işler."""
        deck = self.audio_chain.dj_decks[deck_id]
        deck.position += value * 100
        if deck.position < 0:
            deck.position = 0
        self.update_waveform(deck_id)
        logging.info(f"Deck {deck_id} jog: {value}")
    
    def change_pad_mode(self, mode):
        """Pad modunu değiştirir."""
        self.current_pad_mode = mode
        for pad in self.pads:
            pad.setStyleSheet("background-color: #007bff; border-radius: 6px;")
        self.status_bar.showMessage(f"Pad modu: {mode}")
    
    def show_pad_menu(self, index):
        """Pad sağ tıklama menüsünü gösterir."""
        menu = QMenu(self)
        for mode in self.pad_modes:
            menu.addAction(mode, lambda m=mode: self.change_pad_mode(m))
        menu.exec_(self.pads[index].mapToGlobal(self.pads[index].rect().bottomLeft()))
    
    def trigger_pad(self, index):
        """Pad tetikleme ve animasyon."""
        deck_id = index // 4
        pad_id = index % 4
        deck = self.audio_chain.dj_decks[deck_id]
        try:
            if self.current_pad_mode == "HotCue":
                deck.trigger_hot_cue(pad_id)
            elif self.current_pad_mode == "Loop":
                deck.set_loop(deck.position, deck.position + 10000)
            elif self.current_pad_mode == "Effect":
                fx = EFFECTS_LIST[pad_id % len(EFFECTS_LIST)]
                self.audio_chain.effect_rack.add_effect(fx)
                self.fx_params.addItem(fx)
            elif self.current_pad_mode == "Sample":
                logging.info(f"Sample {pad_id} tetiklendi")
            anim = QPropertyAnimation(self.pads[index], b"styleSheet")
            anim.setStartValue("background-color: #ff5555; border-radius: 6px;")
            anim.setEndValue("background-color: #007bff; border-radius: 6px;")
            anim.setDuration(100)
            anim.start()
            self.status_bar.showMessage(f"Pad {index+1}: {self.current_pad_mode} tetiklendi")
        except Exception as e:
            logging.error(f"Pad tetikleme hatası: {e}")
            QMessageBox.critical(self, "Hata", "Pad işlemi başarısız")
    
    def handle_drop(self, event, index):
        """Pad’e sürüklenen dosyayı yükler."""
        deck_id = index // 4
        file = event.mimeData().urls()[0].toLocalFile()
        if file.endswith(('.flac', '.m4a', '.alac', '.wav')):
            self.load_dj_track(file, deck_id)
    
    def add_effect(self, fx_name):
        """Efekt ekler."""
        try:
            self.audio_chain.effect_rack.add_effect(fx_name)
            self.fx_params.addItem(fx_name)
            self.status_bar.showMessage(f"Efekt eklendi: {fx_name}")
        except Exception as e:
            logging.error(f"Efekt ekleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Efekt eklenemedi: {fx_name}")
    
    def edit_effect_params(self, item):
        """Efekt parametrelerini düzenler."""
        fx_name = item.text()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{fx_name} Parametreleri")
        layout = QVBoxLayout(dialog)
        params = {}
        if fx_name == "Reverb":
            params = {"room_size": QSlider(Qt.Horizontal), "damping": QSlider(Qt.Horizontal), "wet": QSlider(Qt.Horizontal)}
        elif fx_name == "Compressor":
            params = {"threshold": QSlider(Qt.Horizontal), "ratio": QSlider(Qt.Horizontal), "wet": QSlider(Qt.Horizontal)}
        elif fx_name == "Flanger":
            params = {"depth": QSlider(Qt.Horizontal), "rate": QSlider(Qt.Horizontal), "wet": QSlider(Qt.Horizontal)}
        for name, slider in params.items():
            slider.setRange(0, 100)
            slider.setValue(30)
            slider.setToolTip(f"{name.capitalize()}: 0-100%")
            layout.addWidget(QLabel(name))
            layout.addWidget(slider)
        ok_btn = QPushButton("Tamam")
        ok_btn.clicked.connect(dialog.accept)
        layout.addWidget(ok_btn)
        if dialog.exec_():
            new_params = {name: slider.value() / 100 for name, slider in params.items()}
            try:
                for fx in self.audio_chain.effect_rack.active_effects:
                    if fx["name"] == fx_name:
                        fx["params"] = new_params
                        self.audio_chain.effect_rack.add_effect(fx_name, new_params)
                self.status_bar.showMessage(f"{fx_name} parametreleri güncellendi")
            except Exception as e:
                logging.error(f"Efekt parametre hatası: {e}")
                QMessageBox.critical(self, "Hata", "Efekt parametreleri güncellenemedi")
    
    def toggle_mic(self):
        """Mikrofonu açar/kapatır."""
        self.audio_chain.mic_active = not self.audio_chain.mic_active
        self.status_bar.showMessage(f"Mikrofon: {'Açık' if self.audio_chain.mic_active else 'Kapalı'}")
    
    def scan_usb_sd(self):
        """USB/SD cihazlarını tarar."""
        path = QFileDialog.getExistingDirectory(self, "USB/SD Seç")
        if path:
            try:
                self.audio_chain.player.load_playlist(path)
                self.media_list.clear()
                for track in self.audio_chain.player.playlist:
                    self.media_list.addItem(track)
                self.status_bar.showMessage(f"Medya tarandı: {len(self.audio_chain.player.playlist)} parça bulundu")
            except Exception as e:
                logging.error(f"Medya tarama hatası: {e}")
                QMessageBox.critical(self, "Hata", "Medya tarama başarısız")
    
    def filter_media(self, text):
        """Medya listesini filtreler."""
        self.media_list.clear()
        for track in self.audio_chain.player.playlist:
            if text.lower() in os.path.basename(track).lower():
                self.media_list.addItem(track)
    
    def handle_media_keys(self, event):
        """Medya listesindeki klavye olaylarını işler."""
        if event.key() == Qt.Key_Return:
            item = self.media_list.currentItem()
            if item:
                self.preview_track(item)
        elif event.key() in (Qt.Key_Up, Qt.Key_Down):
            QListWidget.keyPressEvent(self.media_list, event)
    
    def generate_ai_remix(self):
        """AI remix üretir."""
        description = self.ai_description.text()
        bpm = int(self.ai_bpm.currentText())
        key = self.ai_key.currentText()
        if description:
            try:
                remix = self.audio_chain.ai_remixer.generate_remix(description, bpm=bpm, key=key)
                sf.write("remix.wav", remix, self.audio_chain.fs)
                self.audio_chain.dj_decks[0].load_track("remix.wav")
                self.update_waveform(0)
                self.status_bar.showMessage(f"AI Remix üretildi: {description}")
            except Exception as e:
                logging.error(f"AI Remix üretme hatası: {e}")
                QMessageBox.critical(self, "Hata", "Remix üretilemedi")
    
    def adjust_master_volume(self, delta):
        """Master ses seviyesini ayarlar."""
        new_volume = max(0, min(100, int(self.audio_chain.master_volume * 100 + delta)))
        self.audio_chain.master_volume = new_volume / 100
        self.status_bar.showMessage(f"Master volume: {new_volume}%")
    
    def adjust_crossfader(self, delta):
        """Crossfader’ı ayarlar."""
        new_value = max(0, min(1, self.audio_chain.dj_mixer.crossfader + delta))
        self.audio_chain.dj_mixer.crossfader = new_value
        self.status_bar.showMessage(f"Crossfader: {int(new_value * 100)}%")
    
    def toggle_player(self):
        """Player’ı oynatır/durdurur."""
        if self.audio_chain.player.playing:
            self.audio_chain.player.pause()
        else:
            if not self.audio_chain.player.current_track and self.audio_chain.player.playlist:
                self.audio_chain.player.play(0)
            else:
                self.audio_chain.player.play()
        self.status_bar.showMessage(f"Player: {'Oynatılıyor' if self.audio_chain.player.playing else 'Durduruldu'}")
    
    def load_playlist(self):
        """Playlist yükler."""
        path = QFileDialog.getExistingDirectory(self, "Playlist Klasörü")
        if path:
            try:
                self.audio_chain.player.load_playlist(path)
                self.player_list.clear()
                for track in self.audio_chain.player.playlist:
                    self.player_list.addItem(os.path.basename(track))
                self.status_bar.showMessage(f"Playlist yüklendi: {len(self.audio_chain.player.playlist)} parça")
            except Exception as e:
                logging.error(f"Playlist yükleme hatası: {e}")
                QMessageBox.critical(self, "Hata", "Playlist yüklenemedi")
    
    def add_clip(self, track_id):
        """Prodüksiyon track’ine clip ekler."""
        file = QFileDialog.getOpenFileName(self, "Clip Yükle", "", "Audio Files (*.wav *.flac)")[0]
        if file:
            try:
                with sf.SoundFile(file) as f:
                    audio = f.read()
                self.audio_chain.production_tracks[track_id].add_clip(audio, 0)
                self.status_bar.showMessage(f"Track {track_id+1}: Clip eklendi")
            except FileNotFoundError:
                logging.error(f"Clip dosyası bulunamadı: {file}")
                QMessageBox.critical(self, "Hata", "Clip yüklenemedi")
    
    def add_midi_note(self, note, velocity, start_time, duration, track_id=0):
        """MIDI nota ekler."""
        self.audio_chain.production_tracks[track_id].add_midi_note(note, velocity, start_time, duration)
        self.update_piano_roll()
        self.status_bar.showMessage(f"Track {track_id+1}: MIDI nota eklendi (Note: {note})")
    
    def add_midi_note_from_click(self, event):
        """Piano roll’a tıklama ile MIDI nota ekler."""
        pos = event
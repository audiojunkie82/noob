import sys
import time
import threading
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from pedalboard import Reverb, Compressor, Flanger
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QGroupBox, QPushButton, QSlider, QLabel, QComboBox, QFileDialog,
                             QLineEdit, QDialog, QFormLayout, QGraphicsView, QGraphicsScene,
                             QMessageBox, QMenu, QProgressDialog, QGraphicsRectItem, QGraphicsLineItem)
from PyQt5.QtGui import QPainterPath, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sabitler
EFFECTS_LIST = ["Reverb", "Compressor", "Flanger"]
SHORTCUTS = {
    "play_pause": "Space",
    "crossfader_left": "Left",
    "crossfader_right": "Right",
    "volume_up": "+",
    "volume_down": "-"
}

class EffectRack:
    def __init__(self):
        self.active_effects = []

    def add_effect(self, name, params=None):
        """Efekt ekler."""
        try:
            params = params or {}
            if name == "Reverb":
                effect = Reverb(
                    room_size=params.get("room_size", 0.5),
                    damping=params.get("damping", 0.5),
                    wet_level=params.get("wet", 0.3)
                )
            elif name == "Compressor":
                effect = Compressor(
                    threshold_db=params.get("threshold", -24.0),
                    ratio=params.get("ratio", 4.0),
                    mix=params.get("wet", 0.3)
                )
            elif name == "Flanger":
                effect = Flanger(
                    depth=params.get("depth", 0.005),
                    rate_hz=params.get("rate", 0.5),
                    mix=params.get("wet", 0.3)
                )
            else:
                raise ValueError(f"Bilinmeyen efekt: {name}")
            self.active_effects = [fx for fx in self.active_effects if fx["name"] != name]
            self.active_effects.append({"name": name, "instance": effect, "params": params})
            logging.info(f"Efekt eklendi: {name}")
        except Exception as e:
            logging.error(f"Efekt ekleme hatası: {e}")
            raise

    def process(self, audio, fs):
        """Efekt zincirini uygular."""
        try:
            processed = audio.copy()
            for fx in self.active_effects:
                processed = fx["instance"](processed, fs)
            return np.clip(processed, -1.0, 1.0)
        except Exception as e:
            logging.error(f"Efekt işleme hatası: {e}")
            return audio

class DJDeck:
    def __init__(self, deck_id, fs=44100):
        self.deck_id = deck_id
        self.fs = fs
        self.audio = None
        self.waveform_cache = None
        self.position = 0
        self.playing = False
        self.bpm = None
        self.loop = None
        self.cue_points = []
        self.effect_rack = EffectRack()

    def load_track(self, file_path):
        """Parçayı yükler."""
        try:
            if file_path.endswith(('.flac', '.m4a', '.alac', '.wav')):
                with sf.SoundFile(file_path) as f:
                    self.audio = f.read()
                    self.fs = f.samplerate
                self.bpm, _ = librosa.beat.beat_track(y=self.audio.mean(axis=0), sr=self.fs)
                self.waveform_cache = self.audio.mean(axis=0)[::10]
                logging.info(f"Deck {self.deck_id}: {file_path} yüklendi, BPM: {self.bpm}")
                return True, ""
            return False, f"Desteklenmeyen format: {os.path.splitext(file_path)[1]}. WAV, FLAC, M4A veya ALAC kullanın."
        except FileNotFoundError:
            return False, f"Dosya bulunamadı: {file_path}. Lütfen dosya yolunu kontrol edin."
        except Exception as e:
            return False, f"Parça yüklenemedi: {str(e)}. Lütfen dosyayı kontrol edin."

    def play(self):
        self.playing = True
        logging.info(f"Deck {self.deck_id}: Oynatılıyor")

    def pause(self):
        self.playing = False
        logging.info(f"Deck {self.deck_id}: Durduruldu")

    def set_loop(self, start, beat_count=8):
        """BPM’e göre loop oluşturur."""
        try:
            if self.bpm:
                beat_duration = self.fs * 60 / self.bpm
                end = start + beat_duration * beat_count
                self.loop = (start, end)
                loop_seconds = (end - start) / self.fs
                logging.info(f"Deck {self.deck_id} loop: {start}-{end} ({loop_seconds:.2f}s, {beat_count} beat)")
            else:
                self.loop = (start, start + 10000)
                logging.warning(f"Deck {self.deck_id}: BPM algılanmadı, varsayılan loop")
            return self.loop
        except Exception as e:
            logging.error(f"Loop oluşturma hatası: {e}")
            return None

    def process(self, block_size):
        """Sinyali işler."""
        try:
            output = np.zeros((block_size, 2))
            if self.audio is None or not self.playing:
                return output
            if self.loop:
                loop_start, loop_end = self.loop
                if self.position >= loop_end:
                    self.position = loop_start
            start = int(self.position)
            end = min(start + block_size, len(self.audio))
            output[:end-start] = self.audio[start:end]
            self.position += block_size
            if self.position >= len(self.audio) and not self.loop:
                self.playing = False
                self.position = 0
            output = self.effect_rack.process(output, self.fs)
            return output
        except Exception as e:
            logging.error(f"Deck {self.deck_id} işleme hatası: {e}")
            return np.zeros((block_size, 2))

class DJMixer:
    def __init__(self):
        self.crossfader = 0.5

    def process(self, deck_outputs, fs):
        """Deck sinyallerini karıştırır."""
        try:
            output = np.zeros_like(deck_outputs[0])
            for i, deck_out in enumerate(deck_outputs):
                gain = 1.0
                if i < 2 and self.crossfader < 0.5:
                    gain = 1 - self.crossfader * 2
                elif i >= 2 and self.crossfader > 0.5:
                    gain = self.crossfader * 2 - 1
                output += deck_out * gain
            return np.clip(output, -1.0, 1.0)
        except Exception as e:
            logging.error(f"Mixer hatası: {e}")
            return np.zeros_like(deck_outputs[0])

class ProductionTrack:
    def __init__(self, fs=44100):
        self.fs = fs
        self.clips = []
        self.midi_notes = []
        self.volume = 1.0
        self.effect_rack = EffectRack()

    def add_clip(self, audio, start_time):
        """Clip ekler."""
        if start_time < 0:
            raise ValueError("Başlangıç zamanı negatif olamaz")
        self.clips.append({"audio": audio, "start": start_time})
        logging.info(f"Clip eklendi: {start_time}s")

    def add_midi_note(self, note, velocity, start, duration):
        """MIDI nota ekler."""
        if not (0 <= note <= 127 and 0 <= velocity <= 127 and start >= 0 and duration > 0):
            raise ValueError("Geçersiz MIDI parametreleri")
        self.midi_notes.append({"note": note, "velocity": velocity, "start": start, "duration": duration})
        logging.info(f"MIDI nota eklendi: {note}")

    def add_effect(self, fx_name, params=None):
        """Track’e efekt ekler."""
        self.effect_rack.add_effect(fx_name, params)
        logging.info(f"Track efekti eklendi: {fx_name}")

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
            output = self.effect_rack.process(output, self.fs)
            return output
        except Exception as e:
            logging.error(f"Track işleme hatası: {e}")
            return output

class AIRemixer:
    def __init__(self, fs=44100):
        self.fs = fs
        self.samples = {
            "kick": np.sin(2 * np.pi * 100 * np.arange(int(fs * 0.1)) / fs) * 0.5,
            "bass": np.sin(2 * np.pi * 60 * np.arange(int(fs * 0.2)) / fs) * 0.4,
            "vocal": np.random.randn(int(fs * 0.5)) * 0.1
        }

    def generate_remix(self, description, bpm=120, duration=60, key="C"):
        """Remix üretir."""
        try:
            tokens = description.lower().split()
            style = "trap" if "trap" in tokens else "house"
            elements = []
            if "bass" in tokens:
                elements.extend(["kick", "bass"])
            if "vocal" in tokens:
                elements.append("vocal")
            output = np.zeros((int(duration * self.fs), 2))
            beat_samples = int(self.fs * 60 / bpm / 4)
            chunk_size = int(self.fs * 10)
            for chunk_start in range(0, len(output), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(output))
                for i in range(chunk_start, chunk_end, beat_samples):
                    for elem in elements:
                        if elem in self.samples and np.random.random() > 0.3:
                            sample = self.samples[elem]
                            start = i % len(output)
                            end = min(start + len(sample), len(output))
                            output[start:end] += np.repeat(sample[:end-start, np.newaxis], 2, axis=1)
            return np.clip(output, -1.0, 1.0)
        except Exception as e:
            logging.error(f"AI Remix hatası: {e}")
            return np.zeros((int(duration * self.fs), 2))

class AudioChain:
    def __init__(self, fs=44100, block_size=128):
        self.fs = fs
        self.block_size = block_size
        self.dj_decks = [DJDeck(i, fs) for i in range(4)]
        self.dj_mixer = DJMixer()
        self.production_tracks = [ProductionTrack(fs) for _ in range(8)]
        self.player = None
        self.ai_remixer = AIRemixer(fs)
        self.master_volume = 1.0

    def process(self):
        """Sinyali işler."""
        try:
            deck_outputs = [deck.process(self.block_size) for deck in self.dj_decks]
            dj_out = self.dj_mixer.process(deck_outputs, self.fs)
            prod_out = np.zeros_like(dj_out)
            for track in self.production_tracks:
                prod_out += track.process(self.block_size, 0)
            output = (dj_out + prod_out) * self.master_volume
            return np.clip(output, -1.0, 1.0)
        except Exception as e:
            logging.error(f"AudioChain hatası: {e}")
            return np.zeros((self.block_size, 2))

class MusicaProOmnibusGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_chain = AudioChain()
        self.deck_widgets = []
        self.track_widgets = []
        self.pads = []
        self.pad_modes = ["HotCue", "Loop", "Effect", "Sample"]
        self.current_pad_mode = "HotCue"
        self.waveform_scale = 1.0
        self.init_gui()
        self.init_audio()

    def init_gui(self):
        """GUI’yi başlatır."""
        self.setWindowTitle("Musica Pro Omnibus")
        self.setStyleSheet("QMainWindow { background-color: #001a3d; } QGroupBox { background-color: #002b5c; border: 1px solid #004080; color: white; } QPushButton { background-color: #007bff; color: white; } QPushButton.pad { background-color: #007bff; border-radius: 6px; }")
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Tema seçici
        theme_selector = QComboBox()
        theme_selector.addItems(["Prusya Mavisi", "Açık Tema"])
        theme_selector.currentTextChanged.connect(self.apply_theme)
        layout.addWidget(theme_selector)

        # Deck’ler
        decks_group = QGroupBox("DJ Deck’leri")
        decks_layout = QHBoxLayout(decks_group)
        for i in range(4):
            deck_widget = QWidget()
            deck_inner = QVBoxLayout(deck_widget)
            load_btn = QPushButton("Yükle")
            load_btn.clicked.connect(lambda _, d=i: self.load_dj_track(deck_id=d))
            play_btn = QPushButton("▶")
            play_btn.clicked.connect(lambda _, d=i: self.toggle_deck_play(d))
            waveform = QGraphicsView()
            waveform.setFixedHeight(80)
            waveform.setStyleSheet("background-color: #002b5c; border: 1px solid #004080;")
            waveform_scene = QGraphicsScene()
            waveform.setScene(waveform_scene)
            waveform.wheelEvent = lambda e, d=i: self.zoom_waveform(e, d)
            waveform.mousePressEvent = lambda e, d=i: self.start_loop_drag(e, d)
            waveform.mouseMoveEvent = lambda e, d=i: self.update_loop_drag(e, d)
            waveform.mouseReleaseEvent = lambda e, d=i: self.end_loop_drag(e, d)
            deck_inner.addWidget(load_btn)
            deck_inner.addWidget(play_btn)
            deck_inner.addWidget(waveform)
            decks_layout.addWidget(deck_widget)
            self.deck_widgets.append({"waveform": waveform_scene, "play_btn": play_btn, "loop_start": None})
        layout.addWidget(decks_group)

        # Crossfader ve Master Volume
        controls_group = QGroupBox("Kontroller")
        controls_layout = QHBoxLayout(controls_group)
        crossfader = QSlider(Qt.Horizontal)
        crossfader.setRange(0, 100)
        crossfader.setValue(50)
        crossfader.valueChanged.connect(lambda v: setattr(self.audio_chain.dj_mixer, 'crossfader', v / 100))
        master_volume = QSlider(Qt.Horizontal)
        master_volume.setRange(0, 100)
        master_volume.setValue(100)
        master_volume.valueChanged.connect(lambda v: setattr(self.audio_chain, 'master_volume', v / 100))
        controls_layout.addWidget(QLabel("Crossfader"))
        controls_layout.addWidget(crossfader)
        controls_layout.addWidget(QLabel("Master Volume"))
        controls_layout.addWidget(master_volume)
        layout.addWidget(controls_group)

        # Efektler
        fx_group = QGroupBox("Efektler")
        fx_layout = QHBoxLayout(fx_group)
        fx_selector = QComboBox()
        fx_selector.addItems(EFFECTS_LIST)
        fx_add_btn = QPushButton("Ekle")
        fx_add_btn.clicked.connect(lambda: self.add_effect(fx_selector.currentText()))
        fx_params = QComboBox()
        fx_params.itemDoubleClicked.connect(self.edit_effect_params)
        fx_layout.addWidget(fx_selector)
        fx_layout.addWidget(fx_add_btn)
        fx_layout.addWidget(fx_params)
        self.fx_params = fx_params
        layout.addWidget(fx_group)

        # Pad’ler
        pad_group = QGroupBox("Pad Kontrolleri")
        pad_layout = QGridLayout(pad_group)
        for i in range(4):
            for j in range(4):
                pad = QPushButton()
                pad.setObjectName("pad")
                pad.setFixedSize(50, 50)
                idx = i * 4 + j
                pad.clicked.connect(lambda _, x=idx: self.trigger_pad(x))
                pad.setContextMenuPolicy(Qt.CustomContextMenu)
                pad.customContextMenuRequested.connect(lambda _, x=idx: self.show_pad_menu(x))
                pad_layout.addWidget(pad, i, j)
                self.pads.append(pad)
        mode_layout = QHBoxLayout()
        for mode in self.pad_modes:
            btn = QPushButton(mode)
            btn.setStyleSheet("background-color: #007bff; border-radius: 5px;")
            btn.clicked.connect(lambda _, m=mode: self.change_pad_mode(m))
            mode_layout.addWidget(btn)
        pad_layout.addLayout(mode_layout, 4, 0, 1, 4)
        layout.addWidget(pad_group)

        # AI Remix
        ai_group = QGroupBox("AI Remix")
        ai_layout = QFormLayout(ai_group)
        self.ai_description = QLineEdit()
        self.ai_bpm = QComboBox()
        self.ai_bpm.addItems([str(i) for i in range(60, 181)])
        self.ai_key = QComboBox()
        self.ai_key.addItems(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        ai_generate_btn = QPushButton("Remix Üret")
        ai_generate_btn.clicked.connect(self.generate_ai_remix)
        ai_layout.addRow("Açıklama", self.ai_description)
        ai_layout.addRow("BPM", self.ai_bpm)
        ai_layout.addRow("Key", self.ai_key)
        ai_layout.addRow(ai_generate_btn)
        layout.addWidget(ai_group)

        # Prodüksiyon
        prod_group = QGroupBox("Prodüksiyon")
        track_layout = QVBoxLayout(prod_group)
        for i in range(8):
            track_widget = QWidget()
            track_inner = QHBoxLayout(track_widget)
            track_label = QLabel(f"Track {i+1}")
            clip_btn = QPushButton("Clip Ekle")
            clip_btn.clicked.connect(lambda _, t=i: self.add_clip(t))
            start_time_input = QLineEdit("0.0")
            start_time_input.setFixedWidth(50)
            start_time_input.setToolTip("Clip başlangıç zamanı (saniye)")
            midi_btn = QPushButton("MIDI Ekle")
            midi_btn.clicked.connect(lambda _, t=i: self.add_midi_note(60, 100, 0, 0.5, track_id=t))
            fx_btn = QPushButton("Efekt Ekle")
            fx_btn.clicked.connect(lambda _, t=i: self.add_track_effect(t))
            volume = QSlider(Qt.Horizontal)
            volume.setRange(0, 100)
            volume.setValue(100)
            volume.valueChanged.connect(lambda v, t=i: setattr(self.audio_chain.production_tracks[t], 'volume', v / 100))
            preview_btn = QPushButton("Önizle")
            preview_btn.clicked.connect(lambda _, t=i: self.preview_track(t))
            export_btn = QPushButton("Playlist’e Ekle")
            export_btn.clicked.connect(lambda _, t=i: self.export_to_playlist(t))
            track_inner.addWidget(track_label)
            track_inner.addWidget(clip_btn)
            track_inner.addWidget(start_time_input)
            track_inner.addWidget(midi_btn)
            track_inner.addWidget(fx_btn)
            track_inner.addWidget(volume)
            track_inner.addWidget(preview_btn)
            track_inner.addWidget(export_btn)
            track_layout.addWidget(track_widget)
            self.track_widgets.append({"label": track_label, "volume": volume, "start_time": start_time_input})
        layout.addWidget(prod_group)

        # Piano Roll
        piano_roll_group = QGroupBox("Piano Roll")
        piano_roll_layout = QVBoxLayout(piano_roll_group)
        piano_roll = QGraphicsView()
        piano_roll.setFixedHeight(200)
        piano_roll_scene = QGraphicsScene()
        piano_roll.setScene(piano_roll_scene)
        piano_roll.mousePressEvent = lambda e, t=0: self.add_midi_note_from_click(e, t)
        piano_roll.mouseDoubleClickEvent = lambda e: self.edit_midi_note(e)
        self.piano_roll_scene = piano_roll_scene
        piano_roll_layout.addWidget(piano_roll)
        layout.addWidget(piano_roll_group)

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Musica Pro Omnibus hazır!")

        # Hoş geldiniz mesajı
        QTimer.singleShot(1000, self.show_welcome_message)

    def init_audio(self):
        """Ses sistemini başlatır."""
        try:
            self.stream = sd.OutputStream(
                samplerate=self.audio_chain.fs,
                blocksize=self.audio_chain.block_size,
                channels=2,
                callback=lambda outdata, frames, time, status: self.audio_callback(outdata, frames, time, status)
            )
            self.stream.start()
        except Exception as e:
            logging.error(f"Ses başlatma hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Ses sistemi başlatılamadı: {str(e)}")

    def audio_callback(self, outdata, frames, time, status):
        """Ses geri çağrısı."""
        if status:
            logging.warning(f"Ses hatası: {status}")
        outdata[:] = self.audio_chain.process()

    def show_welcome_message(self):
        """Hoş geldiniz mesajı."""
        QMessageBox.information(self, "Hoş Geldiniz", "Musica Pro Omnibus’a hoş geldin!\n"
                                "- DJ: Q-R play, 1-4 pad’ler\n"
                                "- Player: Space play/pause\n"
                                "- Prodüksiyon: Q MIDI, C/V kopyala/yapıştır\n"
                                "- Metronom: M aç/kapat")

    def apply_theme(self, theme):
        """Temayı uygular."""
        if theme == "Prusya Mavisi":
            self.setStyleSheet("QMainWindow { background-color: #001a3d; } QGroupBox { background-color: #002b5c; border: 1px solid #004080; color: white; } QPushButton { background-color: #007bff; color: white; } QPushButton.pad { background-color: #007bff; border-radius: 6px; }")
        else:
            self.setStyleSheet("QMainWindow { background-color: #ffffff; } QGroupBox { background-color: #f0f0f0; border: 1px solid #cccccc; color: black; } QPushButton { background-color: #007bff; color: white; }")
        self.status_bar.showMessage(f"Tema: {theme}")

    def load_dj_track(self, item=None, deck_id=None):
        """DJ deck’ine parça yükler."""
        if deck_id is None:
            deck_id = 0
        file = item.text() if item else QFileDialog.getOpenFileName(self, "Parça Yükle", "", "Audio Files (*.flac *.m4a *.alac *.wav)")[0]
        if file:
            success, message = self.audio_chain.dj_decks[deck_id].load_track(file)
            if success:
                self.update_waveform(deck_id)
                self.status_bar.showMessage(f"Deck {deck_id+1}: {os.path.basename(file)} yüklendi, BPM: {self.audio_chain.dj_decks[deck_id].bpm}")
            else:
                QMessageBox.critical(self, "Hata", message)
                self.status_bar.showMessage(f"Yükleme başarısız: {os.path.basename(file)}")

    def toggle_deck_play(self, deck_id):
        """Deck oynatmayı değiştirir."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.playing:
            deck.pause()
            self.deck_widgets[deck_id]["play_btn"].setText("▶")
        else:
            deck.play()
            self.deck_widgets[deck_id]["play_btn"].setText("⏸")
        self.status_bar.showMessage(f"Deck {deck_id+1}: {'Oynatılıyor' if deck.playing else 'Durduruldu'}")

    def update_waveform(self, deck_id):
        """Waveform’u günceller."""
        deck = self.audio_chain.dj_decks[deck_id]
        scene = self.deck_widgets[deck_id]["waveform"]
        scene.clear()
        scale = getattr(self, 'waveform_scale', 1.0)
        if deck.waveform_cache is not None:
            samples = deck.waveform_cache[::int(10 / scale)]
            path = QPainterPath()
            for i, v in enumerate(samples[:500]):
                path.lineTo(i * 200 / 500, -v * 40 / scale)
            scene.addPath(path, QPen(QColor("#007bff")))
            pos = deck.position / len(deck.audio) * 200 if deck.audio is not None else 0
            scene.addLine(pos, -40, pos, 40, QPen(QColor("#ff5555"), 2))
            if deck.bpm:
                beat_interval = self.audio_chain.fs * 60 / deck.bpm
                for i in range(0, len(deck.audio), int(beat_interval)):
                    x = i / len(deck.audio) * 200
                    scene.addLine(x, -20, x, 20, QPen(QColor("#ffffff"), 1))
            if deck.loop:
                start, end = deck.loop
                start_pos = start / len(deck.audio) * 200
                end_pos = end / len(deck.audio) * 200
                scene.addRect(start_pos, -40, end_pos - start_pos, 80, QPen(Qt.NoPen), QBrush(QColor(85, 255, 85, 50)))
                scene.addLine(start_pos, -40, start_pos, 40, QPen(QColor("#55ff55"), 2))
                scene.addLine(end_pos, -40, end_pos, 40, QPen(QColor("#55ff55"), 2))
            for cue in deck.cue_points:
                cue_pos = cue / len(deck.audio) * 200
                scene.addLine(cue_pos, -30, cue_pos, 30, QPen(QColor("#ff0000"), 2))
        self.status_bar.showMessage(f"Deck {deck_id+1}: Waveform güncellendi")

    def zoom_waveform(self, event, deck_id):
        """Waveform’u yakınlaştırır."""
        delta = event.angleDelta().y()
        self.waveform_scale = max(0.1, min(10.0, self.waveform_scale + delta / 1200))
        self.update_waveform(deck_id)

    def start_loop_drag(self, event, deck_id):
        """Loop başlangıcını ayarlar."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.audio is not None:
            pos = event.pos().x() / 200 * len(deck.audio)
            self.deck_widgets[deck_id]["loop_start"] = pos

    def update_loop_drag(self, event, deck_id):
        """Loop aralığını günceller."""
        deck = self.audio_chain.dj_decks[deck_id]
        if deck.audio is not None and self.deck_widgets[deck_id]["loop_start"] is not None:
            pos = event.pos().x() / 200 * len(deck.audio)
            start = min(self.deck_widgets[deck_id]["loop_start"], pos)
            deck.set_loop(start, beat_count=8)
            self.update_waveform(deck_id)

    def end_loop_drag(self, event, deck_id):
        """Loop ayarlamasını tamamlar."""
        self.deck_widgets[deck_id]["loop_start"] = None
        self.status_bar.showMessage(f"Deck {deck_id+1}: Loop ayarlandı")

    def add_effect(self, fx_name):
        """Efekt ekler."""
        try:
            self.audio_chain.dj_decks[0].effect_rack.add_effect(fx_name)
            self.fx_params.addItem(fx_name)
            self.status_bar.showMessage(f"Efekt eklendi: {fx_name}")
        except Exception as e:
            logging.error(f"Efekt ekleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Efekt eklenemedi: {str(e)}")

    def edit_effect_params(self, item):
        """Efekt parametrelerini düzenler."""
        fx_name = item.text()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{fx_name} Parametreleri")
        dialog.setStyleSheet("QDialog { background-color: #002b5c; color: white; }")
        layout = QFormLayout(dialog)
        params = {}
        sliders = {}
        value_inputs = {}
        
        if fx_name == "Reverb":
            params = {
                "room_size": {"range": (0.0, 1.0), "default": 0.5, "unit": "", "label": "Room Size"},
                "damping": {"range": (0.0, 1.0), "default": 0.5, "unit": "", "label": "Damping"},
                "wet": {"range": (0.0, 1.0), "default": 0.3, "unit": "", "label": "Wet Mix"}
            }
        elif fx_name == "Compressor":
            params = {
                "threshold": {"range": (-60.0, 0.0), "default": -24.0, "unit": "dB", "label": "Threshold"},
                "ratio": {"range": (1.0, 20.0), "default": 4.0, "unit": ":1", "label": "Ratio"},
                "wet": {"range": (0.0, 1.0), "default": 0.3, "unit": "", "label": "Wet Mix"}
            }
        elif fx_name == "Flanger":
            params = {
                "depth": {"range": (0.0, 0.01), "default": 0.005, "unit": "s", "label": "Depth"},
                "rate": {"range": (0.0, 5.0), "default": 0.5, "unit": "Hz", "label": "Rate"},
                "wet": {"range": (0.0, 1.0), "default": 0.3, "unit": "", "label": "Wet Mix"}
            }
        
        for name, info in params.items():
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            value_range = info["range"]
            default_value = info["default"]
            slider.setValue(int((default_value - value_range[0]) / (value_range[1] - value_range[0]) * 100))
            value_input = QLineEdit()
            value_input.setFixedWidth(50)
            value_input.setText(f"{default_value:.2f}")
            unit_label = QLabel(f"{default_value:.2f} {info['unit']}")
            
            def update_input(slider_value, input_field, label, v_range, unit):
                value = v_range[0] + (slider_value / 100) * (v_range[1] - v_range[0])
                input_field.setText(f"{value:.2f}")
                label.setText(f"{value:.2f} {unit}")
            
            def update_slider(text, slider, label, v_range, unit):
                try:
                    value = float(text)
                    if v_range[0] <= value <= v_range[1]:
                        slider.setValue(int((value - v_range[0]) / (value_range[1] - value_range[0]) * 100))
                        label.setText(f"{value:.2f} {unit}")
                except ValueError:
                    pass
            
            slider.valueChanged.connect(lambda v, inp=value_input, lbl=unit_label, vr=value_range, u=info["unit"]: update_input(v, inp, lbl, vr, u))
            value_input.textChanged.connect(lambda t, s=slider, lbl=unit_label, vr=value_range, u=info["unit"]: update_slider(t, s, lbl, vr, u))
            
            row_layout.addWidget(slider)
            row_layout.addWidget(value_input)
            row_layout.addWidget(unit_label)
            layout.addRow(f"{info['label']} ({info['unit']})", row_widget)
            sliders[name] = slider
            value_inputs[name] = value_input
        
        preview_btn = QPushButton("Önizleme")
        preview_btn.clicked.connect(lambda: self.preview_effect(fx_name, {n: s.value() / 100 * (p["range"][1] - p["range"][0]) + p["range"][0] for n, s, p in [(n, sliders[n], params[n]) for n in params]}))
        ok_btn = QPushButton("Tamam")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(preview_btn)
        btn_layout.addWidget(ok_btn)
        layout.addRow(btn_layout)
        
        if dialog.exec_():
            new_params = {n: s.value() / 100 * (p["range"][1] - p["range"][0]) + p["range"][0] for n, s, p in [(n, sliders[n], params[n]) for n in params]}
            try:
                for fx in self.audio_chain.dj_decks[0].effect_rack.active_effects:
                    if fx["name"] == fx_name:
                        fx["params"] = new_params
                        self.audio_chain.dj_decks[0].effect_rack.add_effect(fx_name, new_params)
                self.status_bar.showMessage(f"{fx_name} parametreleri güncellendi")
            except Exception as e:
                logging.error(f"Efekt parametre hatası: {e}")
                QMessageBox.critical(self, "Hata", f"Efekt parametreleri güncellenemedi: {str(e)}")

    def preview_effect(self, fx_name, params):
        """Efekti önizler."""
        try:
            test_audio = np.sin(2 * np.pi * 440 * np.arange(int(self.audio_chain.fs * 2)) / self.audio_chain.fs) * 0.5
            effect_rack = EffectRack()
            effect_rack.add_effect(fx_name, params)
            processed = effect_rack.process(test_audio, self.audio_chain.fs)
            sd.play(processed, self.audio_chain.fs)
            self.status_bar.showMessage(f"{fx_name} önizlemesi oynatılıyor")
        except Exception as e:
            logging.error(f"Efekt önizleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Önizleme başarısız: {str(e)}")

    def change_pad_mode(self, mode):
        """Pad modunu değiştirir."""
        self.current_pad_mode = mode
        mode_configs = {
            "HotCue": {"color": "#007bff", "labels": ["Cue 1", "Cue 2", "Cue 3", "Cue 4"]},
            "Loop": {"color": "#00b7eb", "labels": ["4 Beat", "8 Beat", "16 Beat", "32 Beat"]},
            "Effect": {"color": "#ff5555", "labels": ["Reverb", "Comp", "Flanger", "Delay"]},
            "Sample": {"color": "#55ff55", "labels": ["Kick", "Snare", "Hat", "Vocal"]}
        }
        config = mode_configs[mode]
        for i, pad in enumerate(self.pads):
            pad.setStyleSheet(f"background-color: {config['color']}; border-radius: 6px; font-size: 10px;")
            pad.setText(config["labels"][i % 4])
            pad.setToolTip(f"Pad {i+1}: {mode} - {config['labels'][i % 4]}")
        self.status_bar.showMessage(f"{mode} modu aktif")

    def show_pad_menu(self, index):
        """Pad sağ tıklama menüsü."""
        menu = QMenu(self)
        for mode in self.pad_modes:
            action = menu.addAction(mode)
            action.setCheckable(True)
            action.setChecked(mode == self.current_pad_mode)
            action.triggered.connect(lambda _, m=mode: self.change_pad_mode(m))
        menu.exec_(self.pads[index].mapToGlobal(self.pads[index].rect().bottomLeft()))

    def trigger_pad(self, index):
        """Pad tetikleme."""
        deck_id = index // 4
        pad_id = index % 4
        deck = self.audio_chain.dj_decks[deck_id]
        beat_counts = [4, 8, 16, 32]
        try:
            if self.current_pad_mode == "Loop":
                deck.set_loop(deck.position, beat_count=beat_counts[pad_id])
                self.update_waveform(deck_id)
            elif self.current_pad_mode == "HotCue":
                deck.cue_points.append(deck.position)
                self.update_waveform(deck_id)
            elif self.current_pad_mode == "Effect":
                fx = ["Reverb", "Compressor", "Flanger", "Delay"][pad_id]
                deck.effect_rack.add_effect(fx)
                self.status_bar.showMessage(f"Deck {deck_id+1}: {fx} uygulandı")
            elif self.current_pad_mode == "Sample":
                sample = self.audio_chain.ai_remixer.samples[list(self.audio_chain.ai_remixer.samples.keys())[pad_id]]
                sd.play(sample, self.audio_chain.fs)
                self.status_bar.showMessage(f"Deck {deck_id+1}: Sample oynatıldı")
            anim = QPropertyAnimation(self.pads[index], b"styleSheet")
            anim.setStartValue("background-color: #ff5555; border-radius: 6px;")
            anim.setEndValue(f"background-color: {['#007bff', '#00b7eb', '#ff5555', '#55ff55'][pad_id % 4]}; border-radius: 6px;")
            anim.setDuration(100)
            anim.start()
        except Exception as e:
            logging.error(f"Pad tetikleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Pad işlemi başarısız: {str(e)}")

    def generate_ai_remix(self):
        """AI remix üretir."""
        description = self.ai_description.text()
        bpm = int(self.ai_bpm.currentText())
        key = self.ai_key.currentText()
        if description:
            try:
                progress = QProgressDialog("Remix üretiliyor...", "İptal", 0, 100, self)
                progress.setWindowTitle("AI Remix")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                
                def remix_thread():
                    remix = self.audio_chain.ai_remixer.generate_remix(description, bpm=bpm, key=key)
                    sf.write("remix.wav", remix, self.audio_chain.fs)
                    self.audio_chain.dj_decks[1].load_track("remix.wav")
                    progress.setValue(100)
                
                thread = threading.Thread(target=remix_thread, daemon=True)
                thread.start()
                
                while progress.value() < 100 and not progress.wasCanceled():
                    QApplication.processEvents()
                    time.sleep(0.1)
                
                if progress.wasCanceled():
                    self.status_bar.showMessage("Remix üretimi iptal edildi")
                    return
                
                self.update_waveform(1)
                self.status_bar.showMessage(f"AI Remix üretildi: {description} (~2.5s)")
            except Exception as e:
                logging.error(f"AI Remix üretme hatası: {e}")
                QMessageBox.critical(self, "Hata", f"Remix üretilemedi: {str(e)}")

    def add_clip(self, track_id):
        """Clip ekler."""
        file = QFileDialog.getOpenFileName(self, "Clip Yükle", "", "Audio Files (*.wav *.flac)")[0]
        if file:
            try:
                start_time = float(self.track_widgets[track_id]["start_time"].text())
                if start_time < 0:
                    raise ValueError("Başlangıç zamanı negatif olamaz")
                with sf.SoundFile(file) as f:
                    audio = f.read()
                self.audio_chain.production_tracks[track_id].add_clip(audio, start_time)
                self.status_bar.showMessage(f"Track {track_id+1}: Clip eklendi ({start_time}s)")
            except ValueError as e:
                logging.error(f"Clip ekleme hatası: {e}")
                QMessageBox.critical(self, "Hata", f"Geçersiz başlangıç zamanı: {str(e)}")
            except Exception as e:
                logging.error(f"Clip ekleme hatası: {e}")
                QMessageBox.critical(self, "Hata", f"Clip yüklenemedi: {str(e)}")

    def add_midi_note(self, note, velocity, start, duration, track_id=0):
        """MIDI nota ekler."""
        try:
            self.audio_chain.production_tracks[track_id].add_midi_note(note, velocity, start, duration)
            self.update_piano_roll(track_id)
            self.status_bar.showMessage(f"Track {track_id+1}: MIDI nota eklendi")
        except ValueError as e:
            logging.error(f"MIDI ekleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Geçersiz MIDI parametreleri: {str(e)}")

    def add_track_effect(self, track_id):
        """Track’e efekt ekler."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Track Efekti")
        layout = QVBoxLayout(dialog)
        fx_selector = QComboBox()
        fx_selector.addItems(EFFECTS_LIST)
        ok_btn = QPushButton("Ekle")
        ok_btn.clicked.connect(dialog.accept)
        layout.addWidget(fx_selector)
        layout.addWidget(ok_btn)
        if dialog.exec_():
            try:
                self.audio_chain.production_tracks[track_id].add_effect(fx_selector.currentText())
                self.status_bar.showMessage(f"Track {track_id+1}: {fx_selector.currentText()} eklendi")
            except Exception as e:
                logging.error(f"Track efekt hatası: {e}")
                QMessageBox.critical(self, "Hata", f"Efekt eklenemedi: {str(e)}")

    def preview_track(self, track_id):
        """Track’i önizler."""
        try:
            track = self.audio_chain.production_tracks[track_id]
            output = np.zeros((int(self.audio_chain.fs * 5), 2))
            for i in range(0, len(output), 128):
                output[i:i+128] = track.process(128, i)
            sd.play(output, self.audio_chain.fs)
            self.status_bar.showMessage(f"Track {track_id+1} önizlemesi oynatılıyor")
        except Exception as e:
            logging.error(f"Track önizleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Önizleme başarısız: {str(e)}")

    def export_to_playlist(self, track_id):
        """Track’i playlist’e ekler."""
        try:
            track = self.audio_chain.production_tracks[track_id]
            output = np.zeros((int(self.audio_chain.fs * 60), 2))
            for i in range(0, len(output), 128):
                output[i:i+128] = track.process(128, i)
            file_path = f"track_{track_id+1}.wav"
            sf.write(file_path, output, self.audio_chain.fs)
            self.audio_chain.player = type('Player', (), {'playlist': []})()
            self.audio_chain.player.playlist = getattr(self.audio_chain.player, 'playlist', [])
            self.audio_chain.player.playlist.append(file_path)
            self.status_bar.showMessage(f"Track {track_id+1} playlist’e eklendi")
        except Exception as e:
            logging.error(f"Playlist’e ekleme hatası: {e}")
            QMessageBox.critical(self, "Hata", f"Playlist’e eklenemedi: {str(e)}")

    def update_piano_roll(self, track_id=0):
        """Piano roll’u günceller."""
        self.piano_roll_scene.clear()
        track = self.audio_chain.production_tracks[track_id]
        for note in track.midi_notes:
            x = note["start"] * 50
            width = note["duration"] * 50
            y = (127 - note["note"]) * 2
            height = 2
            rect = self.piano_roll_scene.addRect(x, y, width, height, QPen(Qt.NoPen), QBrush(QColor("#007bff")))
            rect.setData(0, note)

    def add_midi_note_from_click(self, event, track_id):
        """Tıklama ile MIDI nota ekler."""
        x = event.pos().x() / 50
        y = event.pos().y() / 2
        note = int(127 - y)
        if 0 <= note <= 127 and x >= 0:
            self.add_midi_note(note, 100, x, 0.5, track_id)
        else:
            QMessageBox.critical(self, "Hata", "Geçersiz nota veya zaman. Nota 0-127, zaman ≥0 olmalı.")

    def edit_midi_note(self, event):
        """MIDI notasını düzenler."""
        item = self.piano_roll_scene.itemAt(event.pos())
        if item:
            note = item.data(0)
            dialog = QDialog(self)
            dialog.setWindowTitle("MIDI Notasını Düzenle")
            layout = QFormLayout(dialog)
            note_input = QLineEdit(str(note["note"]))
            velocity_input = QLineEdit(str(note["velocity"]))
            start_input = QLineEdit(str(note["start"]))
            duration_input = QLineEdit(str(note["duration"]))
            layout.addRow("Nota (0-127)", note_input)
            layout.addRow("Velocity (0-127)", velocity_input)
            layout.addRow("Başlangıç (s)", start_input)
            layout.addRow("Süre (s)", duration_input)
            ok_btn = QPushButton("Tamam")
            ok_btn.clicked.connect(dialog.accept)
            layout.addRow(ok_btn)
            if dialog.exec_():
                try:
                    new_note = int(note_input.text())
                    new_velocity = int(velocity_input.text())
                    new_start = float(start_input.text())
                    new_duration = float(duration_input.text())
                    if not (0 <= new_note <= 127 and 0 <= new_velocity <= 127 and new_start >= 0 and new_duration > 0):
                        raise ValueError("Geçersiz değerler")
                    note.update({"note": new_note, "velocity": new_velocity, "start": new_start, "duration": new_duration})
                    self.update_piano_roll()
                    self.status_bar.showMessage("MIDI notası güncellendi")
                except ValueError as e:
                    QMessageBox.critical(self, "Hata", f"Geçersiz giriş: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicaProOmnibusGUI()
    window.show()
    sys.exit(app.exec_())

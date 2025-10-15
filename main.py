#!/usr/bin/env python3
"""
swipe_ai_assistant.py
Single-file PyQt5 app implementing:
- Floating gradient button with mode switching
- Translator/clipboard QA mode
- Document (PDF) upload + QA mode
- S3 upload + duplicate detection + PDF text extraction (PyPDF2)
- Ollama integration for Q&A with formatted, point-wise answers
- Draggable, resizable, and fully customizable UI

Requirements:
- PyQt5
- boto3
- python-dotenv
- requests
- PyPDF2
- pyperclip
- deep_translator (optional, for translation)
- fuzzywuzzy (optional, for similarity checks)
"""

import sys
import os
import json
import traceback
from functools import partial
from pathlib import Path
from time import time

# Check for display
if os.environ.get('DISPLAY') is None:
    print("⚠️  ERROR: No DISPLAY environment variable found!")
    print("💡 Solution: Run 'export DISPLAY=:0' or ensure you're running in a graphical environment")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"⚠️  Warning: Could not load .env file: {e}")
    print("Continuing with environment variables...")

# --- Config from .env ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")            # bucket for original PDFs
AWS_EXTRACT_BUCKET = os.getenv("AWS_EXTRACT_BUCKET")      # bucket for extracted text
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Fallbacks and sanity
if not AWS_BUCKET_NAME:
    print("Warning: AWS_BUCKET_NAME not set in .env — S3 upload will fail until configured.")

# --- Imports for functionality ---
import boto3
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QComboBox,
    QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox,
    QLineEdit, QMenu, QFileDialog, QLabel, QFrame, QSizeGrip
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal, QPropertyAnimation, QRect, QEasingCurve, QSize
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QLinearGradient, QBrush, QCursor

# Optional libs
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except Exception:
    HAS_TRANSLATOR = False

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True
except Exception:
    HAS_FUZZY = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except Exception:
    HAS_PYPERCLIP = False

from PyPDF2 import PdfReader

# ----------------- Utility functions -----------------
def format_ollama_answer(raw_text: str) -> str:
    """
    Format Ollama answers with proper structure and point-wise presentation
    """
    if not raw_text:
        return "⚠️ No response from model."

    text = raw_text.strip()
    
    # Return formatted text as-is (Ollama will handle formatting based on our prompt)
    return text

# ----------------- S3 Upload Thread -----------------
class UploadThread(QThread):
    progress = pyqtSignal(str)
    extracted_text_signal = pyqtSignal(str, str)  # (extracted_text, s3_key)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = str(file_path)
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            # Check duplicate: list objects keys in bucket
            already = False
            try:
                resp = self.s3.list_objects_v2(Bucket=AWS_BUCKET_NAME, Prefix=filename)
                if 'Contents' in resp:
                    for obj in resp['Contents']:
                        if obj['Key'] == filename:
                            already = True
                            break
            except Exception:
                # silent fallback; still attempt upload
                pass

            if already:
                self.progress.emit(f"⚠️ File '{filename}' already exists in {AWS_BUCKET_NAME}. Skipping upload.")
            else:
                self.s3.upload_file(self.file_path, AWS_BUCKET_NAME, filename)
                self.progress.emit(f"✅ Uploaded '{filename}' → s3://{AWS_BUCKET_NAME}/{filename}")

            # Extract text from PDF
            text = ""
            try:
                reader = PdfReader(self.file_path)
                for p in reader.pages:
                    # page.extract_text() may return None
                    page_text = p.extract_text() or ""
                    text += page_text + "\n\n"
            except Exception as e:
                self.progress.emit(f"⚠️ PDF extraction failed: {e}")
                text = ""

            # Store extracted .txt in extract bucket
            if text and AWS_EXTRACT_BUCKET:
                key = f"{Path(filename).stem}.txt"
                try:
                    self.s3.put_object(Bucket=AWS_EXTRACT_BUCKET, Key=key, Body=text.encode("utf-8"))
                    self.progress.emit(f"✅ Extracted text stored → s3://{AWS_EXTRACT_BUCKET}/{key}")
                    self.extracted_text_signal.emit(text, key)
                except Exception as e:
                    self.progress.emit(f"⚠️ Failed to store extracted text: {e}")
                    self.extracted_text_signal.emit(text, "")  # still provide extracted text
            else:
                # send text back even if not stored
                self.extracted_text_signal.emit(text, "")

        except Exception as e:
            tb = traceback.format_exc()
            self.progress.emit(f"⚠️ UploadThread error: {e}\n{tb}")

# ----------------- Floating AI Button -----------------
class FloatingAIButton(QWidget):
    modeChanged = pyqtSignal(str)  # "translate" or "document"

    def __init__(self, parent=None, diameter=80):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.diameter = diameter
        self.setFixedSize(diameter, diameter)
        self.icon_text = "AI"
        self.old_pos = None
        self.dragging = False
        self.current_mode = "translate"
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setToolTip("Click to open • Right-click for menu")
        # Starting position bottom-right-ish
        screen_geo = QApplication.primaryScreen().availableGeometry()
        start_x = max(50, screen_geo.width() - 150)
        start_y = max(50, screen_geo.height() - 200)
        self.move(start_x, start_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Gradient fill based on mode
        grad = QLinearGradient(0, 0, self.width(), self.height())
        if self.current_mode == "translate":
            grad.setColorAt(0.0, QColor(106, 17, 203))  # purple
            grad.setColorAt(1.0, QColor(33, 150, 243))  # blue
        else:
            grad.setColorAt(0.0, QColor(76, 175, 80))   # green
            grad.setColorAt(1.0, QColor(33, 150, 243))  # blue
        
        brush = QBrush(grad)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.diameter, self.diameter)

        # Icon text
        painter.setPen(QPen(QColor(255,255,255)))
        font = QFont("Arial", int(self.diameter/4), QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.icon_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()
            self.dragging = True
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if not self.dragging or self.old_pos is None:
            return
        delta = event.globalPos() - self.old_pos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.dragging:
                self.dragging = False
                # If barely moved, treat as click
                if self.old_pos and (event.globalPos() - self.old_pos).manhattanLength() < 10:
                    self.modeChanged.emit(self.current_mode)
            self.old_pos = None

    def show_context_menu(self, pos):
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: white;
                border: 2px solid #6a11cb;
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #6a11cb;
            }
        """)

        translate_action = menu.addAction("🌐 Translate Mode")
        document_action = menu.addAction("📄 Document Mode")
        menu.addSeparator()
        exit_action = menu.addAction("❌ Exit")
        
        action = menu.exec_(pos)
        if action == translate_action:
            self.current_mode = "translate"
            self.update()
            self.modeChanged.emit("translate")
        elif action == document_action:
            self.current_mode = "document"
            self.update()
            self.modeChanged.emit("document")
        elif action == exit_action:
            QApplication.quit()

# ----------------- Main AI Assistant Window -----------------
class AIAssistantWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setStyleSheet("background:#0a0e1a; color:white; border:2px solid #6a11cb; border-radius:16px;")
        self.resize(600, 650)
        self.old_pos = None
        self.current_mode = "translate"
        self.last_clip = ""
        self.pdf_extracted_text = ""
        self.current_pdf_key = None
        
        # Window properties for customization
        self.min_width = 400
        self.min_height = 400
        self.max_width = 1200
        self.max_height = 900
        self.setMinimumSize(self.min_width, self.min_height)
        self.setMaximumSize(self.max_width, self.max_height)
        
        self.init_ui()
        
        # Clipboard timer for translate mode
        self.clipboard_timer = QTimer()
        self.clipboard_timer.timeout.connect(self.check_clipboard)
        
        self.position_window()

    def position_window(self):
        screen_geo = QApplication.primaryScreen().availableGeometry()
        self.move(screen_geo.width() - self.width() - 100, 80)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === HEADER ===
        header = self.create_header()
        main_layout.addWidget(header)

        # === MODE SELECTOR ===
        mode_frame = QFrame()
        mode_frame.setStyleSheet("background:#12141f; border-radius:0px; padding:10px;")
        mode_layout = QHBoxLayout()
        
        self.translate_btn = QPushButton("🌐 Translate")
        self.translate_btn.setCheckable(True)
        self.translate_btn.setChecked(True)
        self.translate_btn.clicked.connect(lambda: self.switch_mode("translate"))
        
        self.document_btn = QPushButton("📄 Document Q&A")
        self.document_btn.setCheckable(True)
        self.document_btn.clicked.connect(lambda: self.switch_mode("document"))
        
        for btn in [self.translate_btn, self.document_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background:#1e2030;
                    color:white;
                    border:2px solid #2e3040;
                    border-radius:8px;
                    padding:10px 20px;
                    font-weight:600;
                }
                QPushButton:checked {
                    background:#6a11cb;
                    border-color:#6a11cb;
                }
                QPushButton:hover {
                    background:#2e3550;
                }
            """)
        
        mode_layout.addWidget(self.translate_btn)
        mode_layout.addWidget(self.document_btn)
        mode_frame.setLayout(mode_layout)
        main_layout.addWidget(mode_frame)

        # === CONTENT AREA ===
        content = QFrame()
        content.setStyleSheet("background:#0f1117; border-radius:0px;")
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(15, 15, 15, 15)
        
        # Translate Mode Widgets
        self.translate_widgets = self.create_translate_widgets()
        self.content_layout.addWidget(self.translate_widgets)
        
        # Document Mode Widgets
        self.document_widgets = self.create_document_widgets()
        self.document_widgets.hide()
        self.content_layout.addWidget(self.document_widgets)
        
        content.setLayout(self.content_layout)
        main_layout.addWidget(content, 1)

        # Add resize grip
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("QSizeGrip { width: 16px; height: 16px; }")
        
        self.setLayout(main_layout)

    def create_header(self):
        header = QFrame()
        header.setStyleSheet("background:#1a1d2e; border-radius:16px 16px 0 0; padding:10px;")
        header.setFixedHeight(50)
        
        layout = QHBoxLayout()
        
        title = QLabel("✨ AI Assistant")
        title.setStyleSheet("color:#ffffff; font-size:16px; font-weight:700;")
        
        # Control buttons
        self.settings_btn = QPushButton("⚙️")
        self.minimize_btn = QPushButton("─")
        self.clear_btn = QPushButton("🗑")
        self.close_btn = QPushButton("✕")
        
        for btn in [self.settings_btn, self.minimize_btn, self.clear_btn, self.close_btn]:
            btn.setFixedSize(32, 32)
            btn.setStyleSheet("""
                QPushButton {
                    background:#2e3040;
                    color:white;
                    border:none;
                    border-radius:6px;
                    font-weight:bold;
                }
                QPushButton:hover {
                    background:#ff5252;
                }
            """)
        
        self.settings_btn.setStyleSheet(self.settings_btn.styleSheet().replace("#ff5252", "#9c27b0"))
        self.minimize_btn.setStyleSheet(self.minimize_btn.styleSheet().replace("#ff5252", "#ffa726"))
        self.clear_btn.setStyleSheet(self.clear_btn.styleSheet().replace("#ff5252", "#42a5f5"))
        
        self.settings_btn.clicked.connect(self.show_settings_menu)
        self.minimize_btn.clicked.connect(self.showMinimized)
        self.clear_btn.clicked.connect(self.clear_content)
        self.close_btn.clicked.connect(self.hide)
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.settings_btn)
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.clear_btn)
        layout.addWidget(self.close_btn)
        
        header.setLayout(layout)
        header.mousePressEvent = self.header_mouse_press
        header.mouseMoveEvent = self.header_mouse_move
        header.mouseReleaseEvent = self.header_mouse_release
        
        return header

    def show_settings_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e2e;
                color: white;
                border: 2px solid #9c27b0;
                border-radius: 8px;
                padding: 8px;
            }
            QMenu::item {
                padding: 10px 30px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #9c27b0;
            }
            QMenu::separator {
                height: 1px;
                background: #444;
                margin: 5px 10px;
            }
        """)
        
        # Size options
        size_menu = menu.addMenu("📏 Window Size")
        small_action = size_menu.addAction("Small (500x500)")
        medium_action = size_menu.addAction("Medium (600x650)")
        large_action = size_menu.addAction("Large (800x800)")
        xlarge_action = size_menu.addAction("X-Large (1000x900)")
        
        menu.addSeparator()
        
        # Opacity options
        opacity_menu = menu.addMenu("💡 Window Opacity")
        opacity_100 = opacity_menu.addAction("100% (Solid)")
        opacity_90 = opacity_menu.addAction("90%")
        opacity_80 = opacity_menu.addAction("80%")
        opacity_70 = opacity_menu.addAction("70%")
        opacity_60 = opacity_menu.addAction("60% (Transparent)")
        
        menu.addSeparator()
        
        # Position options
        position_menu = menu.addMenu("📍 Window Position")
        top_left = position_menu.addAction("Top Left")
        top_right = position_menu.addAction("Top Right")
        bottom_left = position_menu.addAction("Bottom Left")
        bottom_right = position_menu.addAction("Bottom Right")
        center = position_menu.addAction("Center")
        
        menu.addSeparator()
        
        # Stay on top toggle
        stay_on_top_action = menu.addAction("📌 Always on Top")
        stay_on_top_action.setCheckable(True)
        stay_on_top_action.setChecked(self.windowFlags() & Qt.WindowStaysOnTopHint)
        
        action = menu.exec_(self.settings_btn.mapToGlobal(self.settings_btn.rect().bottomLeft()))
        
        if action == small_action:
            self.resize(500, 500)
        elif action == medium_action:
            self.resize(600, 650)
        elif action == large_action:
            self.resize(800, 800)
        elif action == xlarge_action:
            self.resize(1000, 900)
        elif action == opacity_100:
            self.setWindowOpacity(1.0)
        elif action == opacity_90:
            self.setWindowOpacity(0.9)
        elif action == opacity_80:
            self.setWindowOpacity(0.8)
        elif action == opacity_70:
            self.setWindowOpacity(0.7)
        elif action == opacity_60:
            self.setWindowOpacity(0.6)
        elif action == top_left:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.move(20, 20)
        elif action == top_right:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.move(screen_geo.width() - self.width() - 20, 20)
        elif action == bottom_left:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.move(20, screen_geo.height() - self.height() - 20)
        elif action == bottom_right:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.move(screen_geo.width() - self.width() - 20, screen_geo.height() - self.height() - 20)
        elif action == center:
            screen_geo = QApplication.primaryScreen().availableGeometry()
            self.move((screen_geo.width() - self.width()) // 2, (screen_geo.height() - self.height()) // 2)
        elif action == stay_on_top_action:
            if stay_on_top_action.isChecked():
                self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            else:
                self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.show()

    def create_translate_widgets(self):
        widget = QFrame()
        layout = QVBoxLayout()
        
        # Language selector
        lang_label = QLabel("🌍 Select Translation Language:")
        lang_label.setStyleSheet("color:#aaaaaa; font-size:13px; font-weight:600;")
        
        self.lang_box = QComboBox()
        self.lang_box.addItems(["english", "hindi", "spanish", "french", "german", "chinese", "arabic", "japanese", "russian", "portuguese", "italian", "korean", "turkish", "dutch", "polish"])
        self.lang_box.setCurrentText("english")
        self.lang_box.setStyleSheet("""
            QComboBox {
                background:#1e2030;
                color:white;
                padding:10px;
                border:2px solid #2e3040;
                border-radius:8px;
                font-size:13px;
            }
            QComboBox:hover {
                border-color:#6a11cb;
            }
            QComboBox::drop-down {
                border:none;
            }
            QComboBox QAbstractItemView {
                background:#1e2030;
                color:white;
                selection-background-color:#6a11cb;
            }
        """)
        # Connect language change to re-translate current text
        self.lang_box.currentTextChanged.connect(self.on_language_changed)
        
        layout.addWidget(lang_label)
        layout.addWidget(self.lang_box)
        
        # Text display area
        display_label = QLabel("📋 Auto-Translation (Copied text will appear here):")
        display_label.setStyleSheet("color:#aaaaaa; font-size:12px; margin-top:15px;")
        
        self.translate_text_area = QTextEdit()
        self.translate_text_area.setReadOnly(True)
        self.translate_text_area.setStyleSheet("""
            QTextEdit {
                background:#12141f;
                color:#ffffff;
                border:2px solid #2e3040;
                border-radius:8px;
                padding:15px;
                font-size:14px;
                line-height:1.6;
            }
        """)
        self.translate_text_area.setPlaceholderText("📋 Copy any text from anywhere...\n\n✨ It will automatically appear here and translate to your selected language!")
        
        layout.addWidget(display_label)
        layout.addWidget(self.translate_text_area, 1)
        
        # Question input for Ollama
        qa_label = QLabel("💬 Ask about copied text:")
        qa_label.setStyleSheet("color:#aaaaaa; font-size:12px; margin-top:10px;")
        
        self.translate_input = QLineEdit()
        self.translate_input.setPlaceholderText("Type your question here...")
        self.translate_input.setStyleSheet("""
            QLineEdit {
                background:#1e2030;
                color:white;
                padding:10px;
                border:2px solid #2e3040;
                border-radius:8px;
            }
            QLineEdit:focus {
                border-color:#6a11cb;
            }
        """)
        
        self.translate_send_btn = QPushButton("Send Question")
        self.translate_send_btn.setStyleSheet("""
            QPushButton {
                background:#6a11cb;
                color:white;
                border:none;
                padding:10px 20px;
                border-radius:8px;
                font-weight:600;
            }
            QPushButton:hover {
                background:#8a31eb;
            }
            QPushButton:pressed {
                background:#5a01bb;
            }
        """)
        self.translate_send_btn.clicked.connect(self.ask_translate_ollama)
        
        qa_row = QHBoxLayout()
        qa_row.addWidget(self.translate_input)
        qa_row.addWidget(self.translate_send_btn)
        
        layout.addWidget(qa_label)
        layout.addLayout(qa_row)
        
        widget.setLayout(layout)
        return widget

    def create_document_widgets(self):
        widget = QFrame()
        layout = QVBoxLayout()
        
        # Upload section
        doc_upload_label = QLabel("📤 Upload Document:")
        doc_upload_label.setStyleSheet("color:#aaaaaa; font-size:12px;")
        
        self.upload_btn = QPushButton("Choose PDF File")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background:#1976d2;
                color:white;
                border:none;
                padding:12px 20px;
                border-radius:8px;
                font-weight:600;
            }
            QPushButton:hover {
                background:#1e88e5;
            }
            QPushButton:pressed {
                background:#1565c0;
            }
        """)
        self.upload_btn.clicked.connect(self.select_file)
        
        self.upload_status = QLabel("No document uploaded yet")
        self.upload_status.setStyleSheet("color:#888888; font-size:11px; font-style:italic;")
        
        layout.addWidget(doc_upload_label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.upload_status)
        
        # Language selector for document responses
        doc_lang_label = QLabel("🌐 Response Language:")
        doc_lang_label.setStyleSheet("color:#aaaaaa; font-size:12px; margin-top:10px;")
        
        self.doc_lang_box = QComboBox()
        self.doc_lang_box.addItems(["english", "hindi", "spanish", "french", "german", "chinese", "arabic", "japanese", "russian", "portuguese"])
        self.doc_lang_box.setStyleSheet("""
            QComboBox {
                background:#1e2030;
                color:white;
                padding:8px;
                border:2px solid #2e3040;
                border-radius:8px;
            }
            QComboBox:hover {
                border-color:#4caf50;
            }
            QComboBox::drop-down {
                border:none;
            }
        """)
        
        layout.addWidget(doc_lang_label)
        layout.addWidget(self.doc_lang_box)
        
        # Document display area
        doc_label = QLabel("📄 Document Q&A:")
        doc_label.setStyleSheet("color:#aaaaaa; font-size:12px; margin-top:10px;")
        
        self.document_text_area = QTextEdit()
        self.document_text_area.setReadOnly(True)
        self.document_text_area.setStyleSheet("""
            QTextEdit {
                background:#12141f;
                color:white;
                border:2px solid #2e3040;
                border-radius:8px;
                padding:10px;
                font-size:13px;
            }
        """)
        self.document_text_area.setPlaceholderText("Upload a PDF and ask questions about it...")
        
        layout.addWidget(doc_label)
        layout.addWidget(self.document_text_area, 1)
        
        # Question input for document
        qa_label = QLabel("❓ Ask about document:")
        qa_label.setStyleSheet("color:#aaaaaa; font-size:12px; margin-top:10px;")
        
        self.document_input = QLineEdit()
        self.document_input.setPlaceholderText("What would you like to know?")
        self.document_input.setStyleSheet("""
            QLineEdit {
                background:#1e2030;
                color:white;
                padding:10px;
                border:2px solid #2e3040;
                border-radius:8px;
            }
            QLineEdit:focus {
                border-color:#4caf50;
            }
        """)
        self.document_input.returnPressed.connect(self.ask_document_ollama)
        
        self.document_send_btn = QPushButton("Ask Question")
        self.document_send_btn.setEnabled(False)
        self.document_send_btn.setStyleSheet("""
            QPushButton {
                background:#4caf50;
                color:white;
                border:none;
                padding:10px 20px;
                border-radius:8px;
                font-weight:600;
            }
            QPushButton:hover:enabled {
                background:#66bb6a;
            }
            QPushButton:pressed:enabled {
                background:#388e3c;
            }
            QPushButton:disabled {
                background:#2e3040;
                color:#666666;
            }
        """)
        self.document_send_btn.clicked.connect(self.ask_document_ollama)
        
        qa_row = QHBoxLayout()
        qa_row.addWidget(self.document_input)
        qa_row.addWidget(self.document_send_btn)
        
        layout.addWidget(qa_label)
        layout.addLayout(qa_row)
        
        widget.setLayout(layout)
        return widget

    def header_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def header_mouse_move(self, event):
        if self.old_pos is not None:
            delta = event.globalPos() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

    def header_mouse_release(self, event):
        self.old_pos = None

    def switch_mode(self, mode):
        self.current_mode = mode
        
        if mode == "translate":
            self.translate_btn.setChecked(True)
            self.document_btn.setChecked(False)
            self.translate_widgets.show()
            self.document_widgets.hide()
            # Start clipboard monitoring with faster interval
            self.clipboard_timer.start(300)  # Check every 300ms for faster response
        else:
            self.translate_btn.setChecked(False)
            self.document_btn.setChecked(True)
            self.translate_widgets.hide()
            self.document_widgets.show()
            self.clipboard_timer.stop()

    def clear_content(self):
        if self.current_mode == "translate":
            self.translate_text_area.clear()
            self.translate_input.clear()
        else:
            self.document_text_area.clear()
            self.document_input.clear()

    def check_clipboard(self):
        """Auto-detect and translate copied text"""
        if not HAS_PYPERCLIP:
            return
        
        try:
            text = pyperclip.paste().strip()
        except Exception:
            return
        
        # Only process if text is new
        if not text or text == self.last_clip:
            return
        
        # Store the new clipboard text
        self.last_clip = text
        
        # Translate immediately with a small delay to prevent UI blocking
        QTimer.singleShot(50, lambda: self.translate_and_display(text))

    def on_language_changed(self):
        """Re-translate when language is changed"""
        if self.last_clip:
            # Re-translate the last copied text with new language
            self.translate_and_display(self.last_clip)

    def translate_and_display(self, text):
        """Translate text to selected language and display"""
        if not text:
            return
        
        target = self.lang_box.currentText()
        
        if not HAS_TRANSLATOR:
            self.translate_text_area.clear()
            self.translate_text_area.setText(
                "⚠️ Translation library not installed!\n\n"
                "Install it with:\npip install deep-translator\n\n"
                f"Original text:\n{text}"
            )
            return
        
        try:
            # Show loading indicator
            self.translate_text_area.clear()
            self.translate_text_area.setText(f"⏳ Translating to {target.upper()}...")
            
            # Force GUI update
            QApplication.processEvents()
            
            # Perform translation
            translator = GoogleTranslator(source='auto', target=target)
            translated_text = translator.translate(text)
            
            # Display translated text with clear formatting
            self.translate_text_area.clear()
            display_text = (
                f"🌍 Language: {target.upper()}\n"
                f"{'─' * 50}\n\n"
                f"{translated_text}"
            )
            self.translate_text_area.setText(display_text)
            
        except Exception as e:
            self.translate_text_area.clear()
            error_msg = (
                f"⚠️ Translation Error!\n\n"
                f"Target Language: {target}\n"
                f"Error: {str(e)}\n\n"
                f"💡 Tips:\n"
                f"• Check internet connection\n"
                f"• Try a different language\n"
                f"• Restart the application\n\n"
                f"Original Text:\n{text[:500]}..."
            )
            self.translate_text_area.setText(error_msg)

    def ask_translate_ollama(self):
        msg = self.translate_input.text().strip()
        if not msg:
            return
        
        if not self.last_clip:
            self.translate_text_area.append("\n⚠️ No copied text available. Copy some text first.")
            return
        
        target_lang = self.lang_box.currentText()
        
        # Enhanced system prompt for detailed responses
        system_prompt = (
            f"You are an expert AI assistant. Answer the user's question in {target_lang} language ONLY.\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"1. Provide detailed, comprehensive answers with multiple paragraphs\n"
            f"2. Structure your answer in point-wise format with clear numbered points\n"
            f"3. Each point should be detailed (3-5 sentences minimum)\n"
            f"4. Include examples, explanations, and context where relevant\n"
            f"5. Use ONLY the information from the copied text below\n"
            f"6. If the copied text doesn't contain enough information to answer, clearly state what's missing\n"
            f"7. Respond entirely in {target_lang} language\n\n"
            f"Copied Text:\n{self.last_clip}"
        )
        
        self.translate_text_area.append(f"\n\n<b><span style='color:#00e676'>You:</span></b> {msg}")
        self.translate_input.clear()
        self.translate_input.setEnabled(False)
        self.translate_send_btn.setEnabled(False)
        
        # Use QTimer to prevent UI freezing
        QTimer.singleShot(100, lambda: self._execute_translate_ollama(system_prompt, msg, target_lang))

    def _execute_translate_ollama(self, system_prompt, msg, target_lang):
        try:
            resp = requests.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Answer in {target_lang} language only: {msg}"}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2000,
                    }
                },
               
            )
            
            assistant_reply = ""
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict):
                        assistant_reply = data["message"].get("content", "")
                    elif "choices" in data and len(data["choices"]) > 0:
                        assistant_reply = data["choices"][0].get("message", {}).get("content", "")
                    elif "response" in data:
                        assistant_reply = data["response"]
                    else:
                        assistant_reply = json.dumps(data)
                else:
                    assistant_reply = str(data)
            except Exception:
                assistant_reply = resp.text
            
            # If response is not in target language, force translate it
            if assistant_reply and HAS_TRANSLATOR:
                try:
                    # Detect if response needs translation
                    if target_lang != "english":
                        assistant_reply = GoogleTranslator(source='auto', target=target_lang).translate(assistant_reply)
                except Exception:
                    pass  # Keep original response if translation fails
            
            formatted = format_ollama_answer(assistant_reply)
            self.translate_text_area.append(f"<b><span style='color:#4fc3f7'>Ollama ({target_lang}):</span></b>\n{formatted}")
            
        except Exception as e:
            self.translate_text_area.append(f"\n⚠️ Ollama request failed: {e}")
        finally:
            self.translate_input.setEnabled(True)
            self.translate_send_btn.setEnabled(True)

    def select_file(self):
        file_tuple = QFileDialog.getOpenFileName(self, "Select PDF file", "", "PDF Files (*.pdf);;All Files (*)")
        file_path = file_tuple[0] if file_tuple else None
        
        if not file_path or not os.path.isfile(file_path):
            self.upload_status.setText("⚠️ No valid file selected")
            return
        
        self.upload_status.setText(f"⏳ Uploading {os.path.basename(file_path)}...")
        self.upload_btn.setEnabled(False)
        
        self.uploader = UploadThread(file_path)
        self.uploader.progress.connect(self.on_upload_progress)
        self.uploader.extracted_text_signal.connect(self.on_extracted_text)
        self.uploader.start()

    def on_upload_progress(self, msg):
        self.upload_status.setText(msg)

    def on_extracted_text(self, text, s3_key):
        self.pdf_extracted_text = text or ""
        if s3_key:
            self.current_pdf_key = s3_key
        
        self.upload_status.setText("✅ PDF uploaded successfully!")
        self.upload_btn.setEnabled(True)
        self.document_send_btn.setEnabled(True)
        
        if self.pdf_extracted_text:
            self.document_text_area.setText(f"✅ Document loaded successfully! You can now ask questions.\n\nDocument ready for Q&A in {self.doc_lang_box.currentText()} language.")
        else:
            self.document_text_area.setText("⚠️ No text extracted. PDF might be image-based or encrypted.")

    def ask_document_ollama(self):
        question = self.document_input.text().strip()
        if not question:
            return
        
        if not self.pdf_extracted_text:
            self.document_text_area.append("\n⚠️ No document uploaded. Please upload a PDF first.")
            return
        
        target_lang = self.doc_lang_box.currentText()
        
        # Enhanced system prompt for detailed, document-restricted responses
        system_prompt = (
            f"You are an expert document analyst. You must answer STRICTLY based on the document content provided below.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Answer ONLY using information from the provided document text\n"
            f"2. If the answer is not in the document, clearly state: 'This information is not available in the provided document'\n"
            f"3. DO NOT use external knowledge or make assumptions beyond the document\n"
            f"4. Provide detailed, comprehensive answers (minimum 150-200 words)\n"
            f"5. Structure your answer in clear numbered points (use 1., 2., 3., etc.)\n"
            f"6. Each point should include detailed explanation (3-5 sentences)\n"
            f"7. Include relevant examples or quotes from the document\n"
            f"8. Add a summary or conclusion at the end\n"
            f"9. If the document contains tables, lists, or structured data, present them clearly\n\n"
            f"DOCUMENT TEXT:\n{self.pdf_extracted_text}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"Remember: Answer ONLY based on the document above. Respond in {target_lang} language with detailed point-wise format."
        )
        
        self.document_text_area.append(f"\n\n<b><span style='color:#00e676'>You:</span></b> {question}")
        self.document_input.clear()
        self.document_input.setEnabled(False)
        self.document_send_btn.setEnabled(False)
        
        # Use QTimer to prevent UI freezing
        QTimer.singleShot(100, lambda: self._execute_document_ollama(system_prompt, question, target_lang))

    def _execute_document_ollama(self, system_prompt, question, target_lang):
        try:
            resp = requests.post(
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"IMPORTANT: You MUST answer in {target_lang} language ONLY. Question: {question}"}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 3000,
                        "top_p": 0.9,
                    }
                },
                
            )
            
            assistant_reply = ""
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict):
                        assistant_reply = data["message"].get("content", "")
                    elif "choices" in data and len(data["choices"]) > 0:
                        assistant_reply = data["choices"][0].get("message", {}).get("content", "")
                    elif "response" in data:
                        assistant_reply = data["response"]
                    else:
                        assistant_reply = json.dumps(data)
                else:
                    assistant_reply = str(data)
            except Exception:
                assistant_reply = resp.text
            
            # Force translate to target language if not already
            if assistant_reply and HAS_TRANSLATOR and target_lang != "english":
                try:
                    assistant_reply = GoogleTranslator(source='auto', target=target_lang).translate(assistant_reply)
                except Exception:
                    pass  # Keep original if translation fails
            
            if not assistant_reply or len(assistant_reply.strip()) < 20:
                assistant_reply = f"⚠️ The model provided an insufficient response. Please try rephrasing your question or ensure the document contains relevant information."
            
            formatted = format_ollama_answer(assistant_reply)
            self.document_text_area.append(f"<b><span style='color:#81d4fa'>Ollama ({target_lang}):</span></b>\n\n{formatted}\n\n{'─'*50}")
            
        except Exception as e:
            self.document_text_area.append(f"\n⚠️ Ollama request failed: {e}\nPlease check if Ollama is running and try again.")
        finally:
            self.document_input.setEnabled(True)
            self.document_send_btn.setEnabled(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position size grip at bottom right
        self.size_grip.move(self.width() - 20, self.height() - 20)

# ----------------- Main Application -----------------
def main():
    try:
        print("🚀 Starting AI Assistant...")
        print(f"📍 Python version: {sys.version}")
        print(f"📍 PyQt5 available: Checking...")
        
        app = QApplication(sys.argv)
        print("✅ QApplication created successfully")
        
        # Set application to not quit when last window closes
        app.setQuitOnLastWindowClosed(False)
        
        # Create main window
        print("📱 Creating main window...")
        main_window = AIAssistantWindow()
        print("✅ Main window created")
        
        # Create floating button
        print("🔘 Creating floating button...")
        floating_button = FloatingAIButton()
        print("✅ Floating button created")
        
        # Connect button to window
        def on_mode_changed(mode):
            print(f"🔄 Mode changed to: {mode}")
            main_window.switch_mode(mode)
            if main_window.isVisible():
                main_window.raise_()
                main_window.activateWindow()
            else:
                main_window.show()
                main_window.raise_()
                main_window.activateWindow()
        
        floating_button.modeChanged.connect(on_mode_changed)
        
        # Show floating button
        floating_button.show()
        floating_button.raise_()
        print("✅ Floating button shown")
        
        # Initially hide the main window
        main_window.hide()
        
        print("\n" + "="*60)
        print("✅ AI ASSISTANT STARTED SUCCESSFULLY!")
        print("="*60)
        print("🔵 Look for a purple/blue floating button on your screen")
        print("🔵 Click the button to open the assistant")
        print("🔵 Right-click the button to switch between modes")
        print("🔵 Press Ctrl+C in terminal to quit")
        print("="*60 + "\n")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start application")
        print(f"❌ Error message: {e}")
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        print("\n💡 Common solutions:")
        print("   1. Ensure you're running in a graphical environment (not SSH without X11)")
        print("   2. Check DISPLAY variable: echo $DISPLAY")
        print("   3. Install PyQt5: pip install PyQt5")
        print("   4. Check requirements: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
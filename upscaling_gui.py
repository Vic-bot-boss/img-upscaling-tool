import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QLineEdit, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal, QProcess
import os

class UpscalingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.input_label = QLabel('Select input image:')
        layout.addWidget(self.input_label)

        self.input_button = QPushButton('Browse')
        self.input_button.clicked.connect(self.select_input_image)
        layout.addWidget(self.input_button)

        self.output_label = QLabel('Select output directory:')
        layout.addWidget(self.output_label)

        self.output_button = QPushButton('Browse')
        self.output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.output_button)

        self.scale_label = QLabel('Enter scale factor:')
        layout.addWidget(self.scale_label)

        self.scale_input = QLineEdit(self)
        self.scale_input.setText('4')
        layout.addWidget(self.scale_input)

        self.method_label = QLabel('Select upscaling method:')
        layout.addWidget(self.method_label)

        self.method_combo = QComboBox(self)
        self.method_combo.addItem("Real-ESRGAN")
        self.method_combo.addItem("Pillow")
        layout.addWidget(self.method_combo)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        self.run_button = QPushButton('Run Upscaling')
        self.run_button.clicked.connect(self.run_upscaling)
        layout.addWidget(self.run_button)

        self.setLayout(layout)
        self.setWindowTitle('Image Upscaling Tool')
        self.show()

    def select_input_image(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "All Files (*);;JPEG Files (*.jpg);;PNG Files (*.png)", options=options)
        if file:
            self.input_image_path = file
            self.input_label.setText(f'Input Image: {file}')

    def select_output_directory(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        if directory:
            self.output_directory = directory
            self.output_label.setText(f'Output Directory: {directory}')

    def run_upscaling(self):
        input_image = self.input_image_path
        output_image = f"{self.output_directory}/upscaled_image.png"
        scale_factor = self.scale_input.text()
        method = self.method_combo.currentText()

        self.status_label.setText('Upscaling in progress...')
        self.progress_bar.setValue(0)

        script_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        if method == "Real-ESRGAN":
            script_path = os.path.join(script_dir, "real_ESRGAN_upscaling.py")
        elif method == "Pillow":
            script_path = os.path.join(script_dir, "pillow_resizing.py")

        self.process = QProcess(self)
        self.process.finished.connect(self.upscaling_finished)
        self.process.start(sys.executable, [script_path, input_image, output_image, scale_factor])

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def upscaling_finished(self):
        self.progress_bar.setValue(100)
        self.status_label.setText('Upscaling finished!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UpscalingApp()
    sys.exit(app.exec_())

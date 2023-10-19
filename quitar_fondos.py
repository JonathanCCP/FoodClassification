import os
import sys
#pip install rembg
from rembg import remove
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QMessageBox
from PIL import Image

class BackgroundRemoverApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Quitar fondos')
        self.setGeometry(100, 100, 400, 200)

        self.input_label = QLabel('Carpeta de entrada:')
        self.input_button = QPushButton('Seleccionar carpeta de entrada', self)
        self.input_button.clicked.connect(self.select_input_folder)

        self.output_label = QLabel('Carpeta de salida:')
        self.output_button = QPushButton('Seleccionar carpeta de salida', self)
        self.output_button.clicked.connect(self.select_output_folder)

        self.process_button = QPushButton('Procesar imágenes', self)
        self.process_button.clicked.connect(self.process_images)

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_button)
        layout.addWidget(self.process_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.input_folder_path = None
        self.output_folder_path = None

    def select_input_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        self.input_folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de entrada", options=options)

    def select_output_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        self.output_folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida", options=options)

    def process_images(self):
        if self.input_folder_path is None or self.output_folder_path is None:
            return
        
        for input_file_name in os.listdir(self.input_folder_path):
            if input_file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(self.input_folder_path, input_file_name)
                output_path = os.path.join(self.output_folder_path, input_file_name)
                
                with open(input_path, "rb") as input_file:
                    output_data = remove(input_file.read())
                
                with open(output_path, "wb") as output_file:
                    output_file.write(output_data)
                
                # Abrir la imagen procesada y redimensionarla a 500x300 con fondo negro
                # Después de procesar la imagen con rembg
                processed_image = Image.open(output_path)
                processed_image = processed_image.resize((500, 300))

                # Convertir la imagen de RGBA a RGB si es necesario
                if processed_image.mode == 'RGBA':
                    processed_image = processed_image.convert('RGB')

                # Guardar la imagen en formato JPEG (sin canal alfa)
                output_path_jpeg = os.path.splitext(output_path)[0] + ".jpg"
                processed_image.save(output_path_jpeg, "JPEG")

                print(f"Fondo removido y la imagen {input_file_name} se ha guardado en {output_path_jpeg}")

        self.show_completion_alert()

    def show_completion_alert(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Proceso completado")
        msg_box.setText("El procesamiento de imágenes ha finalizado.")
        msg_box.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BackgroundRemoverApp()
    window.show()
    sys.exit(app.exec_())

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem, QFileDialog
from GenerarDataSet import GenerarDataset
from ClassifierComparison import clasifierComparison
from testing_model import doPredict

import os




class MyInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.test = []
        self.classNames = {1: 'Torta', 2: 'Flauta', 3: 'Gordita', 4: 'Tamal', 5: 'Pozole'}  # Nombres de clases
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 700, 400)  # Establecer posición y tamaño de la ventana

        layout = QVBoxLayout()

        # Primer campo: Generar DataSet
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("Generar DataSet")
        generate_dataset_button = QPushButton("Generar")
        generate_dataset_button.clicked.connect(self.generate_dataset)
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(generate_dataset_button)
        layout.addLayout(dataset_layout)
        

        # Segundo campo: Entrenar y Validar modelo
        train_layout = QHBoxLayout()
        train_label = QLabel("Entrenar y Validar modelo")
        train_button = QPushButton("Entrenar y Validar")
        train_button.clicked.connect(self.train_validate)
        train_layout.addWidget(train_label)
        train_layout.addWidget(train_button)
        mostrar_tabla = QPushButton("Ocultar/Mostrar")
        mostrar_tabla.clicked.connect(self.toggle_section)
        train_layout.addWidget(mostrar_tabla)
        train_seccion = QVBoxLayout()
        train_seccion.addLayout(train_layout)
        message_layout = QHBoxLayout()

        # Configurar la tabla con 3 filas y 3 columnas
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Algoritmo", "Precisión"])

        # Llenar la tabla con datos
        
        message_layout.addWidget(self.table)
        self.messageWidget = QWidget()
        self.messageWidget.setVisible(False)
        self.messageWidget.setLayout(message_layout)

        train_seccion.addWidget(self.messageWidget)
        layout.addLayout(train_seccion)
        

        # Tercer campo: Probar modelo
        test_layout = QHBoxLayout()
        test_label = QLabel("Probar modelo")
        test_button = QPushButton("Probar")
        test_button.clicked.connect(self.test_model)
        test_layout.addWidget(test_label)
        test_layout.addWidget(test_button)
        

        mostrar_tabla_test = QPushButton("Ocultar/Mostrar")
        mostrar_tabla_test.clicked.connect(self.toggle_section_test)
        test_layout.addWidget(mostrar_tabla_test)
        test_seccion = QVBoxLayout()
        test_seccion.addLayout(test_layout)
        message_layout_test = QHBoxLayout()

        # Configurar la tabla con 3 filas y 3 columnas
        self.table_test = QTableWidget()
        self.table_test.setColumnCount(2)
        self.table_test.setHorizontalHeaderLabels(["Muestra", "Clase"])

        # Llenar la tabla con datos
        
        message_layout_test.addWidget(self.table_test)
        self.messageWidget_test = QWidget()
        self.messageWidget_test.setVisible(False)
        self.messageWidget_test.setLayout(message_layout_test)

        test_seccion.addWidget(self.messageWidget_test)
        layout.addLayout(test_seccion)

        self.setLayout(layout)
        self.setWindowTitle("Clasificador de comida")

    def toggle_section_test(self):
        self.messageWidget_test.setVisible(not self.messageWidget_test.isVisible())

    def toggle_section(self):
        self.messageWidget.setVisible(not self.messageWidget.isVisible())

    def generate_dataset (self):
        print("generate dataset")

        if (GenerarDataset()):
            QMessageBox.information(self, "Éxito", "El dataset se ha generado exitosamente.")
        else:
            QMessageBox.critical(self, "Error", f"Error al generar el dataset")
            os.remove("comida.csv")

    def train_validate(self):
        print("train and validate")
        if os.path.exists("comida.csv"):
            result = clasifierComparison()  
            self.table.setRowCount(len(result))
            if result:
                self.table.setRowCount(len(result))
                self.table.setColumnCount(2) 

                for row, item_data in enumerate(result):
                    algoritmo, presicion = item_data
                    presicion_percentage = f"{presicion:.2%}"  #Convertir precisión a porcentaje
                    item_algoritmo = QTableWidgetItem(algoritmo)
                    item_presicion = QTableWidgetItem(presicion_percentage)
                    self.table.setItem(row, 0, item_algoritmo)
                    self.table.setItem(row, 1, item_presicion)
                self.messageWidget.setVisible(True)
            else:
                QMessageBox.critical(self, "Error", "Por algún motivo el entrenamiento falló")
        else:
            QMessageBox.critical(self, "Error", "No se encontro el archivo 'comida.csv'. Primero debe generar el DataSet")


    def test_model(self):
        print("test model")
        if os.path.exists("modelo_entrenado_LinearSVM.joblib"):
            path_images = self.select_image()
            if path_images:
                result = doPredict(path_images)
                print(result)
                if result:
                    correct_predictions = 0
                    total_predictions = len(result)
                    
                    self.table_test.setRowCount(total_predictions)
                    self.table_test.setColumnCount(2)
                    
                    #Los nombre de las imagenes deben tener el formato: clase_numero.jpg
                    for row, item_data in enumerate(result):
                        name = QTableWidgetItem(str(item_data[0]))
                        actual_class = name.text().split('_')[0]
                        predicted_class = QTableWidgetItem(str(item_data[1][0]))
                        self.table_test.setItem(row, 0, name)
                        self.table_test.setItem(row, 1, predicted_class)
                        
                        if actual_class == predicted_class.text():
                            correct_predictions += 1
                    
                    accuracy = correct_predictions / total_predictions
                    accuracy_percentage = accuracy * 100
                    
                    self.accuracy_label = QLabel(f"Precisión: {accuracy_percentage:.2f}%")
                    self.correct_predictions_label = QLabel(f"Predicciones correctas: {correct_predictions}/{total_predictions}")
                    
                    layout = self.messageWidget_test.layout()
                    layout.addWidget(self.accuracy_label)
                    layout.addWidget(self.correct_predictions_label)
                    
                    self.messageWidget_test.setVisible(True)
        else:
            QMessageBox.critical(self, "Error", "No se encontro el archivo 'modelo_entrenado.joblib'. Primero debe entrenar el modelo")


    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        input_folder_path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de entrada", options=options)

        if input_folder_path:
            return input_folder_path

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyInterface()
    window.show()
    sys.exit(app.exec_())



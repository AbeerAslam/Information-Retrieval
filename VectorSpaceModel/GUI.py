from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QPushButton, QTextEdit, QLabel, QListWidget, QListWidgetItem)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
import sys
import os
from query_processing import search_query

cwd = os.getcwd()
RAW_DATA_FOLDER = os.path.join(cwd, "Raw_Data")  # Update with actual path


class SearchEngineUI(QWidget):
    def __init__(self):
        super().__init__()
        self.result_display = None
        self.results_list = None
        self.search_bar = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Search Engine")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet("background-color: #F5F5F5;")

        layout = QVBoxLayout()

        # Search Label
        label = QLabel("Search", self)
        label.setFont(QFont("Arial", 24, QFont.Bold))
        label.setStyleSheet("color: #4285F4;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Search Bar + Button Layout
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search here...")
        self.search_bar.setFont(QFont("Arial", 14))
        self.search_bar.setStyleSheet("padding: 8px; border-radius: 10px; border: 2px solid #DADCE0;")
        search_layout.addWidget(self.search_bar)

        # Search Button
        search_button = QPushButton(self)
        search_button.setIcon(QIcon("search_icon.png"))
        search_button.setStyleSheet("border: none;")
        search_button.clicked.connect(self.perform_search)
        search_layout.addWidget(search_button)

        layout.addLayout(search_layout)

        self.results_list = QListWidget(self)
        self.results_list.setFont(QFont("Arial", 12))
        self.results_list.setStyleSheet("border-radius: 10px; background-color: white; border: 1px solid #DADCE0;")
        self.results_list.itemClicked.connect(self.display_document)
        layout.addWidget(self.results_list)

        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        self.result_display.setFont(QFont("Arial", 12))
        self.result_display.setStyleSheet(
            "padding: 8px; border-radius: 10px; background-color: white; border: 1px solid #DADCE0;")
        layout.addWidget(self.result_display)

        self.setLayout(layout)

        # Bind Enter key to trigger search
        self.search_bar.returnPressed.connect(self.perform_search)

    def perform_search(self):
        query = self.search_bar.text().strip()
        if not query:
            self.results_list.clear()
            self.result_display.setText("Please enter a query!")
            return

        results = search_query(query,'Stopword-List.txt',"Vectors/tfidf_vectors.json")
        self.results_list.clear()
        self.result_display.clear()

        if results:
            for doc_id in results:
                item = QListWidgetItem(f"Document {doc_id}")
                item.setData(Qt.UserRole, doc_id)
                self.results_list.addItem(item)
        else:
            self.results_list.addItem("No matching documents found.")

    def display_document(self, item):
        doc_id = item.data(Qt.UserRole)
        file_path = os.path.join(RAW_DATA_FOLDER, f"{doc_id[0]}")

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            self.result_display.setText(content)
        else:
            self.result_display.setText(f"Document {doc_id[0]} not found.")


def search_bar():
    app = QApplication(sys.argv)
    window = SearchEngineUI()
    window.show()
    sys.exit(app.exec_())

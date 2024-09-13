import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
import pickle

model = load_model('./cyberbullying_classifier_model.h5')

with open(r'D:\Train\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 50

label_mapping = {
    0: 'religion',
    1: 'age',
    2: 'gender',
    3: 'ethnicity',
    4: 'not_cyberbullying'
}

def classify_text(input_text):
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    probabilities = prediction[0]
    return label_mapping[predicted_class], probabilities

class CyberbullyingClassifierApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Cyberbullying Classifier')
        self.setGeometry(600, 200, 800, 600)

        self.layout = QVBoxLayout()
        self.label = QLabel("Enter your text to classify:")
        self.text_input = QLineEdit()
        self.classify_button = QPushButton("Classify")
        self.result_label = QLabel("The predicted class will appear here.")
        self.probabilities_label = QLabel("Class probabilities will appear here.")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.classify_button)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.probabilities_label)

        self.setLayout(self.layout)

        self.classify_button.clicked.connect(self.on_classify_button_click)

    def on_classify_button_click(self):
        input_text = self.text_input.text()
        if input_text.strip():
            predicted_label, probabilities = classify_text(input_text)
            probabilities_text = "\n".join([f"{label_mapping[i]}: {prob*100:.2f}%" for i, prob in enumerate(probabilities)])
            self.result_label.setText(f'The predicted class is: {predicted_label}')
            self.probabilities_label.setText(f'Class probabilities:\n{probabilities_text}')
        else:
            self.result_label.setText("Please enter some text to classify.")
            self.probabilities_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CyberbullyingClassifierApp()
    window.show()
    sys.exit(app.exec_())

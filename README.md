# Traffic Sign Recognition Project

## Overview

This project involves building a Traffic Sign Recognition system using machine learning techniques. The goal is to develop a model that can accurately identify traffic signs from images, which is a critical component for autonomous vehicles and advanced driver-assistance systems (ADAS).

## Exercise 
Abschlussprojekt: Verkehrszeichenerkennung
Überblick
In diesem Projekt werden Sie die Methoden, welche Sie in der Vorlesung kennengelernt haben, nutzen, um Verkehrszeichen zu klassifizieren.

Sie trainieren und validieren ein Modell, damit es Verkehrszeichen auf Basis des [Belgian Traffic Sign Dataset] (https://btsd.ethz.ch/shareddata/) klassifizieren kann.

Nachdem das Modell trainiert wurde, werden Sie Ihr Modell an Bildern von Verkehrszeichen ausprobieren, die Sie im Internet finden.

Wir haben eine einfache Basislinienimplementierung für die Erkennung mit einem Neuronalennetz mit TensorFlow in Python angehängt, die den Datensatz lädt aus dem Verzeichnis /hdd/data lädt

Teil der Abgabe ist auch, dass Sie eine detaillierte Beschreibung Ihrer Umsetzung erstellen. Sie können die von Udacity erstellte Schreibvorlage verwenden und als Ausgangspunkt für Ihre Ausarbeitung nutzen.

Ihre Einreichung sollte vier Dateien umfassen:

der Quelltext, entweder als Ipython-Notebook oder direkt als Python-Datei (bei Verwendung von Python, sonst Quelltext in anderen Programmiersprachen mit Verweis auf benötigte Pakete und Installationsanweisung für diese Pakete)
der als PDF exportierte Code
Einen gezippten Ordner mit Bildern weiterer Verkehrszeichen, die sie im Internet gefunden haben und zur weiteren Validierung nutzen
PDF Ihrer Ausarbeitung (Deutsch oder Englisch)
Bitte reichen sie Ihr Ergebnis als E-Mail ein unter ihrem Namen und Matrikelnummer ein. Alternativ ist auch der Verweis per E-Mail auf ein öffentliches Repository wie Github möglich, Dort bitte eine Datei MATRIKELNUMMER einchecken, um den Abgleich mit ihren Studieninfos zu ermöglichen.

Ausarbeitung
Ihre Ausarbeitung sollte nicht nur eine detaillierte Beschreibung Ihrer Umsetzung (ggf. mit Zeilennummernverweisen und Codeschnipseln) und der Qualität Ihrer unterschiedlichen Modelle beinhalten.

Zitieren sie auch wissenschaftliche Aufsätze, deren Methoden sie für ihre Umsetzung adaptiert haben und verweisen sie auf Webseiten nennen, wo sie Anregungen gefunden haben.

Sie sollten Bilder in Ihre Beschreibung aufnehmen, um zu demonstrieren, wie Ihr Code mit Beispielen von Verkehrszeichen funktioniert (was wurde richtig erkannt, was falsch…)

Ihre Ausarbeitung sollte ca. 15-20 Seiten umfassen. Ein Deckblatt mit Name und Matrikelnummer und Datum ist erforderlich. Die Ausarbeitung kann in Deutsch oder Englisch formuliert sein.

Teilaufgaben
Die Projektaufgaben lassen sich in folgende Teilschritte gliedern:

Laden des Datensatzes
Exploration, Zusammenfassung und Visualisierung des Datensatzes
Entwurf, Training und Test mehrer Modellarchitekturen
Nutzung der Modelle, um Vorhersagen für neue Bilder zu treffen, und die Prognosegüte zu ermitteln.
Zusammenfassung der Ergebnisse mit einem schriftlichen Bericht
Abgabetermin
Geben sie ihr Projekt spätestens am 8. August 2018 ab.

Datensatz
Verwenden Sie den auf dem Server in hdd/data/Training und hdd/data/Testing gespeicherten Datensatz oder laden Sie ihn direkt von der Website[Belgian Traffic Sign Dataset] (https://btsd.ethz.ch/shareddata/) auf Ihren Computer herunter.

Logins für den Server erhalten sie separat.

Anmerkung
Ihre Einreichung wird elektronisch auf Plagiate überprüft.

## Features

- **Image Preprocessing**: Techniques to enhance image quality and normalize data.
- **Model Training**: Using convolutional neural networks (CNN) to train on traffic sign datasets.
- **Model Evaluation**: Assessing the model's performance using accuracy, precision, recall, and F1-score.
- **Real-Time Recognition**: Implementing real-time traffic sign recognition using a webcam.

## Dataset

The project uses the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset, which contains over 50,000 images of 43 different traffic sign classes.

## Project Structure

- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, model training, and evaluation.
- `src/`: Source code for the project.
  - `preprocessing.py`: Code for data preprocessing.
  - `model.py`: Definition of the CNN model.
  - `train.py`: Script for training the model.
  - `evaluate.py`: Script for evaluating the model.
  - `realtime.py`: Code for real-time traffic sign recognition using webcam.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies required to run the project.

## Installation

To run this project, you need to have Python 3.x installed. Follow the steps below to set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the GTSRB dataset and place it in the `data/` directory.

## Usage

### Training the Model

To train the model, run the following command:
```bash
python src/train.py
```
This script will preprocess the data, train the CNN model, and save the trained model to the `models/` directory.

### Evaluating the Model

To evaluate the trained model, use the following command:
```bash
python src/evaluate.py
```
This script will load the trained model and print out the evaluation metrics.

### Real-Time Recognition

To perform real-time traffic sign recognition using your webcam, run:
```bash
python src/realtime.py
```
This script will activate your webcam and start recognizing traffic signs in real-time.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-branch`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- The [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) for providing the traffic sign images.
- The open-source community for the tools and libraries used in this project.

## Contact

For any questions or suggestions, please contact [your-email@example.com](mailto:your-email@example.com).

---

Thank you for using the Traffic Sign Recognition Project! We hope it helps you in your learning journey and contributes to your success.

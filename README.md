# Traffic Sign Recognition Project

## Overview

This project involves building a Traffic Sign Recognition system using machine learning techniques. The goal is to develop a model that can accurately identify traffic signs from images, which is a critical component for autonomous vehicles and advanced driver-assistance systems (ADAS).

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

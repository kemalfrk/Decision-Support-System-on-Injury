# Fracture Detection and Injury Recovery Decision Support System

This project is a demo of an AI-powered decision support system designed to detect fractures in bone X-rays and help accelerate the recovery process after injuries. The system uses the YOLOv8 deep learning model and Streamlit to automatically detect fractures in bone X-ray images, providing doctors with a user-friendly interface to assist in the diagnosis.

## Project Overview

The AI-based decision support system aims to help doctors quickly detect fractures and make better decisions to speed up the recovery process after injuries. The project consists of the following main components:

- **YOLOv8 Model**: Used to detect fractures in bone X-ray images. The model is trained on a dataset to accurately identify fractured areas.
- **Streamlit Interface**: A user-friendly web application allowing users to upload X-ray images and view results instantly.
- **Decision Support Mechanism**: Based on detected fractures, the system provides suggestions to doctors for speeding up the recovery process.

## Validation Score

![val_result](https://github.com/user-attachments/assets/3aec4cf4-75a7-4fe7-93b6-e425289e1d1c)


## Technologies Used

- **Python**: For model training and application development.
- **YOLOv8**: Object detection model for fracture detection.
- **Streamlit**: For developing the web-based interface.
- **OpenCV**: For image processing and analysis.

## Installation and Usage

Clone the project repository to get started:




  ` git clone https://github.com/kemalfrk/Decision-Support-System-on-Injury.git`

`cd Decision-Support-System-on-Injury` : Go to the directory where the project is located


Install the required packages:

`pip install -r requirements.txt`

`pip install ultralytics`


`streamlit run app.py` :This is the command you'd run in the terminal to launch the Streamlit application.

# Demo



https://github.com/user-attachments/assets/05091ef5-aca1-472a-8d44-cb11ba9e96cb






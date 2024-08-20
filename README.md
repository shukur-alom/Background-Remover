# Background Removal with SAM Model using Gradio

![visualization](https://github.com/shukur-alom/Background-Remove/blob/main/Media/visual.gif)

This project is a simple Gradio-based application that allows users to select regions of interest (ROI) in an image and remove the background from those selected regions using the Segment Anything Model (SAM). The processed image can then be downloaded.

## Features

- **Image Upload:** Users can upload an image through the Gradio interface.
- **Region Selection:** Users can click on the image to select regions of interest (ROI).
- **Background Removal:** The application uses a pre-trained SAM model to remove the background from the selected region.
- **Image Download:** After processing, users can download the image with the background removed.

## How to Use

1. **Upload Image:** Click on the "Click" area to upload an image.
2. **Select Region:** Click on the image to mark points for selecting the region of interest (ROI). Double-clicking marks the final region.
3. **Perform Action:** Click the "Perform Action" button to remove the background from the selected region.
4. **Download Image:** After processing, you can download the image by clicking the download button.

## Requirements

- Python 3.x
- Gradio
- OpenCV
- NumPy
- Ultralytics SAM Model

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Running the Application


This version now specifies that at least 8 GB of RAM is needed to run the project efficiently.

To run the application locally, use the following command:

```bash
python app.py
```

The application will launch in your browser.


## Model

The project uses the Segment Anything Model (SAM) from Ultralytics, which is designed for object segmentation. 
The pre-trained model sam2_l.pt is used in this project.
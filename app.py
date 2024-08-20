import gradio as gr
import cv2
import numpy as np

ROI_coordinates = {
    'x_temp': 0,
    'y_temp': 0,
    'x_new': 0,
    'y_new': 0,
    'clicks': 0,
}

selected_sections = []
selected_image = None

def get_select_coordinates(img, evt: gr.SelectData):
    global selected_sections, selected_image
    sections = []
    selected_image = img
    # update new coordinates
    ROI_coordinates['clicks'] += 1
    ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['y_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['x_new'] = evt.index[0]
    ROI_coordinates['y_new'] = evt.index[1]
    # compare start end coordinates
    x_start = ROI_coordinates['x_new'] if (
        ROI_coordinates['x_new'] < ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
    y_start = ROI_coordinates['y_new'] if (
        ROI_coordinates['y_new'] < ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    x_end = ROI_coordinates['x_new'] if (
        ROI_coordinates['x_new'] > ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
    y_end = ROI_coordinates['y_new'] if (
        ROI_coordinates['y_new'] > ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    if ROI_coordinates['clicks'] % 2 == 0:
        sections.append(
            ((x_start, y_start, x_end, y_end), "Object Selected"))
        selected_sections = sections
        return (img, sections)
    else:
        point_width = int(img.shape[0]*0.05)
        sections.append(((ROI_coordinates['x_new'], ROI_coordinates['y_new'],
                          ROI_coordinates['x_new'] + point_width, ROI_coordinates['y_new'] + point_width),
                        "Object not Selected"))
        return (img, sections)

def button_action():
    global selected_sections, selected_image
    if selected_sections and selected_image is not None:
        print("Button clicked! Performing action with selected sections.")
        # Process the image (e.g., draw a rectangle around the selected ROI)
        x_start, y_start, x_end, y_end = selected_sections[0][0]
        processed_img = cv2.rectangle(np.array(selected_image), (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        return processed_img, "Action performed with selected sections."
    else:
        return selected_image, "No sections selected."

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Click")
        img_output = gr.AnnotatedImage(label="ROI",
                                       color_map={"ROI of Face Detection": "#9987FF",
                                                  "Click second point for ROI": "#f44336"})
    input_img.select(get_select_coordinates, input_img, img_output)
    
    action_button = gr.Button("Perform Action")
    processed_img_output = gr.Image(label="Processed Image")
    action_output = gr.Textbox(label="Action Output")
    action_button.click(button_action, outputs=[processed_img_output, action_output])

if __name__ == '__main__':
    demo.launch(inbrowser=True)
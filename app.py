import gradio as gr
import cv2
import numpy as np
from ultralytics import SAM

model = SAM("sam2_l.pt")

ROI_coordinates = {
    'x_temp': 0,
    'y_temp': 0,
    'x_new': 0,
    'y_new': 0,
    'clicks': 0,
}

selected_sections = []
selected_image = None


def remove_background(img, boxes):
    results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bboxes=boxes)
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for result in results:
        for res in result:
            mask = np.array(res.masks.xy[0])
            cv2.fillPoly(combined_mask, np.int32([mask]), 255)

    background_mask = cv2.bitwise_not(combined_mask)
    img_no_background = cv2.bitwise_and(img, img, mask=combined_mask)
    background = np.full_like(img, 255)
    background = cv2.bitwise_and(background, background, mask=background_mask)
    final_img = cv2.add(img_no_background, background)
    return final_img


def get_select_coordinates(img, evt: gr.SelectData):
    global selected_sections, selected_image
    sections = []
    selected_image = img
    ROI_coordinates['clicks'] += 1
    ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['y_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['x_new'] = evt.index[0]
    ROI_coordinates['y_new'] = evt.index[1]
    x_start = ROI_coordinates['x_new'] if (
        ROI_coordinates['x_new'] < ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
    y_start = ROI_coordinates['y_new'] if (
        ROI_coordinates['y_new'] < ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    x_end = ROI_coordinates['x_new'] if (
        ROI_coordinates['x_new'] > ROI_coordinates['x_temp']) else ROI_coordinates['x_temp']
    y_end = ROI_coordinates['y_new'] if (
        ROI_coordinates['y_new'] > ROI_coordinates['y_temp']) else ROI_coordinates['y_temp']
    if ROI_coordinates['clicks'] % 2 == 0:
        sections.append(((x_start, y_start, x_end, y_end), "Object Selected"))
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
        x_start, y_start, x_end, y_end = selected_sections[0][0]

        processed_img_ = remove_background(
            selected_image, [x_start, y_start, x_end, y_end])

        cv2.imwrite("processed_image.png", cv2.cvtColor(
            processed_img_, cv2.COLOR_BGR2RGB))

        return processed_img_, "processed_image.png"


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Click")
        img_output = gr.AnnotatedImage(label="ROI",
                                       color_map={"Object Selected": "#9987FF",
                                                  "Object not Selected": "#f44336"})
    input_img.select(get_select_coordinates, input_img, img_output)

    action_button = gr.Button("Perform Action")
    processed_img_output = gr.Image(label="Processed Image", image_mode="RGB")
    download_button = gr.File(label="Download Processed Image")

    action_button.click(button_action, outputs=[
                        processed_img_output, download_button])

if __name__ == '__main__':
    demo.launch(inbrowser=True)

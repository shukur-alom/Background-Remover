import gradio as gr

ROI_coordinates = {
    'x_temp': 0,
    'y_temp': 0,
    'x_new': 0,
    'y_new': 0,
    'clicks': 0,
}


def get_select_coordinates(img, evt: gr.SelectData):
    sections = []
    # update new coordinates
    ROI_coordinates['clicks'] += 1
    ROI_coordinates['x_temp'] = ROI_coordinates['x_new']
    ROI_coordinates['y_temp'] = ROI_coordinates['y_new']
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
            ((x_start, y_start, x_end, y_end), "ROI of Face Detection"))
        print(f"ROI coordinates: {x_start, y_start, x_end, y_end}")
        return (img, sections)
    else:
        point_width = int(img.shape[0]*0.05)
        sections.append(((ROI_coordinates['x_new'], ROI_coordinates['y_new'],
                          ROI_coordinates['x_new'] + point_width, ROI_coordinates['y_new'] + point_width),
                        "Click second point for ROI"))
        return (img, sections)


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Click")
        img_output = gr.AnnotatedImage(label="ROI",
                                       color_map={"ROI of Face Detection": "#9987FF",
                                                  "Click second point for ROI": "#f44336"})
    input_img.select(get_select_coordinates, input_img, img_output)


if __name__ == '__main__':
    demo.launch(inbrowser=True)

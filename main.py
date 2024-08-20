import cv2
from ultralytics import SAM
import numpy as np

model = SAM("sam2_l.pt")

start_point = None
end_point = None
drawing = False
boxes = []
points = []

# Mouse callback function

def get_points(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        img_copy = img.copy()

        results = model(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB),
                        points=points, labels=[1 for i in range(len(points))])
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for result in results:
            for res in result:
                mask = np.array(res.masks.xy[0])
                cv2.polylines(img_copy, np.int32([mask]), True, (0, 0, 255), 1)

                cv2.fillPoly(combined_mask, np.int32([mask]), 255)

        # Invert the mask to get the background
        background_mask = cv2.bitwise_not(combined_mask)

        # Apply the mask to the original image
        img_no_background = cv2.bitwise_and(img, img, mask=combined_mask)

        # Optionally, you can set the background to a specific color (e.g., white)
        background = np.full_like(img, 255)
        background = cv2.bitwise_and(
            background, background, mask=background_mask)

        # Combine the foreground and the background
        final_img = cv2.add(img_no_background, background)

        cv2.imshow('image', img_copy)
        cv2.imshow('final_img', final_img)
        cv2.imwrite('output.png', img=cv2.resize(final_img, (w_, h_)))


def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, img, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            img_copy = img.copy()
            cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        boxes.append([start_point[0], start_point[1],
                     end_point[0], end_point[1]])

        img_copy = img.copy()
        cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)

        results = model(cv2.cvtColor(
            img_copy, cv2.COLOR_BGR2RGB), bboxes=boxes)
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for result in results:
            for res in result:
                mask = np.array(res.masks.xy[0])
                cv2.polylines(img_copy, np.int32([mask]), True, (0, 0, 255), 1)

                cv2.fillPoly(combined_mask, np.int32([mask]), 255)

        # Invert the mask to get the background
        background_mask = cv2.bitwise_not(combined_mask)

        # Apply the mask to the original image
        img_no_background = cv2.bitwise_and(img, img, mask=combined_mask)

        # Optionally, you can set the background to a specific color (e.g., white)
        background = np.full_like(img, 255)
        background = cv2.bitwise_and(
            background, background, mask=background_mask)

        # Combine the foreground and the background
        final_img = cv2.add(img_no_background, background)

        cv2.imshow('image', img_copy)
        cv2.imshow('final_img', final_img)
        cv2.imwrite('Output/output.png', img=cv2.resize(final_img, (w_, h_)))


# Load an image
img = cv2.imread('data/kamran-ch-5M_SaAMIlZQ-unsplash.jpg')
h_, w_ = img.shape[:2]
print(h_, w_)

img = cv2.resize(img, (640, 680))
cv2.imshow('image', img)

cv2.setMouseCallback('image', draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

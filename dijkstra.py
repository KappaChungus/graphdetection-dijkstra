import subprocess
import sys

import cv2
import pytesseract
import numpy as np
import os
from scipy.spatial import distance

pytesseract.pytesseract.tesseract_cmd = r"tesseract.exe"


def detect_graph(image_path, mode='xd'):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, (100, 150, 50), (140, 255, 255))  # HSV range for blue
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if mode == 'debug':
        print(f"Number of detected blue contours: {len(blue_contours)}")

    vertices = []
    for i, contour in enumerate(blue_contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10 and 0.7 < w / h < 1.4:
            vertices.append((x, y, w, h))

    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))  # Lower red
    red_mask += cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))  # Upper red
    vertex_names = []

    if mode == 'debug':
        output_dir = "training_data"
        os.makedirs(output_dir, exist_ok=True)

    for idx, (x, y, w, h) in enumerate(vertices):

        region_left = x - 4
        region_bottom = y
        region_top = max(0, region_bottom - 19)
        region_right = min(region_left + 18, img.shape[1])

        label_region = red_mask[region_top:region_bottom, region_left:region_right]
        label_region_inverted = cv2.bitwise_not(label_region)
        if mode == 'debug':
            label_image_path = os.path.join(output_dir, f"vertex_{x}_{y}.png")
            cv2.imwrite(label_image_path, label_region_inverted)  # Save the inverted image

        # OCR to extract the label text with confidence score
        config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ \
-c tessedit_single_char_block=1 \
-c segment_penalty_garbage=10 \
-c classify_bln_numeric_mode=1"
        result = pytesseract.image_to_data(label_region, config=config, output_type=pytesseract.Output.DICT)
        highest_confidence = -1
        best_text = ''
        for i, word in enumerate(result['text']):
            if word.strip():  # Ignore empty strings
                conf = int(result['conf'][i])
                if conf > highest_confidence:
                    highest_confidence = conf
                    best_text = word

        if best_text == '':
            best_text = 'I'
        if best_text == 'CG':
            best_text = 'G'
        if mode == 'debug':
            print(
                f"Detected text for vertex at ({x}, {y}) with highest confidence: '{best_text}' (confidence: {highest_confidence})")
            print(best_text)
        vertex_names.append(best_text)
    if mode == 'debug':
        for i, (x, y, w, h) in enumerate(vertices):
            print(f"Vertex at ({x}, {y}) with name: {vertex_names[i]}")

    detect_edges(image_path, vertices, vertex_names, mode)

    return vertices, vertex_names

def extract_red_regions(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    return red_mask


def detect_edges(image_path, vertices, vertex_names, mode):
    img = cv2.imread(image_path)
    color_img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 190])
    upper_gray = np.array([180, 10, 200])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    if mode == 'debug':
        cv2.imwrite("extracted_edges.png", cleaned_mask)

    red_mask = extract_red_regions(image_path)

    height, width = cleaned_mask.shape
    one_neighbor_pixels = []
    neighbor_offsets = [
        (-1, 0),  # Up
        (-1, 1),  # Up-right
        (0, 1),  # Right
        (1, 1),  # Right-bottom
        (1, 0),  # Bottom
        (1, -1),  # Bottom-left
        (0, -1),  # Left
        (-1, -1)  # Left-up
    ]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if cleaned_mask[y, x] == 255:
                white_neighbors = 0
                for offset_y, offset_x in neighbor_offsets:
                    neighbor_y = y + offset_y
                    neighbor_x = x + offset_x
                    if cleaned_mask[neighbor_y, neighbor_x] == 255:
                        white_neighbors += 1
                if white_neighbors == 1:
                    one_neighbor_pixels.append((y, x))
    one_neighbor_pixels.sort(key=lambda p: p[1])
    if mode == 'debug':
        print(f"Sorted positions of pixels with exactly one white neighbor:\n{one_neighbor_pixels}")

    with open("detected_graph.txt", "w") as file:
        for pixel in one_neighbor_pixels:
            end_pixel = find_end_of_edge(cleaned_mask, pixel)
            y1, x1 = pixel
            y2, x2 = end_pixel
            color_img[y1, x1] = (0, 0, 255)
            color_img[y2, x2] = (255, 0, 0)
            segment_length = distance.euclidean((x1, y1), (x2, y2))

            # Skip short segments that are less than 22 pixels long
            if segment_length < 22:
                continue
            if mode == 'debug':
                print(f"{pixel} {end_pixel}")

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            edge_roi = red_mask[center_y - 14:center_y + 14, center_x - 14:center_x + 19]
            if mode == 'debug':
                debug_path = f"edge_roi_{y1}_{x1}_{y2}_{x2}.png"
                cv2.imwrite(debug_path, edge_roi)

            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            result = pytesseract.image_to_data(edge_roi, config=config, output_type=pytesseract.Output.DICT)

            highest_confidence = -1
            best_weight = None
            for i, text in enumerate(result['text']):
                if text.strip().isdigit():
                    confidence = int(result['conf'][i])
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_weight = int(text)

            if best_weight is None:
                best_weight = 11
                if mode == 'debug':
                    print("No weight detected for edge:")

            # Find closest vertices to start and end points
            closest_start_vertex = min(
                vertices,
                key=lambda v: distance.euclidean((v[0] + v[2] // 2, v[1] + v[3] // 2), (x1, y1))
            )
            closest_end_vertex = min(
                vertices,
                key=lambda v: distance.euclidean((v[0] + v[2] // 2, v[1] + v[3] // 2), (x2, y2))
            )

            # Get the names of the closest vertices
            start_vertex_index = vertices.index(closest_start_vertex)
            end_vertex_index = vertices.index(closest_end_vertex)

            start_vertex_name = vertex_names[start_vertex_index]
            end_vertex_name = vertex_names[end_vertex_index]

            if best_weight == 1 and highest_confidence < 60:
                best_weight = 11
            if mode == 'debug':
                print(f"{start_vertex_name} {end_vertex_name} {best_weight} {highest_confidence}")
            else:
                file.write(f"{start_vertex_name} {end_vertex_name} {best_weight}\n")

    if mode == 'debug':
        cv2.imwrite("visualized_edges.png", color_img)

    return one_neighbor_pixels


def find_end_of_edge(image, start_pixel):
    origin = start_pixel  # Keep track of the origin for angle calculations
    last_direction = None  # To track whether we moved up or down last time
    desired_angle = None

    while True:
        if last_direction == 'down':
            directions = [(1, 0), (1, 1), (0, 1)]  # down/right
        elif last_direction == 'up':
            directions = [(-1, 0), (-1, 1), (0, 1)]  # up/right
        else:
            directions = [(-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)]  # right

        next_pixel, desired_angle = find_in_direction(image, start_pixel, origin, directions, desired_angle)

        if next_pixel is None:
            return start_pixel

        if next_pixel[0] > start_pixel[0]:
            last_direction = 'down'
        elif next_pixel[0] < start_pixel[0]:
            last_direction = 'up'

        start_pixel = next_pixel


import math


def find_in_direction(image, current_pixel, origin, directions, desired_angle=None):
    """
    Finds the next white pixel in the specified directions based on angle alignment.
    If there are multiple candidates, it will prioritize maintaining the initial direction angle.
    """
    candidates = []
    y, x = current_pixel

    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1] and image[ny, nx] == 255:
            candidates.append((ny, nx))

    if not candidates:
        return None, None

    if len(candidates) == 1:
        return candidates[0], desired_angle

    if desired_angle is None:
        # First time we encounter multiple candidates, calculate the current angle and remember it
        current_dx = x - origin[1]
        current_dy = y - origin[0]
        current_angle = math.atan2(current_dy, current_dx) if current_dx or current_dy else 0

        desired_angle = current_angle  # Remember the initial angle

    # Calculate angles for all candidates and select the one closest to the desired_angle
    best_candidate = None
    smallest_angle_diff = float('inf')

    for candidate in candidates:
        candidate_dx = candidate[1] - origin[1]
        candidate_dy = candidate[0] - origin[0]
        candidate_angle = math.atan2(candidate_dy, candidate_dx)
        angle_diff = abs(candidate_angle - desired_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

        if angle_diff < smallest_angle_diff:
            smallest_angle_diff = angle_diff
            best_candidate = candidate

    return best_candidate, desired_angle


if len(sys.argv)!=3:
    subprocess.run(["dijkstra.exe", "detected_graph.txt"])
else:
    print('generating detected_graph.txt from image: ', sys.argv[1], '... please wait...\n')
    detect_graph(sys.argv[1])
    print('Check if content of detected_graph.txt matches with image!\n'
          'If it is incorrect create it manually and run this script again with no arguments.\n')
    subprocess.run(["dijkstra.exe", "detected_graph.txt", sys.argv[2]])

import json
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

def draw_box_text(draw, font, left, right, top, bottom, display_str_list):
    draw.line([(left, top), (left, bottom), (right, bottom),(right, top), (left, top)], width=4, fill='blue')
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), 
            (left + text_width,text_bottom)],fill='white')
        draw.text((left + margin, text_bottom - text_height - margin),
            display_str,fill='black',font=font)
        text_bottom -= text_height - 2 * margin

def process(category_index, image, data, draw_boxes=False, save_it=False):
    result = dict()
    max_boxes_to_draw=20
    min_score_thresh=.5
  
    (boxes, scores, classes) = data

    im_width, im_height = image.size

    scores = np.reshape(scores, [-1])
    boxes = np.reshape(boxes, [-1, 4])
    classes = np.reshape(classes, [-1])

    if boxes.shape[0] < max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()
    
    count = 0
    for i in range(max_boxes_to_draw):
        if (scores[i] > min_score_thresh) and (classes[i] in category_index.keys()):

            box = tuple(boxes[i].tolist())
            class_name = category_index[classes[i]]['name']
            display_str = str(class_name)

            ymin, xmin, ymax, xmax  = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            
            if draw_boxes:
                draw_box_text(draw, font, left, right, top, bottom, [display_str])
            
            payload = dict()
            payload["label"] = class_name
            bbox = dict()
            bbox["left"] = str(int(left))
            bbox["top"] = str(int(top))
            bbox["right"] = str(int(right))
            bbox["bottom"] = str(int(bottom))
            box_ref = bbox
            payload["box"] = box_ref
            result[str(i)] = payload
            count += 1
    
    if save_it:
        image.save('/var/www/html/images/out.jpg')

    return result, str(count)

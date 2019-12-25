#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np

def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45
    webcam_scale = 1.5

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'], compile=False)

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' == input_path[:6]: # do detection on the Xth webcam given the parameter 'webcamX'
        video_reader = cv2.VideoCapture(int(input_path[6:]))

        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True:
                images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh) 
                    
                    if webcam_scale != 1
                        img_shape = images[i].shape
                        images[i] = cv2.resize(images[i], dsize=(int(img_shape[1] * webcam_scale), int(img_shape[0] * webcam_scale)), interpolation=cv2.INTER_NEAREST)

                    images[i] = draw_receipt(images[i], batch_boxes[i], config['model']['labels'], config['entrees'], obj_thresh) 
                    cv2.imshow('Chinese Cafeteria Food Recognition', images[i])
                images = []
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif input_path[-4:] == '.mp4': # do detection on a video
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'MPEG'), 50.0, (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   
                        images[i] = draw_receipt(images[i], batch_boxes[i], config['model']['labels'], config['entrees'], obj_thresh)
                        # show the video with detection bounding boxes          
                        if show_window:
                            cv2.imshow('Chinese Cafeteria Food Recognition', images[i])

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27:
                    break  # esc to quit

        if show_window:
            cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
            image = draw_receipt(image, boxes, config['model']['labels'], config['entrees'], obj_thresh)
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))

def draw_receipt(image, boxes, labels, entrees, obj_thresh):
    row_height = int(35 * 1e-3 * image.shape[0])
    x = image.shape[1] + 10
    y = row_height

    # get detected_entrees
    detected_entrees = set()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                detected_entrees.add(labels[i])

    image_with_receipt = cv2.copyMakeBorder(image, 0, 0, 0, int(510 * 1e-3 * image.shape[0]), cv2.BORDER_CONSTANT, (0, 0, 0))

    # draw total expense
    total_exp = 0
    cv2.putText(img=image_with_receipt, text='====== Expense ======', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
    if(len(detected_entrees) > 0):
        y += row_height
        for detected_entree in detected_entrees:
            price = entrees[detected_entree]['price']
            total_exp += price
            cv2.putText(img=image_with_receipt, text='{0}'.format(detected_entree), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
            cv2.putText(img=image_with_receipt, text='{0:>29}'.format('$' + str(price)), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
            y += row_height
    y += row_height
    
    cv2.putText(img=image_with_receipt, text='{0}'.format('TOTAL'), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=5)
    cv2.putText(img=image_with_receipt, text='{0:>29}'.format('$' + str(total_exp)), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=3)
    
    # draw nutrition facts
    total_fat = 0
    total_carbs = 0
    total_protein = 0
    y += row_height * 2
    cv2.putText(img=image_with_receipt, text='==== Nutrition Facts ====', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
    if(len(detected_entrees) > 0):
        # header
        y += row_height
        cv2.putText(img=image_with_receipt, text='{0:>9} {1:>9} {2:>9}'.format('Fat', 'Carbs', 'Protein'), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=3)
        
        y += row_height
        for detected_entree in detected_entrees:
            nutrition_fact = entrees[detected_entree]
            total_fat += nutrition_fact['fat']
            total_carbs += nutrition_fact['carbohydrate']
            total_protein += nutrition_fact['protein']
            cv2.putText(img=image_with_receipt, text='{0}({1}{2})'.format(detected_entree, str(nutrition_fact['serving_amount']), nutrition_fact['amount_unit']), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
            y += row_height
            cv2.putText(img=image_with_receipt, text='{0:>9}{1:>9}{2:>9}'.format(str(nutrition_fact['fat']) + 'g', str(nutrition_fact['carbohydrate']) + 'g', str(nutrition_fact['protein']) + 'g'), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
            y += row_height
    y += row_height
    
    cv2.putText(img=image_with_receipt, text='{0}'.format('TOTAL'), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=5)
    y += row_height
    cv2.putText(img=image_with_receipt, text='{0:>9}{1:>9}{2:>9}'.format(str(round(total_fat, 1)) + 'g', str(round(total_carbs, 1)) + 'g', str(round(total_protein, 1)) + 'g'), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=2)
    y += row_height
    cv2.putText(img=image_with_receipt, text='{0}'.format(' Calories'), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=3)
    total_cal = 9 * total_fat + 4 * total_carbs + 4 * total_protein
    cv2.putText(img=image_with_receipt, text='{0:>29}'.format(round(total_cal, 1)), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0, 255, 0), thickness=3)
    
    return image_with_receipt

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)

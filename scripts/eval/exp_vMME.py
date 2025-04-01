import os
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

from tqdm import tqdm
import pandas as pd
import json
import argparse
import time

from transformers import CLIPModel, CLIPProcessor, CLIPTextConfig
from tqdm import tqdm
from peft import PeftModel, PeftConfig

warnings.filterwarnings("ignore")
# Load the base model
pretrained = "models/{version}"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = None, None, None, None

prompt = "{question}. Please choose the right answer from the following options: {options}."

clip_processor = CLIPProcessor.from_pretrained("clip-ViT-B-32/0_CLIPModel")

def access_video_by_sec(video_path, max_frame_num=3600):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    frames = []
    frame_count = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            ret, frame = cap.retrieve()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame_rgb)
            frames.append(frame)
        frame_count += 1
    cap.release()
    if len(frames) > max_frame_num:
        sampled_indices = np.linspace(0, len(frames)-1, max_frame_num, dtype=int)
        frames = [frames[i] for i in sampled_indices]
    return frames

def load_dataset(path):
    df = pd.read_parquet(path)
    dataset = []
    fields = ['duration', 'domain', 'sub_category', 'videoID', 'task_type', 'question', 'options', 'answer']
    for idx, row in df.iterrows():
        data = {k:row[k] for k in fields}
        data['options'] = list(data['options'])
        dataset.append(data)
    return dataset

def inference(PIL_images, query, question):
    video_frames = [np.array(img) for img in PIL_images]
    video_frames = np.array(video_frames)

    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors.append(frames)

    image_inputs = clip_processor(images=PIL_images, return_tensors='pt').to(device)
    query_inputs = clip_processor(text=[question], return_tensors='pt', truncation=True, max_length=CLIPTextConfig().max_position_embeddings).to(device)
    kwargs = {'image_inputs': [image_inputs], 'query_inputs': [query_inputs]}
    kwargs['offload_params'] = list(kwargs.keys())

    # Prepare conversation input
    conv_template = "qwen_1_5"
    query = "{DEFAULT_IMAGE_TOKEN}\n{query}".format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, query=query)
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1,
        modalities=["video"],
        **kwargs
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--dataset_path', default='dataset/Video-MME/videomme/test-00000-of-00001.parquet')
    arg_parser.add_argument('--video_dir', default='dataset/Video-MME/data')
    arg_parser.add_argument('--output_dir', default='dataset/Video-MME/output')
    arg_parser.add_argument('--version', default='ViLAMP-llava-qwen')
    arg_parser.add_argument('--split', default='1_1')
    arg_parser.add_argument('--max_frame_num', type=int, default=600)

    args = arg_parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pretrained = pretrained.format(version=args.version)
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="flash_attention_2")
    model.eval()
    model.to('cuda')

    args.output_file = os.path.join(args.output_dir, '{}-{}.json'.format(args.version, args.split))
    print('LOADING DATASET...')
    dataset = load_dataset(args.dataset_path)

    cur_id, chunk_nums = args.split.split('_')
    cur_id = int(cur_id)
    chunk_nums = int(chunk_nums)
    dataset = [dataset[i] for i in range(0, len(dataset)) if i % chunk_nums == cur_id-1]

    print('DATASET LENGTH: {}'.format(len(dataset)))

    with open(args.output_file, 'w') as f:
        for data in tqdm(dataset):
            question = data['question']
            options = data['options']
            query = prompt.format(question=question, options=options)
            videoID = data['videoID']
            video_path = os.path.join(args.video_dir, '{videoID}.mp4'.format(videoID=videoID))
            try:
                frames = access_video_by_sec(video_path, max_frame_num=args.max_frame_num)
                response = inference(frames, query, question)
            except Exception as e:
                print(e)
                response = '[E]' + str(e)

            obj = {'videoID': videoID, 'duration': data['duration'], 'question': question, 'options': options, 'answer': data['answer'], 'response': response}

            f.write(json.dumps(obj, ensure_ascii=False))
            f.write('\n')

**Below are the download links for the datasets used during the training of the ViLAMP model, as well as the data processing scripts.**

## Webvid
### URL
https://huggingface.co/datasets/sailvideo/webvid10m
### Data Scale
10M caption data, 2681 tar archives. Each archive contains several `<video-id>.json` and `<video-id>.mp4` pairs.
### Processing
```bash
tar -vxf *.tar
```
```python
import os
import json
import random
files = os.listdir('./')
files = [file for file in files if '.json' in file]
with open('filtered_train.json', 'r') as f:
    data = json.load(f)
    prompts = [_['QA'][0]['i'] for _ in data]
dataset = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
    inst = random.sample(prompts, 1)[0]
    conversations = [
        {'from': 'human', 'value': '<image>\n{inst}'.format(inst=inst)},
        {'from': 'gpt', 'value': data['text']}
    ]
    obj = {
        'conversations': conversations,
        'video': 'webvid/data/{}.mp4'.format(data['<ID>']),
        'query': inst
    }
    dataset.append(obj)
```

[filtered_train.json](https://www.modelscope.cn/datasets/AI-ModelScope/webvid-caption/resolve/master/filtered_train.json)

## InternVid
### URL
https://huggingface.co/datasets/OpenGVLab/InternVid
### Data Scale
10M, it is recommended to use the `video_clip` function to extract the desired segment immediately after downloading a video to reduce storage space usage.
### Processing
```python
from moviepy.editor import VideoFileClip

def time_cvt(time_str):
    """Convert time string into seconds (Format supported - HH:MM:SS、MM:SS or SS）"""
    parts = list(map(float, time_str.split(':')))
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 1:
        return parts[0]
    else:
        raise ValueError("Invalid Time Format")

def video_clip(input_path, output_path, start_time, end_time):
    if type(start_time) == str:
        start_time = start_time.split('.')[0]
        start_time = time_cvt(start_time)
    if type(end_time) == str:
        end_time = end_time.split('.')
        end_time = time_cvt(end_time)

    with VideoFileClip(input_path) as video:
        clipped = video.subclip(start_time, end_time)
        clipped.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        clipped.close()
```
```python
import os
import json
import random

with open('filtered_train.json', 'r') as f:
    data = json.load(f)
    prompts = [_['QA'][0]['i'] for _ in data]

dataset = []
videos = set(os.listdir('videos'))
idx = 0
with open('InternVid-10M-flt.jsonl', 'r') as f:
    line = f.readline()
    while line:
        data = json.loads(line)
        if idx not in videos:
            line = f.readline()
            idx += 1
            continue
        inst = random.sample(prompts, 1)[0]
        conversations = [
            {'from': 'human', 'value': '<image>\n{inst}'.format(inst=inst)},
            {'from': 'gpt', 'value': data['Caption']}
        ]
        obj = {
            'conversations': conversations,
            'video': 'internvid/videos/{}.mp4'.format(idx),
            'query': inst
        }
        dataset.append(obj)
        line = f.readline()
        idx += 1
```

[filtered_train.json](https://www.modelscope.cn/datasets/AI-ModelScope/webvid-caption/resolve/master/filtered_train.json)

## ShareGPTVideo
### URL
https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction
### Data Scale
900k
### Processing
```bash
cd train_300k
tar -zxvf *.tar.gz
cd ..
cd train_600k
tar -zxvf *.tar.gz
```
```python
# video_instruction/train/pretrain
import json
import os
from tqdm import tqdm

train_300k = set(os.listdir('train_300k'))
train_600k = set(os.listdir('train_600k'))

dataset = []
with open('video_caption_pretrain.jsonl', 'r') as f:
    line = f.readline()
    while line:
        dataset.append(json.loads(line))
        line = f.readline()

new_dataset = []
for data in tqdm(dataset):
    video_id = data['video']
    if video_id in train_300k:
        video_id = 'train_300k/{}'.format(video_id)
    elif video_id in train_600k:
        video_id = 'train_600k/{}'.format(video_id)
    else:
        continue
    question = data['conversations'][0]['value'].replace('\n<video>', '').replace('<video>\n', '')
    answer = data['conversations'][1]['value']
    conversations = [
        {'from': 'human', 'value': '<image>\n{}'.format(question)},
        {'from': 'gpt', 'value': answer}
    ]
    data = {
        'conversations': conversations,
        'video': 'ShareGPTVideo/{}'.format(video_id),
        'query': question
    }
    new_dataset.append(data)
```

## OpenVid
### URL
https://huggingface.co/datasets/nkp37/OpenVid-1M
### Data Scale
1M
### Processing
```bash
# for files < 50GB 
unzip -j *.zip -d video_folder
# for files > 50GB
cat OpenVid_part<id>_part* > OpenVid_part<id>.zip
unzip -j OpenVid_part<id>.zip -d video_folder
```
```python
# data/train
import json
import pandas as pd
import random
with open('filtered_train.json', 'r') as f:
    data = json.load(f)
    prompts = [_['QA'][0]['i'] for _ in data]
df = pd.read_csv("OpenVid-1M.csv")
dataset = []
for idx, row in df.iterrows():
    video_id = row['video']
    caption = row['caption']
    inst = random.sample(prompts, 1)[0]
    conversations = [
        {'from': 'human', 'value': '<image>\n{inst}'.format(inst=inst)},
        {'from': 'gpt', 'value': caption}
    ]
    obj = {
        'conversations': conversations,
        'video': 'OpenVid/video_folder/{}.mp4'.format(video_id),
        'query': inst
    }
    dataset.append(obj)
```

[filtered_train.json](https://www.modelscope.cn/datasets/AI-ModelScope/webvid-caption/resolve/master/filtered_train.json)

## Vript
### URL
https://huggingface.co/datasets/Mutonix/Vript
### Data Scale
400k
### Processing
```bash
# vript_long_videos_clips
unzip *.zip
mv clips_1_of_1095/* videos
...
```
```python
# vript_captions
import json
import random

with open('filtered_train.json', 'r') as f:
    data = json.load(f)
    prompts = [_['QA'][0]['i'] for _ in data]

with open('vript_long_videos_captions.jsonl', 'r') as f:
    dataset = [json.loads(_) for _ in f]
new_dataset = []
for data in dataset:
    video_id = data['meta']['video_id']
    clip_id = data['clip_id']
    caption = data['caption']['shot_type']
    inst = random.sample(prompts, 1)[0]
    conversations = [
        {'from': 'human', 'value': '<image>\n{inst}'.format(inst=inst)},
        {'from': 'gpt', 'value': caption}
    ]
    obj = {
        'conversations': conversations,
        'video': 'Vript/videos/{}/{}.mp4'.format(video_id, clip_id),
        'query': inst
    }
    dataset.append(obj)
```

## VideoChat2-IT
### URL
https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT
### Data Scale
video QA (200k) + video reasoning (100k)
### Processing
```python
# video
import os
import json

sub_sets = [
    'vqa/ego_qa/train.json', 
    'vqa/sharegptvideo/train_240k.json', 
    'vqa/tgif_frame_qa/train.json', 
    'vqa/tgif_transition_qa/train.json', 
    'vqa/webvid_qa/train.json',
    'reasoning/clevrer_mc/train.json',
    'reasoning/clevrer_qa/train.json',
    'reasoning/next_qa/train.json'
    ]

new_dataset = []
for sub_set in sub_sets:
    with open(sub_set, 'r') as f:
        dataset = json.load(sub_set)
    for data in dataset:
        question = data['q']
        answer = data['a']
        video = data['video']
        conversations = [
            {'from': 'human', 'value': '<image>\n{question}'.format(question=question)},
            {'from': 'gpt', 'value': answer}
        ]
        obj = {
            'conversations': conversations,
            'video': 'VideoChat2/videos/{}'.format(video),
            'query': question
        }
        new_dataset.append(obj)
```

## EgoTaskQA
### URL
Please go to the official dataset website at https://sites.google.com/view/egotaskqa to register, and the download link will be sent to you via email.
### Data Scale
40k
### Processing
```python
# qa/
import os
import json
files = ['direct/train_qas.json', 'indirect/train_qas.json']
new_dataset = []
for file in files:
    with open(file, 'r') as f:
        dataset = json.load(f)
    for data in dataset:
        interval = data['interval']
        question = data['question']
        answer = data['answer']
        conversations = [
            {'from': 'human', 'value': '<image>\n{question}'.format(question=question)},
            {'from': 'gpt', 'value': answer}
        ]
        obj = {
            'conversations': conversations,
            'video': 'EgoTaskQA/qa_videos/{}.mp4'.format(interval),
            'query': question
        }
        new_dataset.append(obj)
```

## CLEVRER
### URL
http://clevrer.csail.mit.edu/
### Data Scale
700k
### Processing
```python
import json

with open('train.json', 'r') as f:
    dataset = json.load(f)
new_dataset = []
idx2opts = ['A', 'B', 'C', 'D', 'E', 'F']
for data in dataset:
    questions = data['questions']
    video = data['video_filename']
    for qa in questions:
        question = qa['question']
        answer = None
        if 'choices' in qa:
            choices = qa['choices']
            options = ['{}. {}'.format(idx2opts[_['choice_id']], _['choice'])]
            for _ in choices:
                if _['answer'] == 'correct':
                    answer = '{}. {}'.format(idx2opts[_['choice_id']], _['choice'])
                    break
            conversations = [
                {'from': 'human', 'value': '<image>\n{question}. Please choose the right answer from the following options: {options}'.format(question=question, options=options)},
                {'from': 'gpt', 'value': answer}
            ]
            data = {
                'conversations': conversations,
                'video': 'clevrer/data/{}'.format(video),
                'query': question
            }
            new_dataset.append(data)
        else:
            answer = data['answer']
            conversations = [
                {'from': 'human', 'value': '<image>\n{question}'.format(question=question)},
                {'from': 'gpt', 'value': answer}
            ]
            data = {
                'conversations': conversations,
                'video': 'clevrer/data/{}'.format(video),
                'query': question
            }
            new_dataset.append(data)
```

## LLaVA-Video
### URL
https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K
### Data Scale
178K multi-turn dialogue data
### Processing
```python
import json
import os
from tqdm import tqdm
import cv2

def process_mc_dataset(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    new_dataset = []
    for data in dataset:
        try:
            id = data['id']
        except:
            continue
        data_source = data['data_source']
        video = data['video']
        conv = data['conversations']
        assert len(conv) % 2 == 0
        qa = []
        for i in range(0, len(conv), 2):
            question = conv[i]
            answer = conv[i+1]
            assert question['from'] == 'human' and answer['from'] == 'gpt'
            question = question['value']
            answer = answer['value']
            question = question.replace('<image>\n', '')
            question, options = question.split('\n')[0], question.split('\n')[1:-1]
            if question is None or options is None:
                print(conv[i], conv[i+1])
                continue
            qa.append({'question': question, 'options': options, 'answer': answer})
        new_dataset.append({'id': id, 'data_source': data_source, 'video': video, 'qa': qa})
    return new_dataset

def parse_mc_dataset(dataset):
    new_dataset = []
    for data in dataset:
        video = data['video']
        qa = data['qa']
        data_source = data['data_source']
        for q_a in qa:
            question = q_a['question']
            options = q_a['options']
            answer = q_a['answer']
            conversations = [
                {'from': 'human', 'value': '<image>\n{question}. Please choose the right answer from the following options: {options}.'.format(question=question, options=options)},
                {'from': 'gpt', 'value': answer}
            ]
            obj = {
                'conversations': conversations,
                'video': 'llava-video/{}/{}'.format(data_source, video),
                'query': question
            }
            new_dataset.append(obj)
    return new_dataset

def process_oe_dataset(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    new_dataset = []
    for data in dataset:
        try:
            id = data['id']
            data_source = data['data_source']
            video = data['video']
            conv = data['conversations']
            assert len(conv) % 2 == 0
            qa = []
            for i in range(0, len(conv), 2):
                question = conv[i]
                answer = conv[i+1]
                assert question['from'] == 'human' and answer['from'] == 'gpt'
                question = question['value']
                answer = answer['value']
                question = question.replace('<image>\n', '')
                qa.append({'question': question, 'answer': answer})
            new_dataset.append({'id': id, 'data_source': data_source, 'video': video, 'qa': qa})
        except Exception as e:
            print(e)
            continue
    return new_dataset

def parse_oe_dataset(dataset):
    new_dataset = []
    for data in dataset:
        video = data['video']
        qa = data['qa']
        data_source = data['data_source']
        for q_a in qa:
            question = q_a['question']
            answer = q_a['answer']
            conversations = [
                {'from': 'human', 'value': '<image>\n{question}.'.format(question=question)},
                {'from': 'gpt', 'value': answer}
            ]
            obj = {
                'conversations': conversations,
                'video': 'llava-video/{}/{}'.format(data_source, video),
                'query': question
            }
            new_dataset.append(obj)
    return new_dataset

if __name__ == '__main__':
    dir_list = [file for file in os.listdir('./') if os.path.isdir(file)]

    new_dataset = []
    for sub_set in tqdm(dir_list):
        files = [file for file in os.listdir(sub_set) if file.endswith('.json')]
        for file in files:
            if 'mc' in file:
                dataset = process_mc_dataset(os.path.join(sub_set, file))
                dataset = parse_mc_dataset(dataset)
            # else:
            #     dataset = process_oe_dataset(os.path.join(sub_set, file))
            #     dataset = parse_oe_dataset(dataset)
            new_dataset += dataset
```

## MovieChat
### URL
https://huggingface.co/datasets/Enxin/MovieChat-1K_train
### Data Scale
1k multi-turn dialogue data
### Processing
```python
import os
import json

files = os.listdir('jsons')
dataset = []
for file in files:
    with open('jsons/{}'.format(file), 'r') as f:
        dataset.append(json.load(f))
new_dataset = []
for data in dataset:
    video_path = data['info']['video_path']
    qa_list = data['global']
    for qa in qa_list:
        question = qa['question']
        answer = qa['answer']
        conversations = [
                {'from': 'human', 'value': '<image>\n{question}.'.format(question=question)},
                {'from': 'gpt', 'value': answer}
            ]
        obj = {
            'conversations': conversations,
            'video': 'MovieChat/raw_videos/{}'.format(video_path),
            'query': question
        }
        new_dataset.append(obj)
```

## PerceptionTest
### URL
https://huggingface.co/datasets/lmms-lab/PerceptionTest_Val
### Data Scale
20k
### Processing
```bash
unzip videos_chunked_01.zip
unzip videos_chunked_02.zip
```
```python
import pandas as pd

df = pd.read_parquet('test-00000-of-00001.parquet')
dataset = []
for idx, row in df.iterrows():
    video = row['video_name']
    question = row['question']
    options = row['options']
    answer = options[row['answer_id']]
    conversations = [
        {'from': 'human', 'value': '<image>\n{question}. Please choose the right answer from the following options: {options}.'.format(question=question, options=options)},
        {'from': 'gpt', 'value': answer}
    ]
    obj = {
        'conversations': conversations,
        'video': 'PerceptionTest/videos/{}.mp4'.format(video),
        'query': question
    }
    dataset.append(obj)
```

## STAR
### URL
Dataset
https://pan.baidu.com/s/1AjOzqX5WxrZvDr5ZKpVn_w
Access Code: 6v8u
Video Source
https://prior.allenai.org/projects/charades
### Data Scale
60k Multi-Choice
### Processing
```python
import json
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def video_clip(input_path, output_path, start_time, end_time):
    with VideoFileClip(input_path) as video:
        clipped = video.subclip(start_time, end_time)
        clipped.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        clipped.close()

with open('STAR_train.json', 'r') as f:
    dataset = json.load(f)
new_dataset = []
idx2opt = ['A', 'B', 'C', 'D', 'E', 'F']
for data in tqdm(dataset):
    question_id = data['question_id']
    video_id = data['video_id']
    start_time = float(data['start'])
    end_time = float(data['end'])
    video_clip(
        'videos/{}.mp4'.format(video_id),
        'cliped_videos/{}.mp4'.format(question_id),
        start_time,
        end_time)
    question = data['question']
    options = [_['choice'] for _ in data['choices']]
    answer = data['answer']
    options = ['{}. {}'.format(idx2opt[idx], options[idx]) for idx in range(len(options))]
    for option in options:
        if answer == ' '.join(option.split(' ')[1:]):
            answer = option
            break
    conversations = [
        {'from': 'human', 'value': '<image>\n{question}. Please choose the right answer from the following options: {options}.'.format(question=question, options=options)},
        {'from': 'gpt', 'value': answer}
    ]
    obj = {
        'conversations': conversations,
        'video': 'STAR/cliped_videos/{}.mp4'.format(question_id)
    }
    dataset.append(obj)
```


## NExTQA
### URL
https://huggingface.co/datasets/lmms-lab/NExTQA
### Data Scale
52k Open-ended QA
### Processing
```bash
unzip videos.zip
```
```python
# OE/
import json
import pandas as pd
df = pd.read_parquet('train-00000-of-00001.parquet')
dataset = []
for idx, row in df.iterrows():
    video = row['video']
    question = row['question']
    answer = row['answer']
    conversations = [
        {'from': 'human', 'value': '<image>\n{question}.'.format(question=question)},
        {'from': 'gpt', 'value': answer}
    ]
    obj = {
        'conversations': conversations,
        'video': 'nextqa/videos/{}.mp4'.format(video),
        'query': question
    }
    dataset.append(obj)
```

## FineVideo
### URL
https://huggingface.co/datasets/HuggingFaceFV/finevideo
### Data Scale
200k
### Processing
```python
import pandas as pd
import json
import os
import cv2
from tqdm import tqdm
import numpy as np

def write_video(chunks):
    files = os.listdir('data')
    for file in tqdm(files[:chunks]):
        try:
            chunk = pd.read_parquet('data/{}'.format(file))
        except:
            print(file)
            continue
        chunk_idx = int(file[6:11])
        for idx, row in chunk.iterrows():
            video = row['mp4']
            video_file = 'finevideo/video/{}-{}.mp4'.format(chunk_idx, idx)
            if not os.path.exists(video_file):
                with open(video_file, 'wb') as f:
                    f.write(video)

def parse_dataset(chunks):
    dataset = []
    files = os.listdir('data')
    for file in tqdm(files[:chunks]):
        try:
            chunk = pd.read_parquet('data/{}'.format(file))
        except:
            print(file)
            continue
        chunk_idx = int(file[6:11])
        for idx, row in chunk.iterrows():
            qa_list = row['json']['content_metadata']['qAndA']
            for qa in qa_list:
                conversations = [
                    {'from': 'human', 'value': '<image>\n{}'.format(qa['question'])},
                    {'from': 'gpt', 'value': qa['answer']}
                ]
                data = {
                    'conversations': conversations,
                    'video': 'finevideo/video/{}-{}.mp4'.format(chunk_idx, idx),
                    'query': qa['question']
                }
                dataset.append(data)
    return dataset
```

## CinePile
### URL
https://huggingface.co/datasets/tomg-group-umd/cinepile
### Data Scale
300k
### Processing
```python
import os
import json
import pandas as pd

dataset = []
idx2opt = ['A', 'B', 'C', 'D', 'E', 'F']
for i in range(3):
    df = pd.read_parquet('v2/train-0000{}-of-00003.parquet'.format(i))
    for idx, row in df.iterrows():
        videoID = row['videoID']
        question = row['question']
        options = row['choices']
        answer = row['answer_key']
        options = ['{}. {}'.format(idx2opt[idx], options[idx]) for idx in range(len(options))]
        for option in options:
            if answer == ' '.join(option.split(' ')[1:]):
                answer = option
                break
        conversations = [
            {'from': 'human', 'value': '<image>\n{question}. Please choose the right answer from the following options: {options}.'.format(question=question, options=options)},
            {'from': 'gpt', 'value': answer}
        ]
        obj = {
            'conversations': conversations,
            'video': 'CinePile/videos/{}.mp4'.format(videoID)
        }
        dataset.append(obj)
```

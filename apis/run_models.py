import numpy as np
import pandas as pd
import re
import torch
from transformers import AutoTokenizer
from transformers import logging

logging.set_verbosity_error()  # ignore transformer warning


# kogpt model
class Kogpt:
    # check model path
    MODEL_PATH = "../ML_models/jm_model.pt"
    TOKENIZER_NAME = "skt/kogpt2-base-v2"

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        self.device = device
        self.model = torch.load(self.MODEL_PATH, map_location=device)

    def predict(self, sent):
        # set my model max len
        MAX_LEN = 60

        self.model.eval()
        self.tokenizer.padding_side = "right"

        # Define PAD Token = EOS Token = 50256

        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_sent = self.tokenizer(
            sent,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN
        )

        tokenized_sent.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_sent["input_ids"],
                attention_mask=tokenized_sent["attention_mask"],
            )

        logits = outputs[0]
        logits = logits.detach().cpu()
        result = logits.argmax(-1)

        return np.array(result)[0]


# koelectra model
class Koelectra:
    # check model path
    MODEL_PATH = "../ML_models/koelectra_imbalanced.pt"  # manage.py 기준 상대 경로
    TOKENIZER_NAME = "monologg/koelectra-base-v3-discriminator"

    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        self.device = device
        self.model = torch.load(self.MODEL_PATH, map_location=device)

    def predict(self, sent):
        # set my model max len
        MAX_LEN = 64

        self.model.eval()

        tokenized_sent = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN
        )

        tokenized_sent.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_sent["input_ids"],
                attention_mask=tokenized_sent["attention_mask"],
                token_type_ids=tokenized_sent["token_type_ids"]
            )

        logits = outputs[0]
        logits = logits.detach().cpu()
        result = logits.argmax(-1)

        return np.array(result)[0]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load KOGPT
kogpt = Kogpt()

# Load KOELECTRA
koelectra = Koelectra(device)


# Load KOBERT


def get_sentence(json_content):
    target = json_content["images"][0][0]['fields'][0]['inferText']
    start = False
    sentences = []

    for d in json_content["images"][0][0]['fields'][2:]:
        x = d['boundingPoly']['vertices'][0]['x']
        content = d['inferText']

        if start == True:
            if x < 900:
                temp += content + " "

            else:
                sentences.append(temp)
                start = False

        else:
            if x < 900 and content == target:
                start = True
                temp = ""

    print(sentences)
    func = lambda s: s[:list(re.finditer(f"오전|오후", s))[-1].span()[0] - 1]
    sentences = [func(i) for i in sentences]
    return sentences


def handle_received_json(json_content):
    # 데이터가 실제로 존재하는지 확인
    if not json_content:
        return None

    j = pd.read_json(json_content, lines=True)
    sentences = get_sentence(j)
    print(sentences)

    # Make Prediction
    label_decoder = {0: "sadness",
                     1: "fear",
                     2: "disgust",
                     3: "neutral",
                     4: "happiness",
                     5: "angry",
                     6: "surprise"}

    # if predict only last sentence
    sentence = sentences[-1]
    print(sentence)

    # KOGPT Prediction
    kogpt_prediction = kogpt.predict(sentence)

    # KOELECTRA Prediction
    koelectra_prediction = koelectra.predict(sentence)

    # KOBERT Prediction
    kobert_prediction = 0

    print(f"kogpt_prediction : {kogpt_prediction} : {label_decoder[kogpt_prediction]}")
    print(f"koelectra_prediction : {koelectra_prediction} : {label_decoder[koelectra_prediction]}")
    # print(f"kobert_prediction : {kobert_prediction} : {label_decoder[kobert_prediction]}")

    # ensemble
    # prediction_list = [kogpt_prediction, koelectra_prediction, kobert_prediction]
    prediction_list = [kogpt_prediction, koelectra_prediction]
    ensemble_result = max(prediction_list, key=prediction_list.count)
    print(f"ensemble_result : {ensemble_result} : {label_decoder[ensemble_result]}")
    print()

    return int(ensemble_result)

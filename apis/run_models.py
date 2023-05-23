import numpy as np
import pandas as pd
import re

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import logging
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
import multiprocessing

logging.set_verbosity_error()  # ignore transformer warning

# kobert model
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


# kogpt model
class Kogpt:
    # check model path
    MODEL_PATH = "../ML_models/jm_model.pt"  # manage.py 기준 상대 경로
    TOKENIZER_NAME = "skt/kogpt2-base-v2"

    def __init__(self):
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
        logits = 0.9 * logits
        result = logits.argmax(-1)
        print("kogpt", logits)

        return logits, np.array(result)[0]


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
        print("koelectra", logits)

        return logits, np.array(result)[0]


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# kobert model
class Kobert:
    MODEL_PATH = "../ML_models/kobert_ver2.pt"  # manage.py 기준 상대 경로

    def __init__(self, device):
        self.device = device
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)  # use the created tokenizer
        self.model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)  # BERTClassifier instance
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=device))
        self.model.eval()

    def predict(self, sent):
        # Tokenize the input sentence
        tokens = self.tokenizer(sent)

        # Convert tokens to token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Convert the token IDs to a tensor
        token_tensor = torch.tensor([token_ids])

        # Move the tensor to the appropriate device
        token_tensor = token_tensor.to(self.device)

        # Create a tensor for segment ids
        segment_tensor = torch.zeros_like(token_tensor)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(token_tensor, [len(token_ids)], segment_tensor)

        # Get the predicted class index
        predicted_index = torch.argmax(outputs)
        logits = outputs[0]
        logits = logits.detach().cpu()

        return logits, predicted_index.item()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multiprocessing.set_start_method("spawn", True)

# Load KOGPT
kogpt = Kogpt()

# Load KOELECTRA
koelectra = Koelectra(device)

# Load KOBERT
kobert = Kobert(device)


def get_sentence(json_content):
    target = json_content["images"][0][0]['fields'][0]['inferText']
    print(target)
    start = False
    sentences = []

    for d in json_content["images"][0][0]['fields'][2:]:
        x = d['boundingPoly']['vertices'][0]['x']
        content = d['inferText']

        if start == True:
            if x < 1440:
                temp += content + " "

            else:
                sentences.append(temp)
                start = False

        else:
            if x < 1440 and content == target:
                start = True
                temp = ""

    print(sentences)
    func = lambda s: s[:list(re.finditer(f"오전|오후", s))[-1].span()[0] - 1]
    sentences = [func(i) for i in sentences]
    print(sentences)
    return sentences


def handle_received_json(json_content):
    # 데이터가 실제로 존재하는지 확인
    if not json_content:
        return None

    j = pd.read_json(json_content, lines=True)
    sentences = get_sentence(j)

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
    kogpt_score, kogpt_prediction = kogpt.predict(sentence)

    # KOELECTRA Prediction
    koelectra_score, koelectra_prediction = koelectra.predict(sentence)

    # KOBERT Prediction
    kobert_score, kobert_prediction = kobert.predict(sentence)

    ensemble = kogpt_score + koelectra_score + kobert_score

    print("ensemble", ensemble)

    print(f"kogpt_prediction : {kogpt_prediction} : {label_decoder[kogpt_prediction]}")
    print(f"koelectra_prediction : {koelectra_prediction} : {label_decoder[koelectra_prediction]}")
    print(f"kobert_prediction : {kobert_prediction} : {label_decoder[kobert_prediction]}")

    # ensemble
    ensemble_result = np.argmax(ensemble.numpy())
    print(f"ensemble_result : {ensemble_result} : {label_decoder[ensemble_result]}")
    print()

    return int(ensemble_result)

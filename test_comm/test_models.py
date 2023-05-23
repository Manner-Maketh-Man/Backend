import json
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


# kogpt model
class Kogpt:
    # check model path
    MODEL_PATH = "../../ML_models/jm_model.pt"  # manage.py 기준 상대 경로
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
        result = logits.argmax(-1)
        print("kogpt     =", logits)

        return logits, np.array(result)[0]


# koelectra model
class Koelectra:
    # check model path
    MODEL_PATH = "../../ML_models/koelectra_imbalanced.pt"  # manage.py 기준 상대 경로
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
        print("koelectra =", logits)

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
    MODEL_PATH = "../../ML_models/kobert_ver2.pt"  # manage.py 기준 상대 경로

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
        print("kobert    = ", logits)

        return logits, predicted_index.item()


def get_sentence(file):
    target = file["images"][0][0]['fields'][0]['inferText']
    start = False
    sentences = []
    print(target)
    for d in file["images"][0][0]['fields'][2:]:
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

    func = lambda s: s[:list(re.finditer(f"오전|오후", s))[-1].span()[0] - 1]
    print(sentences)
    sentences = [func(i) for i in sentences]
    print(sentences)
    return sentences


def test(test_sentences):
    while test_sentences:
        sentence = test_sentences.pop()
        emotion = sentence.split(":")[0]
        sentence = sentence.split(":")[1]
        print("sentence =", sentence)

        # KOGPT Prediction
        kogpt_score, kogpt_prediction = kogpt.predict(sentence)

        # KOELECTRA Prediction
        koelectra_score, koelectra_prediction = koelectra.predict(sentence)

        # KOBERT Prediction
        kobert_score, kobert_prediction = kobert.predict(sentence)

        ensemble = kogpt_score + koelectra_score + kobert_score

        print("ensemble  =", ensemble)

        print(f"kogpt_prediction     = {kogpt_prediction} : {label_decoder[kogpt_prediction]}")
        print(f"koelectra_prediction = {koelectra_prediction} : {label_decoder[koelectra_prediction]}")
        print(f"kobert_prediction    = {kobert_prediction} : {label_decoder[kobert_prediction]}")

        # ensemble
        ensemble_result = np.argmax(ensemble.numpy())
        print(f"ensemble_result      = {ensemble_result} : {label_decoder[ensemble_result]}")
        print("answer = " + emotion)
        print()


def dump_json(file_name):
    if file_name.find("_dump") == -1:
        temp = json.load(open("./" + file_name, "r"))
        file_name = file_name.split(".")[0] + "_dump.json"
        with open(file_name, "w") as json_file:
            json.dump(temp, json_file, indent=None, ensure_ascii=False)
            print(file_name + " is dumped")
    return file_name


if __name__ == "__main__":
    # Read JSON File
    JSON_FILE_NAME = "test2_dump.json"

    # for dump json file
    JSON_FILE_NAME = dump_json(JSON_FILE_NAME)

    f = pd.read_json(JSON_FILE_NAME, lines=True)
    sentences = get_sentence(file=f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multiprocessing.set_start_method("spawn", True)

    # Load KOGPT
    kogpt = Kogpt()

    # Load KOELECTRA
    koelectra = Koelectra(device)

    # Load KOBERT
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    kobert = Kobert(device)

    # Make Prediction
    label_decoder = {0: "sadness",
                     1: "fear",
                     2: "disgust",
                     3: "neutral",
                     4: "happiness",
                     5: "angry",
                     6: "surprise"}

    # for test
    test_sentences = ["happiness:너 덕분에 할수 있었어 고맙다ㅠㅠㅠ",
                      "surprise:엥 진짜로 ?? 심각한데..;",
                      "angry:왜 자꾸 하지 말란 짓을 골라서 하냐.. 뭐하자는거임?",
                      "neutral:3시까지 만나자",
                      "neutral:정문 앞에서 만나",
                      "neutral:과제 다 했어 ??",
                      "fear:갑자기 무섭게 왜그래..",
                      "angry:하..만나서 얘기해 이젠 너무 힘들다",
                      "neutral:오키 ~ 내일보자",
                      "surprise:헉!!",
                      "surprise:와우.. 진짜로??ㅋㅋㅋㅋ",
                      "surprise:다시 사귄다고??! 거짓말하지마 ",
                      "fear:나 근데 어두운 곳에서 잠 자는게 너무 무서워.. 조금 어려울 것 같아..",
                      "angry:너 왜 자꾸 나한테 그런식으로 대하는거야?",
                      "sadness:울고싶다..",
                      "sadness:미안해.. 내가 잘못했어 한번만 용서해주면 안될까..?",
                      "happiness:헐..좋아!!",
                      "angry:아ㅋㅋㅋ 이제 내 남자친구 아니야 말도 꺼내지마 꼴도 보기 싫어",
                      "disgust:나 이제 걔 숨소리 마저 혐오스러워",
                      ]

    isTest = False
    # isTest = True

    if isTest:
        test(test_sentences)
        exit()

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

    print("ensemble  =", ensemble)

    print(f"kogpt_prediction     = {kogpt_prediction} : {label_decoder[kogpt_prediction]}")
    print(f"koelectra_prediction = {koelectra_prediction} : {label_decoder[koelectra_prediction]}")
    print(f"kobert_prediction    = {kobert_prediction} : {label_decoder[kobert_prediction]}")

    # ensemble
    ensemble_result = np.argmax(ensemble.numpy())
    print(f"ensemble_result      = {ensemble_result} : {label_decoder[ensemble_result]}")
    print()

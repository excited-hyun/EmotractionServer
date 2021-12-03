from flask import Flask, request, jsonify,make_response
import json

from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint


app = Flask(__name__)

@app.route('/original', methods=['POST'])
def postOriginal():
    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

    goemotions = MultiLabelPipeline(
        model=model,
        tokenizer=tokenizer,
        threshold=0.3
    )

    data = request.get_json(silent=True, cache=False, force=True)
    print("Received data:", data)
    text = data['text']
    print("Text:", text)

    texts = []
    texts.append(text)

    result = goemotions(texts)
#    print(result)

    labels = result[0]['labels']
    scores = result[0]['scores']
    convert_score = []
    for item in scores :
        convert_score.append(float(item))
    print(labels, convert_score)

    json_object = {}
    json_object['labels'] = labels
    json_object['scores'] = convert_score

#    print(json_object)

    response = make_response(json.dumps(json_object,ensure_ascii=False).encode('utf-8'))
    print(json_object)
    return response

#pprint(goemotions(texts))

@app.route('/group', methods=['POST'])
def postGroup():
    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-group")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-group")

    goemotions = MultiLabelPipeline(
        model=model,
        tokenizer=tokenizer,
        threshold=0.3
    )
    data = request.get_json(silent=True, cache=False, force=True)
    print("Received data:", data)
    text = data['text']
    print("Text:", text)

    texts = []
    texts.append(text)

    result = goemotions(texts)
#    print(result)

    labels = result[0]['labels']
    scores = result[0]['scores']
    convert_score = []
    for item in scores :
        convert_score.append(float(item))
    print(labels, convert_score)

    json_object = {}
    json_object['labels'] = labels
    json_object['scores'] = convert_score

#    print(json_object)

    response = make_response(json.dumps(json_object,ensure_ascii=False).encode('utf-8'))
    print(json_object)
    return response


@app.route('/ekman', methods=['POST'])
def postEkman():
    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-ekman")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-ekman")

    goemotions = MultiLabelPipeline(
        model=model,
        tokenizer=tokenizer,
        threshold=0.3
    )

    data = request.get_json(silent=True, cache=False, force=True)
    print("Received data:", data)
    text = data['text']
    print("Text:", text)

    texts = []
    texts.append(text)

    result = goemotions(texts)
#    print(result)

    labels = result[0]['labels']
    scores = result[0]['scores']
    convert_score = []
    for item in scores :
        convert_score.append(float(item))
    print(labels, convert_score)

    json_object = {}
    json_object['labels'] = labels
    json_object['scores'] = convert_score

#    print(json_object)

    response = make_response(json.dumps(json_object,ensure_ascii=False).encode('utf-8'))
    print(json_object)
    return response

app.run(host="0.0.0.0", port=5000)

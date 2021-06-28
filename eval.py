

import torch
from base_options import BaseOptions

from transformers import AdamW, get_linear_schedule_with_warmup,BertTokenizerFast,XLNetTokenizerFast
from torch.nn import DataParallel
from dataset import SquadDataset
from model import BertForSQuAD,XLNetForSQuAD
from util import *
from torch.utils.data import DataLoader
import json
def save_result(opt):
    os.environ["TOKENIZERS_PARALLELISM"] = "ture"
    
    if opt.model == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif opt.model == "xlnet":
        tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    valid_dataset = SquadDataset(tokenizer, opt, split = 'valid')
    print(len(valid_dataset))

    valid_loader = DataLoader(valid_dataset,batch_size = opt.batch_size,shuffle = False,num_workers = opt.num_workers)
    name = opt.name + '_' +opt.model

    ckp_dir = os.path.join(opt.checkpoint_path, name + '_model')
    if opt.model == "bert":
        model = BertForSQuAD.from_pretrained(ckp_dir)
    elif opt.model == "xlnet":
        model = XLNetForSQuAD.from_pretrained(ckp_dir)


    model.to(device)
    model.eval()
    pred_result,result = {},{}
    for i, data in enumerate(valid_loader):
        test_losses = AverageMeter()
        with torch.no_grad():
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            start_positions = data['start_positions'].to(device)
            end_positions = data['end_positions'].to(device)
            ans_exists = data['ans_exists'].to(device)
            outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, ans_exists = ans_exists)
            loss = outputs["loss"]
            pred_start_positions = torch.argmax(outputs["start_logits"], dim = 1)
            pred_end_positions = torch.argmax(outputs["end_logits"], dim = 1)
            pred_ans_exists = torch.argmax(outputs["ans_logits"], dim = 1)

            for j,id in enumerate(data["ids"]):
                if pred_start_positions[j] >= pred_end_positions[j] or pred_ans_exists[j] == 0:
                    pred_result[id] = " "
                    print(j)

                else:
                    pred_ans = input_ids[j][int(pred_start_positions[j]):int(pred_end_positions[j])]
                    pred_ans = tokenizer.decode(pred_ans)
                    pred_result[id] = pred_ans
                    print(pred_ans)

                if start_positions[j] >= end_positions[j]:
                    result[id] = " "
                    print(j)
                else:
                    ans = input_ids[j][int(start_positions[j]):int(end_positions[j])]
                    ans = tokenizer.decode(ans)
                    result[id] = ans
                    print(ans)


    pred_result_path = os.path.join("results", name + "_pred_result.json")
    result_path = os.path.join("results", name + "_result.json")
    with open(pred_result_path, "w") as f:
        f.write(json.dumps(pred_result))
    with open(result_path, "w") as f:
        f.write(json.dumps(result))

def evaluate(dataset, predictions):
    f1 = exact_match = total = k = 0
    for article_id in range(len(data['data'])):
        paragraphs = data['data'][article_id]['paragraphs']
        for paragraph in paragraphs:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                if prediction == " ":
                    if not ground_truths:
                        exact_match += 1
                        f1 += 1
                elif ground_truths:
                    exact_match += exact_match_score(prediction, ground_truths)
                    f1 += f1_score(prediction, ground_truths)
                    # print(exact_match_score(prediction, ground_truths), f1_score(prediction, ground_truths),prediction,ground_truths)
                else:
                    total -= 1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':
    opt = BaseOptions().parse()
    save_result(opt)
    path = "data/dev-v2.0.json"
    if opt.model == "bert":
        pred_path = "results/SQuAD_bert_pred_result.json"
    else:
        pred_path = "results/SQuAD_xlnet_pred_result.json"
    with open(path) as f:
        data = json.load(f)
    with open(pred_path) as f:
        predictions = json.load(f)
    result = evaluate(data, predictions)
    print(result)
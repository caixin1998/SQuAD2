import torch
from base_options import BaseOptions

from transformers import AdamW, get_linear_schedule_with_warmup,BertTokenizerFast,XLNetTokenizerFast
from torch.nn import DataParallel
from dataset import SquadDataset
from model import BertForSQuAD,XLNetForSQuAD
from util import *
from torch.utils.data import DataLoader

opt = BaseOptions().parse()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.devices
os.environ["TOKENIZERS_PARALLELISM"] = "ture"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if opt.model == "bert":
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# elif opt.model == "xlnet":
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')

train_dataset = SquadDataset(tokenizer, opt, split = 'train')
valid_dataset = SquadDataset(tokenizer, opt, split = 'valid')

train_loader = DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True,num_workers = opt.num_workers)
valid_loader = DataLoader(valid_dataset,batch_size = opt.batch_size,shuffle = False,num_workers = opt.num_workers)
check_path = opt.checkpoint_path



epoch = 0
best_loss = 1e20
epochs = opt.epochs
if opt.model == "bert":
    if opt.pretrained_model is not None:
        model = BertForSQuAD.from_pretrained(opt.pretrained_model)
    else:
        model = BertForSQuAD.from_pretrained('bert-base-uncased')
elif opt.model == "xlnet":
    if opt.pretrained_model is not None:
        model = XLNetForSQuAD.from_pretrained(opt.pretrained_model)
    else:
        model = XLNetForSQuAD.from_pretrained('xlnet-base-cased')
name = opt.name + "_" + opt.model
if opt.visualize:
    from tensorboardX import SummaryWriter  
    if os.path.exists(name + '_log'):
        shutil.rmtree(name + '_log')
    writer = SummaryWriter(name + '_log')  
model.train()
model.to(device)
if torch.cuda.device_count() > 1:
    model = DataParallel(model, device_ids=[int(i) for i in opt.devices.split(',')])
    
optimizer = AdamW(model.parameters(), lr=opt.lr, correct_bias=True)
total_steps = int(epochs * len(train_dataset) / opt.batch_size)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps,num_training_steps=total_steps)
     
max_grad_norm = opt.max_grad_norm    
if opt.sink:
    saved = load_checkpoint(check_path,  name + '_checkpoint.pth.tar')
    if saved:
        print('Loading checkpoint for epoch %05d.' % (saved['epoch']))
        state = saved['state_dict']
        try:
            model.module.load_state_dict(state)
        except:
            model.load_state_dict(state)
        epoch = saved['epoch']
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
    else:
        print('Warning: Could not read checkpoint!')
            
for e in range(epoch, epochs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    model.train()
    for i, data in enumerate(train_loader):
        input_ids = data['input_ids'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        start_positions = data['start_positions'].to(device)
        end_positions = data['end_positions'].to(device)
        ans_exists = data['ans_exists'].to(device)

        outputs = model(input_ids,  token_type_ids = token_type_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, ans_exists = ans_exists)
        #outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

        loss = outputs["loss"].mean()
        # print(loss.shape)
        # print(poetry_ids)
        optimizer.zero_grad()     
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()   
        scheduler.step()
        losses.update(loss.data.item(), input_ids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if opt.visualize:
            writer.add_scalar('Train/Loss', loss.data.item(), e * len(train_loader) + i)
        
        print('Epoch (train): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
               e, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses))
    model.eval()
    test_losses = AverageMeter()
    for i, data in enumerate(valid_loader):
        with torch.no_grad():
            input_ids = data['input_ids'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            start_positions = data['start_positions'].to(device)
            end_positions = data['end_positions'].to(device)
            ans_exists = data['ans_exists'].to(device)

            outputs = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, ans_exists = ans_exists)
            loss = outputs["loss"].mean()
            # start_logits = outputs["start_logits"]
            
        test_losses.update(loss.data.item(), input_ids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if opt.visualize:
            writer.add_scalar('Test/Loss', loss.data.item(), e * len(valid_loader) + i)
        
        print('Epoch (valid): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
               e, i, len(valid_loader), batch_time=batch_time,
               data_time=data_time, loss=test_losses))


    is_best = test_losses.avg < best_loss
    best_loss = min(test_losses.avg, best_loss)

    save_checkpoint(check_path, name + '_checkpoint.pth.tar',{
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict()}, is_best)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_dir = os.path.join(check_path, name + '_model')
    model_to_save.save_pretrained(output_dir)
    print(test_losses.avg, best_loss, is_best)
    if is_best:
        output_dir = os.path.join(check_path, 'best_' + name + '_model')
        model_to_save.save_pretrained(output_dir)

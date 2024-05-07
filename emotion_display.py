import tkinter as tk 
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
  
# Top level window 
frame = tk.Tk() 
frame.title("TextBox Input") 
frame.geometry('800x450') 
# Function for getting Input 
# from textbox and printing it  
# at label widget 

class BERTClass(torch.nn.Module):
    def _init_(self):
        super(BERTClass, self)._init_()
        self.bert_model = BertModel.from_pretrained('prajjwal1/bert-small', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(512, 8)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


comp = [['', 'aggressiveness', 'contempt', 'frozenness', 'pride', 'envy', 'outrage', 'dominance'],
        ['', '', 'cynicism', 'anxiety', 'optimism', 'pessimism', 'confusion', 'hope'],
        ['', '', '', 'shame', 'morbidness', 'remorse', 'unbelief', 'ambivalence'],
        ['', '', '', '', 'guilt', 'despair', 'awe', 'submission'],
        ['', '', '', '', '', 'bittersweetness', 'delight', 'love'],
        ['', '', '', '', '', '', 'disapproval', 'sentimentality'],
        ['', '', '', '', '', '', '', 'curiosity'],
        ['', '', '', '', '', '', '', '']]

def calcOutput():
    i = options[selected_option1.get()]
    j = options[selected_option2.get()]

    i, j = min(i,j), max(i,j)
    lbl4.config(text = "Resultant complex emotion: " + comp[i][j])


def printInput(): 
    inp = inputtxt.get(1.0, "end-1c") 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Loading pretrained model (best model)
    # FTmodel = BERTClass()
    FTmodel = torch.load("bestvalacc_model3.bin", map_location=torch.device(device))
    # FTmodel.load_state_dict(FTmodel['model_state_dict'])
    FTmodel = FTmodel.to(device)

    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')

    encoded_text = tokenizer.encode_plus(
        inp,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    token_type_ids = encoded_text['token_type_ids'].to(device)
    output = FTmodel(input_ids, attention_mask, token_type_ids)
    # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
    output = torch.sigmoid(output).detach().cpu()

    # thresholding at 0.5
    primary = output.flatten().round().numpy()
    primary_labels = ""
    for idx, p in enumerate(primary):
        if p==1:
            primary_labels += ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'][idx] + ", "

    lbl1.config(text = "Primary Emotions Detected: " + primary_labels[:-2])

    complex_labels = ""
    for i in range(8):
        for j in range(i+1,8):
            if primary[i]==1 and primary[j]==1:
                complex_labels += comp[i][j] + ", "

    lbl2.config(text = "Possible Complex Emotions: " + complex_labels[:-2])

  
# TextBox Creation 
inputtxt = tk.Text(frame, 
                   height = 5, 
                   width = 50) 
  
inputtxt.pack() 
  
# Button Creation 
printButton = tk.Button(frame, 
                        text = "Predict",  
                        command = printInput) 
printButton.pack() 
  
# Label Creation 
lbl1 = tk.Label(frame, text = "") 
lbl1.pack() 

# Label Creation 
lbl2 = tk.Label(frame, text = "") 
lbl2.pack() 

lbl3 = tk.Label(frame, text = "OR")
lbl3.pack()


dum = tk.Label(frame, text = "")
dum.pack()

# Dropdown menu options
options = {"anger":0, "anticipation":1, "disgust":2, "fear":3, "joy":4, "sadness":5, "surprise":6, "trust":7}

selected_option1 = tk.StringVar()
selected_option1.set("joy")
p1 = tk.Label(frame, text = "Primary Emotion 1: ")
p1.pack()
drop1 = tk.OptionMenu(frame, selected_option1, *(options.keys()))
drop1.pack()


dum1 = tk.Label(frame, text = "")
dum1.pack()

selected_option2 = tk.StringVar()
selected_option2.set("sadness")
p2 = tk.Label(frame, text = "Primary Emotion 2: ")
p2.pack()
drop2 = tk.OptionMenu(frame, selected_option2, *(options.keys()))
drop2.pack()

dum2 = tk.Label(frame, text = "")
dum2.pack()

printButton = tk.Button(frame, 
                        text = "Predict",  
                        command = calcOutput) 
printButton.pack() 

lbl4 = tk.Label(frame, text = "")
lbl4.pack()

frame.mainloop()
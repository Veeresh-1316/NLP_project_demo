# NLP_project_demo

## Instruction
- Requirments
  - tkinter
  - numpy
  - torch
  - BertModel, BertTokenizer from transformers
- Download 'bestvalacc_model3.bin' from the repository and place it under the same dirctory as 'emotion_display.py'
  - This has the saved best Bert model for the testing
- To Run:
  - python3 emotion_display.py (OR python emotion_display.py)
  - Upon successful running, a tkinter window pops up
  - Enter the sentence to be emotion-detected in the textbox and press the predict button to obtain results
  - You can also choose 2 Primary emotions from the drop-downs to obtain the complex emotion formed by the combination of the chosen 2 primary emotions

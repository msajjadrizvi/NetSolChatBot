import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

dont=0

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "NetSol ChatBot"
print("Welcome to NetSol Chat Bot!\n {You can type 'quitchat' to exit anytime}\n\n")
while True:
    sentence = input("You: ")
    if sentence == "quitchat":
        break
    if sentence == "":
        print(f"{bot_name}: Please say something... \n")
        continue

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.95:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                dont=0
    else:
        print(f"{bot_name}: Sorry I don't understand as I am still in the process of learning...\n")
        dont=dont+1
        
    if dont==3:
        print(f"{bot_name}: I am unable to get you, please contact +9233454257785 for further assistance.\n")
        dont=0
        break
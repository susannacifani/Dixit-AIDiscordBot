import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import openai
import random

#Narrator role
#input: the 6 card in his hands

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
openai.api_key = 'KEY'


def embedding_extraction(cards):
    cards_path = "cards/" + cards
    imgs = Image.open(cards_path)
    
    inputs = processor(images=imgs, return_tensors="pt")
    
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    
    image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
    return(image_embedding)


def GPT_generation(image_embedding):   
    embedding_text = f"I have an image represented by this embedding {image_embedding.shape} which contains visual information extracted by CLIP. Imagine you are the storyteller in a Dixit game. Based on this embedding provide five different very short clue, no more than four words."
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "system", "content": "You are a creative storyteller giving cryptic and evocative clues, based on visual content provided as an image embedding, like in a Dixit game."},
              {"role": "user", "content": embedding_text}],
      max_tokens=50,
      temperature=0.9,
      top_p=1.0
    )
    
    clues = response['choices'][0]['message']['content'].strip()
    print(clues)
    return clues


def generate_hint(cards):
    card = random.choice(cards) # Sceglie una carta casuale tra quelle 6 che ha in mano
    image_embedding = embedding_extraction(card)
    clues = GPT_generation(image_embedding)
    clues_splitted = clues.split("\n")
    cleaned_clues = [clue.lstrip('0123456789. ').rstrip('.') for clue in clues_splitted if clue]  # Rimuove i numeri e gli spazi
    random_clue = random.choice(cleaned_clues)
    print(random_clue)
    return random_clue, card # Ritorna l'indizio e la relativa carta scelta

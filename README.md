# AI-PIXEL-open-AI-Version
Hi I'm p1xo0, this is my ai i made its for a robot, that I'm making For coffee and cleaning my room and cleaning up my space.


from http import client
import numpy as np
from datetime import datetime
import math
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv
api_key = os.getenv("YOUR API KEY") #MAKE SURE YOU ADD YOUR API KEY FROM https://platform.openai.com/api-keys

print("Key loaded:", bool(api_key))  

client = OpenAI(api_key=api_key)

def ask_chatgpt(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Actual Error: {e}"

if __name__ == "__main__":
    print("Testing connection...")
    print("Robot:", ask_chatgpt("Say hello!"))


loaded = load_dotenv()
print(f"1. Did .env file load? {loaded}")

print(f"2. Python is looking in: {os.getcwd()}")

key = os.getenv("YOUR API KEY") #MAKE SURE YOU ADD YOUR API KEY FROM https://platform.openai.com/api-keys
if key:
    print(f"3. Key found! (Starts with: {key[:8]}...)")
else:
    print("3. Key not found .env file.")

def try_calculate(text):
    try:
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round, "pow": pow})
        return eval(text, {"__builtins__": None}, allowed_names)
    except:
        return None


all_vocab = ["hi", "coffee", "christmas", "time", "help", "write", "report"] 
word_to_idx = {w: i for i, w in enumerate(all_vocab)}
intents = ["hello", "coffee", "christmas", "time", "deep_task"]
intent_to_idx = {name: i for i, name in enumerate(intents)}
idx_to_intent = {i: name for name, i in intent_to_idx.items()}


weights = np.random.randn(len(all_vocab), len(intents)) * 0.01
biases = np.zeros((1, len(intents)))

def predict_intent(text):

    vec = np.zeros((1, len(all_vocab)))
    for word in text.lower().split():
        if word in word_to_idx: vec[0, word_to_idx[word]] = 1.0
    logits = vec @ weights + biases
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return idx_to_intent[np.argmax(probs)], np.max(probs)


def ask_chatgpt(prompt):
    try:
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", #You can change the model too but make sure you check the chatgpt bot if it is the right model by saying http from date import BUT DONT                                       #TELL IT TO UPGRADE ITS SELF.
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000 #change the tokens if you have pro on chatgpt
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
      
        return f"Deep brain error: {str(e)}"

def generate_mega_report(topic):
    """The 10,000-word strategy: Outline -> Section by Section."""
    print(f"robot: Starting Mega-Report on '{topic}'...")
    
    outline_prompt = f"Create a detailed 10-chapter outline for a 10,000-word book about: {topic}. Return only the chapter titles."
    outline_raw = ask_chatgpt(outline_prompt, "You are a professional book editor.")
    chapters = [line.strip() for line in outline_raw.split('\n') if line.strip() and any(char.isdigit() for char in line[:3])]
    
    full_text = f"# MEGA REPORT: {topic.upper()}\n\n"
    
  
    for i, chapter in enumerate(chapters):
        print(f"robot: Writing {chapter} ({i+1}/{len(chapters)})...")
        chapter_prompt = f"Write a 1,000-word, highly detailed chapter for: {chapter}. Context: This is for a book about {topic}."
        chapter_content = ask_chatgpt(chapter_prompt, "You are an expert technical writer. Be exhaustive and thorough.")
        full_text += f"\n\n## {chapter}\n\n{chapter_content}"
    

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    return f"I have finished the report! It is saved as {filename}. It contains roughly {len(full_text.split())} words."


def get_best_answer(user_input):

    calc = try_calculate(user_input)
    if calc is not None: return f"The answer is {calc}"


    intent, confidence = predict_intent(user_input)
    
    user_lower = user_input.lower()
    

    if any(word in user_lower for word in ["write a book", "long report", "essay", "1000 words", "mega"]):
        return generate_mega_report(user_input)
    

    if confidence < 0.5 or intent == "deep_task":
        return ask_chatgpt(user_input)


    if intent == "time": return f"It's {datetime.now().strftime('%H:%M')}"
    if intent == "hello": return "Hey there, Master!"
    
    return ask_chatgpt(user_input)


if __name__ == "__main__":
    print("AI 'SMART-MODE' Online. (Type 'exit' to quit)")
    while True:
        user: str = input("MASTER: ").strip()
        if user.lower() == "exit": break
        if not user: continue
        
        response = get_best_answer(user)
        print("robot:", response)

 
    np.savez("robot_brain.npz", weights=weights, biases=biases)

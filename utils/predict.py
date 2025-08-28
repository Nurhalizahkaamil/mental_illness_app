import torch
import torch.nn as nn
import json
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

# === Konfigurasi path ===
TOKENIZER_PATH = "models/tokenizer_lv2"
MODEL_LVL1_PATH = "models/model_lv1.pt"
MODEL_LVL2_PATH = "models/model_lv2.pt"
LABEL_MAPPING_LVL1_PATH = "models/label_mapping_lv1.json"
LABEL_MAPPING_LVL2_PATH = "models/label_mapping_lvl2.json"

# === Load label mapping ===
with open(LABEL_MAPPING_LVL1_PATH, "r", encoding="utf-8") as f:
    label_mapping_lvl1 = json.load(f)
with open(LABEL_MAPPING_LVL2_PATH, "r", encoding="utf-8") as f:
    label_mapping_lvl2 = json.load(f)

# === Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model class dari training ===
class IndoBERTweetGRUClassifier(nn.Module):
    def __init__(self, model_name, hidden_size=256, num_labels=2, dropout=0.3):
        super(IndoBERTweetGRUClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size, 
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        gru_output, _ = self.gru(last_hidden_state)
        pooled = torch.mean(gru_output, dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# === Caching model dan tokenizer
@st.cache_resource
def get_models_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    model_lvl1 = IndoBERTweetGRUClassifier(
        model_name="indolem/indobertweet-base-uncased",
        num_labels=len(label_mapping_lvl1)
    )
    model_lvl1.load_state_dict(torch.load(MODEL_LVL1_PATH, map_location=device))
    model_lvl1.to(device)
    model_lvl1.eval()

    model_lvl2 = IndoBERTweetGRUClassifier(
        model_name="indolem/indobertweet-base-uncased",
        num_labels=len(label_mapping_lvl2)
    )
    model_lvl2.load_state_dict(torch.load(MODEL_LVL2_PATH, map_location=device))
    model_lvl2.to(device)
    model_lvl2.eval()

    return tokenizer, model_lvl1, model_lvl2

# === Utilities
def to_py(val):
    return val.item() if hasattr(val, "item") else val

def is_news_account(tweets: List) -> bool:
    news_keywords = ["berita", "tempo", "kompas", "detik", "cnn", "liputan", "media", "bisnis", "politik"]
    def get_text(t):
        if isinstance(t, dict) and "text" in t:
            return t["text"]
        return str(t)
    count_links = sum("http" in get_text(t).lower() for t in tweets)
    match_keywords = sum(any(word in get_text(t).lower() for word in news_keywords) for t in tweets)
    return count_links >= 3 or match_keywords >= 2

def predict(texts: List) -> List[Dict]:
    # Pastikan texts adalah list of string
    texts = [t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in texts]
    tokenizer, model_lvl1, model_lvl2 = get_models_and_tokenizer()
    results = []

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        # Level 1 prediction
        logits_lvl1 = model_lvl1(input_ids=input_ids, attention_mask=attention_mask)
        probs_lvl1 = torch.softmax(logits_lvl1, dim=1)
        preds_lvl1 = torch.argmax(probs_lvl1, dim=1)
        ids_lvl1 = preds_lvl1.cpu().numpy()
        labels_lvl1 = [label_mapping_lvl1.get(str(idx), str(idx)) for idx in ids_lvl1]

    # === Prepare input for Level 2 only if Level 1 says 'mental_illness'
    idx_for_lvl2 = [i for i in range(len(texts)) if labels_lvl1[i] == "mental_illness"]
    texts_for_lvl2 = [texts[i] for i in idx_for_lvl2]
    idx_map = {j: i for j, i in enumerate(idx_for_lvl2)}

    labels_lvl2 = ["none"] * len(texts)
    probs_lvl2 = [0.0] * len(texts)

    if texts_for_lvl2:
        encoded_lvl2 = tokenizer(texts_for_lvl2, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids_lvl2 = encoded_lvl2["input_ids"].to(device)
        attention_mask_lvl2 = encoded_lvl2["attention_mask"].to(device)

        with torch.no_grad():
            logits_lvl2 = model_lvl2(input_ids=input_ids_lvl2, attention_mask=attention_mask_lvl2)
            probs = torch.softmax(logits_lvl2, dim=1)
            preds = torch.argmax(probs, dim=1)
            for j in range(len(texts_for_lvl2)):
                i = idx_map[j]
                labels_lvl2[i] = label_mapping_lvl2.get(str(preds[j].item()), str(preds[j].item()))
                probs_lvl2[i] = float(probs[j][preds[j]].item())

    # === Combine results
    for i in range(len(texts)):
        results.append({
            "text": texts[i],
            "label_lv1": labels_lvl1[i],
            "confidence_lv1": float(to_py(probs_lvl1[i][preds_lvl1[i]])),
            "label_lv2": labels_lvl2[i],
            "confidence_lv2": probs_lvl2[i],
        })

    return results

# === Untuk Streamlit dan database
def predict_user_and_save(username: str, tweets: List, db) -> Dict:
    # Pastikan tweets bisa diproses baik list of dict maupun list of string
    if is_news_account(tweets):
        return {
            "status": "akun_berita",
            "message": "Akun terdeteksi sebagai portal berita, klasifikasi tidak dilakukan.",
            "prediction": None,
            "tweets": tweets
        }

    texts = [t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in tweets]
    results = predict(texts)
    best = max(results, key=lambda x: x["confidence_lv2"])

    label_lv1 = best["label_lv1"]
    conf_lv1 = best["confidence_lv1"]
    label_lv2 = best["label_lv2"]
    conf_lv2 = best["confidence_lv2"]

    if label_lv1 == "mental_illness" and label_lv2 not in ["none", "unknown"]:
        final_status = "terindikasi"
        debug_flag = "ok"
    else:
        final_status = "normal"
        debug_flag = "lvl1_none" if label_lv1 == "none" else "low_conf"

    db.insert_result({
        "username": username,
        "label_lv1": label_lv1,
        "confidence_lv1": conf_lv1,
        "label_lv2": label_lv2,
        "confidence_lv2": conf_lv2,
        "final_status": final_status,
        "debug_flag": debug_flag,
        "tweets": tweets
    })

    return {
        "status": "success",
        "label_lv1": label_lv1,
        "confidence_lv1": conf_lv1,
        "label_lv2": label_lv2,
        "confidence_lv2": conf_lv2,
        "final_status": final_status,
        "debug_flag": debug_flag,
        "tweets": tweets
    }

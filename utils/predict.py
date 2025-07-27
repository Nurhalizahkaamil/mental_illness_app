import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from download_models import download_and_extract
download_and_extract()


# === Konfigurasi path ===
TOKENIZER_PATH = "models/tokenizer_lv2"
MODEL_LVL1_PATH = "models/model_lv1.pt"
MODEL_LVL2_PATH = "models/model_lv2.pt"
LABEL_MAPPING_LVL1_PATH = "models/label_mapping_lv1.json"
LABEL_MAPPING_LVL2_PATH = "models/label_mapping_lvl2.json"

# === Load alat bantu ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# === Load label mapping ===
with open(LABEL_MAPPING_LVL1_PATH, "r", encoding="utf-8") as f:
    label_mapping_lvl1 = json.load(f)
with open(LABEL_MAPPING_LVL2_PATH, "r", encoding="utf-8") as f:
    label_mapping_lvl2 = json.load(f)

# === Model class dari training ===
class IndoBERTweetGRUClassifier(nn.Module):
    def __init__(self, model_name, hidden_size=256, num_labels=2, dropout=0.3):
        super(IndoBERTweetGRUClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=True)
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

# === Load model level 1 ===
model_lvl1 = IndoBERTweetGRUClassifier(
    model_name="indolem/indobertweet-base-uncased", 
    num_labels=len(label_mapping_lvl1)
)
model_lvl1.load_state_dict(torch.load(MODEL_LVL1_PATH, map_location=device))
model_lvl1.to(device)
model_lvl1.eval()

# === Load model level 2 ===
model_lvl2 = IndoBERTweetGRUClassifier(
    model_name="indolem/indobertweet-base-uncased", 
    num_labels=len(label_mapping_lvl2)
)
model_lvl2.load_state_dict(torch.load(MODEL_LVL2_PATH, map_location=device))
model_lvl2.to(device)
model_lvl2.eval()

# === Utility ===
def to_py(val):
    return val.item() if hasattr(val, "item") else val

def is_news_account(tweets: List[str]) -> bool:
    news_keywords = ["berita", "tempo", "kompas", "detik", "cnn", "liputan", "media", "bisnis", "politik"]
    count_links = sum("http" in t.lower() for t in tweets)
    match_keywords = sum(any(word in t.lower() for word in news_keywords) for t in tweets)
    return count_links >= 3 or match_keywords >= 2

# === Prediction ===
def predict(texts: List[str]) -> List[Dict]:
    results = []
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits_lvl1 = model_lvl1(input_ids=input_ids, attention_mask=attention_mask)
        probs_lvl1 = torch.softmax(logits_lvl1, dim=1)
        preds_lvl1 = torch.argmax(probs_lvl1, dim=1)
        ids_lvl1 = preds_lvl1.cpu().numpy()
        labels_lvl1 = [label_mapping_lvl1.get(str(idx), str(idx)) for idx in ids_lvl1]

    with torch.no_grad():
        logits_lvl2 = model_lvl2(input_ids=input_ids, attention_mask=attention_mask)
        probs_lvl2 = torch.softmax(logits_lvl2, dim=1)
        preds_lvl2 = torch.argmax(probs_lvl2, dim=1)
        ids_lvl2 = preds_lvl2.cpu().numpy()
        labels_lvl2 = [label_mapping_lvl2.get(str(idx), str(idx)) for idx in ids_lvl2]

    for i in range(len(texts)):
        results.append({
            "text": texts[i],
            "label_lv1": labels_lvl1[i],
            "confidence_lv1": float(to_py(probs_lvl1[i][preds_lvl1[i]])),
            "label_lv2": labels_lvl2[i],
            "confidence_lv2": float(to_py(probs_lvl2[i][preds_lvl2[i]]))
        })

    return results

# === Untuk Streamlit dan database ===
def predict_user_and_save(username: str, tweets: List[str], db) -> Dict:
    if is_news_account(tweets):
        return {
            "status": "akun_berita",
            "message": "Akun terdeteksi sebagai portal berita, klasifikasi tidak dilakukan.",
            "prediction": None,
            "tweets": tweets
        }

    results = predict(tweets)
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

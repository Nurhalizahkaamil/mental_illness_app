import os
import json
import tweepy
import time
from dotenv import load_dotenv
load_dotenv()  # ← Ini penting, untuk load .env

BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# === Client API v2 ===
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# === Folder cache untuk menyimpan tweet lokal ===
CACHE_DIR = os.getenv("TWEET_CACHE_DIR", "tweet_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_recent_tweets_with_cache(username, limit=5, force_refresh=False):
    username = username.lower()
    cache_file = os.path.join(CACHE_DIR, f"{username}.json")

    # 1. Gunakan cache jika tersedia dan tidak diminta refresh
    if os.path.exists(cache_file) and not force_refresh:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                print(f"✅ Loaded from cache: {username}")
                return cached
        except Exception:
            print(f"⚠️ Cache corrupt untuk {username}, akan ambil ulang.")

    # 2. Ambil dari Twitter
    clean_tweets = []
    try:
        user_response = client.get_user(username=username)
        if user_response.data is None:
            print("❌ Username tidak ditemukan.")
            return None

        user_id = user_response.data.id

        tweets_response = client.get_users_tweets(
            id=user_id,
            max_results=20,
            tweet_fields=["text", "referenced_tweets"],
            expansions=["referenced_tweets.id", "referenced_tweets.id.author_id"]
        )

        if not tweets_response.data:
            print("⚠️ Tidak ada tweet.")
            return "REPOST_DETECTED"

        # Map referenced tweets by id (untuk retweet, quoted tweet, dsb)
        referenced_tweets_map = {}
        if tweets_response.includes and "tweets" in tweets_response.includes:
            for rt in tweets_response.includes["tweets"]:
                referenced_tweets_map[rt.id] = rt.text

        for tweet in tweets_response.data:
            # Cek apakah tweet ini retweet (referenced_tweets tipe retweet)
            if tweet.referenced_tweets:
                is_retweet = False
                for ref in tweet.referenced_tweets:
                    if ref.type == "retweeted":
                        is_retweet = True
                        original_id = ref.id
                        break
                if is_retweet and original_id in referenced_tweets_map:
                    text = f"RT: {referenced_tweets_map[original_id]}"
                else:
                    text = tweet.text.strip()
            else:
                text = tweet.text.strip()

            if text:
                clean_tweets.append(text)
            if len(clean_tweets) == limit:
                break

        if len(clean_tweets) < limit:
            return "REPOST_DETECTED"

        # Simpan ke cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(clean_tweets, f, ensure_ascii=False)

        print(f"✅ Fetched & cached: {username}")
        return clean_tweets

    except tweepy.TooManyRequests:
        print("❌ Terkena rate limit. Silahkan masukan tweet manual.")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached
        else:
            return None

    except Exception as e:
        print(f"❌ Error saat ambil tweet: {e}")
        return None

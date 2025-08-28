import os
import json
import tweepy
from dotenv import load_dotenv
from datetime import timezone, timedelta

# Load .env
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not BEARER_TOKEN:
    raise ValueError("❌ BEARER_TOKEN tidak ditemukan di .env")

# Client API v2
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Folder cache
CACHE_DIR = os.getenv("TWEET_CACHE_DIR", "tweet_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Zona waktu WIB
WIB = timezone(timedelta(hours=7))

def get_recent_tweets_with_cache(username, limit=10, force_refresh=False):
    """
    Ambil tweet terbaru dari username, simpan di cache, dan tampilkan waktu posting.
    """
    # Validasi username
    if not username or not username.strip():
        print("❌ Username kosong. Masukkan username tanpa '@'.")
        return None

    username = username.strip().replace("@", "").lower()
    cache_file = os.path.join(CACHE_DIR, f"{username}.json")

    # Gunakan cache jika ada dan tidak force refresh
    if os.path.exists(cache_file) and not force_refresh:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                print(f"✅ Loaded from cache: {username}")
                return cached
        except Exception:
            print(f"⚠️ Cache corrupt untuk {username}, akan ambil ulang.")

    clean_tweets = []
    try:
        # Ambil user_id
        user_response = client.get_user(username=username)
        if user_response.data is None:
            print(f"❌ Username @{username} tidak ditemukan.")
            return None

        user_id = user_response.data.id

        # Ambil tweet
        tweets_response = client.get_users_tweets(
            id=user_id,
            max_results=limit,
            tweet_fields=["text", "referenced_tweets", "created_at"],
            expansions=["referenced_tweets.id", "referenced_tweets.id.author_id"]
        )

        if not tweets_response.data:
            print("⚠️ Tidak ada tweet ditemukan.")
            return "REPOST_DETECTED"

        # Map untuk tweet referensi
        referenced_tweets_map = {}
        if tweets_response.includes and "tweets" in tweets_response.includes:
            for rt in tweets_response.includes["tweets"]:
                referenced_tweets_map[rt.id] = {
                    "text": rt.text,
                    "created_at": rt.created_at.astimezone(WIB).strftime("%Y-%m-%d %H:%M:%S WIB")
                    if hasattr(rt, "created_at") and rt.created_at else None
                }

        # Proses tweet
        for tweet in tweets_response.data:
            tweet_time = tweet.created_at.astimezone(WIB).strftime("%Y-%m-%d %H:%M:%S WIB") \
                if hasattr(tweet, "created_at") and tweet.created_at else None

            text = tweet.text.strip()

            # Jika retweet
            if tweet.referenced_tweets:
                for ref in tweet.referenced_tweets:
                    if ref.type == "retweeted" and ref.id in referenced_tweets_map:
                        text = f"RT: {referenced_tweets_map[ref.id]['text']}"
                        tweet_time = referenced_tweets_map[ref.id]["created_at"] or tweet_time
                        break

            if text:
                clean_tweets.append({
                    "type": "tweet",
                    "text": text,
                    "created_at": tweet_time
                })

            if len(clean_tweets) >= limit:
                break

        if len(clean_tweets) < limit:
            return "REPOST_DETECTED"

        # Simpan cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(clean_tweets, f, ensure_ascii=False, indent=2)

        print(f"✅ Fetched & cached: {username}")
        return clean_tweets

    except tweepy.TooManyRequests:
        print("❌ Rate limit tercapai. Gunakan cache atau coba lagi nanti.")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    except tweepy.Unauthorized:
        print("❌ Bearer token tidak valid atau tidak punya izin akses.")
        return None

    except tweepy.BadRequest as e:
        print(f"❌ Permintaan API salah: {e}")
        return None

    except Exception as e:
        print(f"❌ Error tak terduga: {e}")
        return None

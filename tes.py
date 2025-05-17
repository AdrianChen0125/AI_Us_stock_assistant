from newspaper import Article

def fetch_full_text(url: str) -> dict:
    try:
        article = Article(url)
        article.download()
        article.parse()

        return {
            "title": article.title,
            "text": article.text,
            "publish_date": article.publish_date,
            "authors": article.authors
        }
    except Exception as e:
        print(f"Failed to fetch full text for {url}: {e}")
        return {}

# 測試網址（可替換成 News API 拿到的任一新聞連結）
test_url = "https://www.theverge.com/news/664811/us-china-pause-tariffs-90-days"

if __name__ == "__main__":
    result = fetch_full_text(test_url)

    print("\n--- Title ---")
    print(result.get("title", ""))

    print("\n--- Publish Date ---")
    print(result.get("publish_date", ""))

    print("\n--- Authors ---")
    print(result.get("authors", ""))

    print("\n--- Full Text ---")
    print(result.get("text", ""))
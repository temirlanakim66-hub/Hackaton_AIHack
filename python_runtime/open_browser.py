import sys
import time
import urllib.request
import webbrowser


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
    health_url = url.rstrip("/") + "/api/health"

    for _ in range(150):
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.3)


if __name__ == "__main__":
    main()

import requests

urls = [
    "https://www.google.com",
    "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300",
    "https://data-api.binance.vision/api/v3/ping",
    "https://api.binance.com/api/v3/ping",
]

session = requests.Session()
session.trust_env = False

for url in urls:
    print(f"\nTesting: {url}")

    try:
        response = session.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        print("Status:", response.status_code)
        print("Response:", response.text[:150])

    except Exception as error:
        print("Failed:", repr(error))
import requests
import json

AURORA_API = "https://november7-730026606190.europe-west1.run.app/messages/"

def fetch_messages():
    response = requests.get(AURORA_API)
    data = response.json()
    return data.get("items", [])

def main():
    messages = fetch_messages()
    print(f"Total messages: {len(messages)}\n")

    # Print each message in a readable format
    for m in messages:
        print(f"ID: {m.get('id')}")
        print(f"User: {m.get('user_name')}")
        print(f"Message: {m.get('message')}")
        print("-" * 60)

if __name__ == "__main__":
    main()

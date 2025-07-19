import requests
import os

def send_whatsapp_message(to, message):
    instance_url = os.environ["ULTRAMSG_URL"]
    token = os.environ["ULTRAMSG_TOKEN"]
    payload = {
        "token": token,
        "to": to,
        "body": message
    }
    requests.post(f"{instance_url}/messages/chat", data=payload)
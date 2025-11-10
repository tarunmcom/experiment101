import requests
import json

def send_slack(msg):
    data = {
            "text": msg
        }
    requests.post("https://hooks.slack.com/services/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", json.dumps(data))

send_slack("Hello, worldgihljljlj!")
import json
import re
import time

import requests
import markdown

from bs4 import BeautifulSoup
from tqdm import tqdm

o = open("data2.jsonl", "w+", encoding="utf-8")

with open("data.jsonl", encoding="utf-8") as f:
    for line in tqdm(f):
        time.sleep(1)

        d = json.loads(line.rstrip("\n"))

        url = d["html_url"]
        url = url.replace("github.com/", "raw.githubusercontent.com/") + "/master/README.md"

        try:
            readme = requests.get(url).text
            readme = markdown.markdown(readme)
            bs = BeautifulSoup(readme)
            readme = bs.get_text()
            readme = re.sub("\s\s+" , " ", readme)
            d["readme"] = readme
            o.write(json.dumps(d)+"\n")
        except:
            print(url)

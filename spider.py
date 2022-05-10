import requests
from bs4 import BeautifulSoup
import os
import traceback

def download(url, filename,name):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)


if os.path.exists('imgs') is False:
    os.makedirs('imgs')

start = 341508
end = 341500
for i in range(start, end ,-1):
    url = f'http://konachan.net/post/show/{i}'
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    for img in soup.find_all('img', class_="image"):
        target_url = img['src']
        filename = os.path.join('imgs', target_url.split('/')[-1])
        name = os.path.join('imgs', str(i)) + '.jpg' 
        download(target_url, filename,name)
    print('%d / %d' % (i, end))

import json
import shutil
import requests
import urllib.request

if __name__ == '__main__':
    print('magic the gathering')
    with open('data/en-cars.json', errors='ignore') as json_file:
        data = json.load(json_file)

    en_cards = []
    img_indices = []
    count = 0
    for idx, card in enumerate(data):
        try:
            if card['lang'] == 'en' and card['image_status'] == 'highres_scan':
                uid = card['id']
                url = card['image_uris']['normal']
                en_cards.append(card)
                # urllib.request.urlretrieve(url, f"data/eng_cards/{uid}.jpg")
                # tried to copy from HF 'ordered' ds, didnt work
                #shutil.copy(f'data/cards/train/{idx}.jpg', f'data/eng_cards/{uid}.jpg')
                count += 1
                lng = card['lang']
                status = card['image_status']
                print(f'{count}: {lng} {status}')
                # print(f'{count} english and highres')
            else:
                #print(f'ignored {idx}')
                pass
        except Exception as e:
            print(f"Skipping row {idx} due to error: {e}")

    with open("data/en-cards.json", "w") as json_file:
        json.dump(en_cards, json_file, indent=4)

    print('fin')

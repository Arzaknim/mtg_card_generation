import json
import mana_parser

if __name__ == '__main__':
    print('magic the gathering')

    with open('data/en-cards.json', errors='ignore', encoding='utf-8') as json_file:
        data = json.load(json_file)

    img_caption_list = []
    for idx, card in enumerate(data):
        new_dct = {}
        new_dct['id'] = card['id']
        new_dct['file_name'] = f'{card["id"]}.jpg'
        name = card['name']
        type = card['type_line'].replace('â€”', '-')
        mana = mana_parser.replace_mana_symbols(card['mana_cost'])

        try:
            ora_text = mana_parser.replace_mana_symbols(card['oracle_text'])
        except Exception as e:
            ora_text = ''

        try:
            flavor = card['flavor_text']
        except Exception as e:
            flavor = ''

        try:
            power = card['power']
        except Exception as e:
            power = ''

        try:
            toughness = card['toughness']
        except Exception as e:
            toughness = ''

        capt = f'<NAME>{name} \n<TYPE>{type} \n<MANA>{mana} \n<ORACLE>{ora_text} \n<FLAVOR>{flavor} \n<POWER>{power} \n<TOUGHNESS>{toughness}'
        new_dct['text'] = capt
        img_caption_list.append(new_dct)

    #json
    # with open("data/img-captions.json", "w") as json_file:
    #     json.dump(img_caption_list, json_file, indent=4)

    #jsonl
    with open("data/eng_cards/metadata.jsonl", "w") as json_file:
        for dct in img_caption_list:
            json.dump(dct, json_file)
            json_file.write("\n")
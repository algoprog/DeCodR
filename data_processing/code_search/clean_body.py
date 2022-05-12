import csv
import html

from selectolax.parser import HTMLParser
from typing import List, Dict, Any

TAGS_TO_REMOVE = 'a,abbr,address,area,article,aside,audio,b,base,bdi,bdo,blockquote,body,br,button,canvas,caption,cite,code,col,colgroup,data,datalist,dd,del,details,dfn,dialog,div,dl,dt,em,embed,fieldset,figcaption,figure,footer,form,h1,h2,h3,h4,h5,h6,head,header,hgroup,hr,html,i,iframe,img,input,ins,kbd,label,legend,li,link,main,map,mark,menu,meta,meter,nav,noscript,object,ol,optgroup,option,output,p,param,picture,pre,progress,q,rp,rt,ruby,s,samp,script,section,select,slot,small,source,span,strong,style,sub,summary,sup,table,tbody,td,template,textarea,tfoot,th,thead,time,title,tr,track,u,ul,var,video,wbr'.split(
    ','
)
TAGS_TO_REMOVE.remove('code')
TAGS_TO_REMOVE.remove('body')


def read_csv(path: str, column_names: List[str]) -> List[Dict[str, Any]]:
    data = []
    print('Reading CSV...')
    with open(path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i % 10_000 == 0:
                print(f'On line {i}')
            # if i > 10:
            #     break
            entry = {}
            for name, value in zip(column_names, row):
                entry[name] = value
            data.append(entry)
    return data


def write_csv(csv_data, path: str, column_names: List[str]) -> None:
    print('Writing CSV...')
    with open(path, 'w') as f:
        writer = csv.writer(f)
        for i, row in enumerate(csv_data):
            if i % 10_000 == 0:
                print(f'On line {i}')
            writer.writerow([row[name] for name in column_names])


def process_csv(csv_data, processing_fn, column_name):
    print('Processing CSV...')
    for i, row in enumerate(csv_data):
        if i % 10_000 == 0:
            print(f'On line {i}')
        row[column_name] = processing_fn(row[column_name])


def clean_html(text: str) -> str:
    tree = HTMLParser(text)
    remove_tags = set([child.tag for child in tree.body.traverse()]) - set(['body', 'code'])
    tree.unwrap_tags(list(remove_tags))
    clean_text = html.unescape(tree.body.html)[6:-7]
    return clean_text


def clean_csv(csv_path, csv_column_names):
    csv_data = read_csv(csv_path, csv_column_names)
    process_csv(csv_data, clean_html, 'body')
    return csv_data


def main():
    print(TAGS_TO_REMOVE)
    csv_path = (
        '/work/jkillingback_umass_edu/data/stack-overflow-data/large_answers.csv'
    )
    # csv_column_names = [
    #     'id',
    #     'title',
    #     'tags',
    #     'body',
    #     'acceptedAnswerId',
    #     'score',
    #     'views',
    # ]
    csv_column_names = ['id', 'questionId', 'body', 'score']
    output_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/large_answers_clean.csv'
    cleaned_csv = clean_csv(csv_path, csv_column_names)
    write_csv(cleaned_csv, output_path, csv_column_names)

if __name__ == '__main__':
    main()

import requests
import json
import time

from github import Github


file1 = open("links-between-papers-and-code.json", "r")
file2=open("papers-with-abstracts.json","r")
fileOutput=open("GithubRepoData.txt","w+")

jsonData=json.load(file1)
file2Json=json.load(file2)

file2Dict={}

for i in file2Json:
    temp=i["paper_url"]
    file2Dict[temp]=(i["tasks"],i["methods"])
g = Github("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")

c=0
for i in jsonData:
    c+=1
    print('progress',c)
    d={}
    stls=i["repo_url"].split("/")
    paperUrl=i["paper_url"]
    v1,v2=stls[-2],stls[-1]
    repositName=f"{v1}/{v2}"
    try:
        repo=g.get_repo(repositName)
        d['topics']=repo.get_topics()
    except Exception as e:
        print(e)
        continue
    d['name']=repo.name
    d['description']=repo.description
    #d['topics']=repo.get_topics()
    d['html_url']=repo.html_url
    d['created_at']=str(repo.created_at)
    d['updated_at']=str(repo.updated_at)
    file2Temp=file2Dict.get(paperUrl,None)

    if(file2Temp):
        d['tasks']=file2Temp[0]
        d['methods']=file2Temp[1]
    else:
        d['tasks']=[]
        d['methods']=[]
    
    fileOutput.write(json.dumps(d)+'\n')

    time.sleep(1.6)

print('successfully generated data')

import json 

## TC
for id in range(1, 11):
    file_path = 'dataset/TC1-all_including-GT/TC1-{}/1-{}-gemsec.dat'.format(id, id)

    with open('dataset/TC1-all_including-GT/TC1-{}/1-{}-gemsec.json'.format(id, id)) as f:
        data = json.load(f)
        
    print(data)

    with open(file_path, 'w') as file:
        for key, value in data.items():
            key = int(key) + 1
            file.write("{}    {}\n".format(key, value))

## Real world
folders = ['dolphin', 'football', 'karate', 'mexican', 'polbooks', 'railway', 'strike']

for folder in folders: 
    file_path = 'dataset/real-world dataset/{}/gemsec-regu.dat'.format(folder)
    
    with open('dataset/real-world dataset/{}-gemsec-regu.json'.format(folder)) as f:
        data = json.load(f)
        
    print(data)

    with open(file_path, 'w') as file:
        for key, value in data.items():
            key = int(key) + 1
            file.write("{}    {}\n".format(key, value))

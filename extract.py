import os
import csv
import json

data = {}

all_files = os.listdir(os.getcwd() + "/files")

for filename in all_files:
    with open(os.getcwd() + "\\files\\" + filename, "r", encoding="cp1252") as csvfile:
        csvreader = csv.reader(csvfile)
        for i in range(5):
            next(csvreader)

        state_data = {}

        for row in csvreader:
            state = row[1].title()
            if len(state) == 0:
                continue
            if state not in data.keys():
                data[state] = {}
            state_data = data[state]

            district = row[3].title()
            if len(district) == 0:
                continue

            if district not in state_data.keys():
                state_data[district] = {}
            dist_data = state_data[district]
            
            mandal = row[5].title()
            if len(mandal) == 0:
                dist_data["All Blocks"] = ["All Villages"]
                continue
            if mandal not in dist_data.keys():
                dist_data[mandal] = []
            man_data = dist_data[mandal]

            village = row[7].title()
            man_data.append(village)

with open('data.txt', 'w') as file:
    file.write(json.dumps(data))
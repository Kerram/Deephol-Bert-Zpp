# TODO: Incorporate into create_pretraining_data

import json

json_txt = ""

with open("train.json", "r") as f:
	json_txt = f.read()	

lis = json.loads(json_txt)

with open("train_data.txt", "w") as f:
	for sample in lis:
		for thm in sample["thms"]:
			f.write(thm)
			f.write("\n")

		for goal in sample["goal_asl"]:
			f.write(goal)
			f.write("\n")

		f.write(sample["goal"])
		f.write("\n")


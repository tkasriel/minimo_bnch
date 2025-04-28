import json

with open("saved_outputs/outcomes_8_simplifyseed.json") as f:
    out_dict = json.load(f)

out_dict = [i for i in out_dict if not i["hindsight"]]

with open("saved_outputs/outcomes_8_simplifyseed_nohindsight.json", "w") as f:
    json.dump(out_dict, f)
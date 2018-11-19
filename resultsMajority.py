import sys
import os

directory_in_str = ""
if len(sys.argv) > 1:
    directory_in_str = sys.argv[1]
else:
    directory_in_str = "resultsBallots"

directory = os.fsencode(directory_in_str)
out_file = open("resultsMajority.txt", "w")

files = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        files.append(open(directory_in_str + "\\" + filename, "r"))

matched_lines = []
for f in files:
    for i, line in enumerate(f):
        if len(matched_lines) < i + 1:
            matched_lines.append([])
        matched_lines[i].append(line)

for lines in matched_lines:
    out_line = None
    votes = {}
    for line in lines:
        split = line.split()
        if not out_line:
            out_line = split[0]
        for i in range(1, len(split)):
            votes.setdefault(split[i], 0)
            votes[split[i]] += 1
    votes = [(k, v) for k, v in votes.items()]
    votes = sorted(votes, key=lambda t: t[1], reverse=True)
    for tup in votes[0:5]:
        out_line += " " + tup[0]
    out_file.write(out_line + "\n")

out_file.seek(out_file.tell() - 2)
out_file.truncate()
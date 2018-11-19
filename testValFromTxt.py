import sys

in_file = None
if len(sys.argv) > 1:
    in_file = open(sys.argv[1].replace(".txt", "") + ".txt", "r")
else:
    # in_file = open("resultsVAL.txt", "r")
    in_file = open("resultsMajority.txt", "r")

num1 = 0
num5 = 0
total = 0
for line in in_file:
    if not line:
        continue
    split = line.split(" ")
    label = int(split[0].split("/")[2].split("\\")[1])
    top5 = split[1:6]
    top5[4] = top5[4].replace("\n", "")
    top5 = [int(x) for x in top5]
    top1 = top5[0]
    # print(label, top5)
    num1 += 1 if label == top1 else 0
    num5 += 1 if label in top5 else 0
    total += 1
print("TOP 1:", str(num1 / total))
print("TOP 5:", str(num5 / total))

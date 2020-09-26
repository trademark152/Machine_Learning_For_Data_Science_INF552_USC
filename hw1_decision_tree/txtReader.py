import csv

with open('dt_data.txt', 'r') as in_file:
    stripped = []
    count=1
    for line in in_file:
        # 1st line
        if count==1:
            line.strip("\n\r") # remove leading and trailing white spaces
            line = line[1:-2] # remove leading and trailing (,)
        else:
            line = line[4:-2]  # remove leading "id:" and trailing ;
        line = line.replace(" ", "")  # remove white spaces between words
        # print(line)
        stripped.append(line)
        count+=1


    lines = (line.split(",") for line in stripped if line)

    with open('dt_data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

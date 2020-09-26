import Node

data = []
f = open('Weather.csv')

for line in f:
	line = line.strip("\r\n")
	data.append(line.split(','))
tree = {'outlook': {'overcast': 'yes', 'rainy': {'temperature': {'mild': {'humidity': {'high': {'windy': {'FALSE': 'yes', 'TRUE': 'no'}}, 'normal': 'yes'}}, 'cool': {'humidity': {'normal': {'windy': {'FALSE': 'yes', 'TRUE': 'no'}}}}}}, 'sunny': {'temperature': {'hot': 'no', 'mild': {'humidity': {'high': 'no', 'normal': 'yes'}}, 'cool': 'yes'}}}}
attributes = ['outlook', 'temperature', 'humidity', 'windy', 'play']
count = 0
for entry in data:
	count += 1
	tempDict = tree.copy()
	result = ""
	while(isinstance(tempDict, dict)):
		root = Node.Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
		tempDict = tempDict[list(tempDict.keys())[0]]
		index = attributes.index(root.value)
		value = entry[index]
		if(value in tempDict.keys()):
			child = Node.Node(value, tempDict[value])
			result = tempDict[value]
			tempDict = tempDict[value]
		else:
			print ("can't process input %s" % count)
			result = "?"
			break
	print ("entry%s = %s" % (count, result)
)
def read(splitPath):
	trainSet = []
	validationSet = []
	with open(splitPath) as fp:
		for i in range(1, 22873):
			if fp.readline()[0] == '1':
				trainSet.append(i)
			else:
				validationSet.append(i)
	return (trainSet, validationSet)

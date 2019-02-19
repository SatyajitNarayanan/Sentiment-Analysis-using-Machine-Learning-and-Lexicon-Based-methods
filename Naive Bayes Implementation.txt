

# Naive Bayes Classification code

#Define function to split dataset with ratio
def Datasplit(data, Ratiosplit):
   X = data.ix[:, 'Polarity':'label'].values
   train, test = train_test_split(X, test_size=(1-Ratiosplit))
   train=train.tolist()
   test=test.tolist()
   return [train,test]


# Calculating the mean and standard deviation for every class
def summ(dataset):
	summaries = [(np.mean(numbers), np.var(numbers)) for numbers in zip(*dataset)]
	del summaries[-1]
	return summaries

def Classwisesumm(dataset):
    dataclass = {}
    for i in range(len(dataset)):
        loop = dataset[i]
        if (loop[-1] not in dataclass):
            dataclass[loop[-1]] = []
        dataclass[loop[-1]].append(loop)
        
    measures = {}
    for clas, vals in dataclass.items():
        measures[clas] = summ(vals)
    return measures


#Defining the Gaussian Probability Density Function
def prob(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


#Classify the test data based on likelihood ratio

def Classify(s, testin):
    def makeprediction(s, testin):
        ClassProb = {}
        for cv, cs in s.items():
            ClassProb[cv] = 1
            for i in range(len(cs)):
                mean, stdev = cs[i]
                x = testin[i]
                ClassProb[cv] *= prob(x, mean, stdev) #Calculating class probability
        labelv=[]
        llprob=[]
        likelihood=[]
        labelfinal= None
        bp = -10
        for i in ClassProb.keys():
            labelv.append(i)
        for i in ClassProb.values():
            llprob.append(i)
        for i in range(len(llprob)):
            likelihood.append(llprob[i]/sum(llprob)) #Calculating likelihood
        for i in range(len(labelv)):
            if labelfinal is None or likelihood[i]>bp: 
                bp = likelihood[i] #Assigning likelihood
                labelfinal = labelv[i]
        return labelfinal    
    classifications = []
    for i in range(len(testin)):
        result = makeprediction(s, testin[i])
        classifications.append(result)
    return classifications


# Running the classifier and calculating the accuracy of predictions

def NVB(dataset):
	training, test = Datasplit(dataset, Ratiosplit) #Calling split function
	print('{0} rows are split into {1} rows for training and {2} rows for testing'.format(len(dataset),len(training),len(test)))
	model = Classwisesumm(training)
	classification = Classify(model, test)
	Diagonal = 0
	for x in range(len(test)): #Calculating accuracy of classification
            if test[x][-1] == classification[x]:
                Diagonal += 1
	accuracy = (Diagonal/float(len(test)))*100.0
	print('Accuracy obtained = {0}%'.format(accuracy))


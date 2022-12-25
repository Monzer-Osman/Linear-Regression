import numpy as np
import pandas as pd
# from scipy.optimize import minimize
import numpy as np
import pandas as pd

def main():
    #TODO put the data file path in path variable 
    path = "" 
    columnsNames = ["symboling", "wheelbase", "carlength","carwidth",
                    "carheight", "curbweight", "enginesize", "boreratio",
                    "stroke", "compressionratio", "horsepower", "peakrpm",
                    "citympg", "highwaympg", "price"]

    modelRecords = readDataFromCsvFile(path, columnsNames)
    modelRecords = modelRecords.iloc[1:, 0:]
    changeColumnsType(modelRecords, columnsNames)

    modelRecords = scaleData(modelRecords)
    modelRecords = addColumn(modelRecords, 0, "Ones", 1)

    modelRecordsColumnLength = modelRecords.shape[1]
    
    xValues = getSubSetOfData(modelRecords, 0, modelRecordsColumnLength - 1)
    yValues = getSubSetOfData(modelRecords, modelRecordsColumnLength - 1, modelRecordsColumnLength)

    matrixOf_X_Values = getMatrixOfData(xValues)
    matrixOf_Y_Values = getMatrixOfData(yValues)
    matrixOfThetaValues = np.matrix(np.zeros(modelRecordsColumnLength - 1))

    alpha = 0.1
    iterations = 1000

    computedWeightValues, costValues = gradientDescent(matrixOf_X_Values, matrixOf_Y_Values, matrixOfThetaValues, alpha, iterations)    
    totalCostValue = computeCost(matrixOf_X_Values, matrixOf_Y_Values, computedWeightValues)

    print()
    print("the precentage of contribution for each future in predincting the prices : ")
    for i in range(computedWeightValues.shape[1] - 1):
        print(columnsNames[i],' : {0:.2f}%'.format((computedWeightValues[0, i+1] / computedWeightValues[0,1:].sum())))
    print()

    # print('weight values = \n' , computedWeightValues)
    # print()
    # print('cost values = \n' , costValues[0:50])
    # print()
    # print('total Cost value = ' , totalCostValue)
    # print('**************************************')

def readDataFromCsvFile(path, listOfColumnsNames):
    return pd.read_csv(path, header = None, names = listOfColumnsNames)

def addColumn(modelRecords, position, columnName, value):
    modelRecords.insert(position, columnName, value)
    return modelRecords

def changeColumnsType(modelRecords, columnsName):
    for name in columnsName:
        modelRecords[name] = modelRecords[name].astype(str).astype(float)

def scaleData(modelRecords):
    return (modelRecords - modelRecords.mean()) / modelRecords.std()

def getSubSetOfData(modelRecords, firstIndex, lastIndex):
    return modelRecords.iloc[0: , firstIndex: lastIndex] 

def getMatrixOfData(trainingData):
    return np.matrix(trainingData.values)

def computeCost(xValues, yValues, thetaValues):
    costEquationStep1 = np.power(((xValues * thetaValues.T) - yValues), 2)
    costEquationStep2 = np.sum(costEquationStep1)
    costEquationFinalstep = costEquationStep2 / (2 * len(xValues))
    return costEquationFinalstep

def gradientDescent(xValues, yValues, theta, alpha, iterations):

    zerosMatrix = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iterations)
    
    for i in range(iterations):
        error = (xValues * theta.T) - yValues    
        for j in range(parameters):
            term = np.multiply(error, xValues[:,j])
            zerosMatrix[0, j] = theta[0, j] - ((alpha / len(xValues)) * np.sum(term))
            
        theta = zerosMatrix
        cost[i] = computeCost(xValues, yValues, theta)
        
    return theta, cost

if __name__ == '__main__': 
    main()

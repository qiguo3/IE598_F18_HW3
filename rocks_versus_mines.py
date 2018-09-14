#Listing 2-1: Sizing Up a New Data Set—rockVmineSummaries.py (Output: outputRocksVMinesSummaries.txt)
__author__ = 'mike_bowles'
import urllib.request
import sys

#read data from uci data repositor

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

data = urllib.request.urlopen(target_url)

#arrange data into list for labels and list of lists for attributes
xList = []
labels = [] 
for line in data: 
     #split on comma 
     line= bytes.decode(line)
     row = line.strip().split(",") 
     xList.append(row)

nrow = len(xList) 
ncol = len(xList[1])
sys.stdout.write("Number of Rows of Data = " + str(nrow) + '\n')  
sys.stdout.write("Number of Columns of Data = " + str(ncol)+ '\n\n') 
 
"""
Output:  
Number of Rows of Data = 208  
Number of Columns of Data = 61 
"""

# Listing 2-2: Determining the Nature of Attributes—rockVmineContents.py (Output: outputRocksVMinesContents.txt)
type = [0]*3
colCounts = [] 

for col in range(ncol):
    for row in xList: 
        try: 
            a = float(row[col]) 
            if isinstance(a, float): 
                type[0] += 1 
        except ValueError: 
            if len(row[col]) > 0: 
                type[1] += 1 
            else:
                type[2] += 1 
    
    colCounts.append(type)
    type = [0]*3 

sys.stdout.write("Col#" + '\t' + "Number" + '\t' +                  
                 "Strings" + '\t ' + "Other\n")

iCol = 0 
for types in colCounts: 
    sys.stdout.write(str(iCol) + '\t' + str(types[0]) + '\t' +                      
                     str(types[1]) + '\t' + str(types[2]) + "\n")     
    iCol += 1

"""
output:
Col#    Number  Strings  Other
0       208     0       0
1       208     0       0
2       208     0       0
...
58      208     0       0
59      208     0       0
60      0       208     0
"""

import numpy as np 

#Listing 2-3: Summary Statistics for Numeric and Categorical Attributes—rVMSummaryStats.py (Output: outputSummaryStats.txt)   
col = 3  
colData = [] 
for row in xList:
    colData.append(float(row[col]))

colArray = np.array(colData) 
colMean = np.mean(colArray) 
colsd = np.std(colArray) 
sys.stdout.write('\n'+"Mean = " + '\t' + str(colMean) + '\n' +              
                 "Standard Deviation = " + '\t ' + str(colsd) + "\n\n")
"""
Mean =  0.053892307692307684
Standard Deviation =     0.04641598322260027
"""

#calculate quantile boundaries
ntiles = 4
percentBdry = [] 
for i in range(ntiles+1):   
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")  
print(percentBdry)  
sys.stdout.write(" \n")
"""
Boundaries for 4 Equal Percentiles 
[0.0058, 0.024375, 0.04405, 0.0645, 0.4264]
"""

#run again with 10 equal intervals 
ntiles = 10
percentBdry = [] 
for i in range(ntiles+1):   
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("\nBoundaries for 10 Equal Percentiles \n")  
print(percentBdry)  
sys.stdout.write(" \n")
"""
Boundaries for 10 Equal Percentiles 
[0.0058, 0.0141, 0.022740000000000003, 0.027869999999999995, 0.03622, 0.04405, 0.05071999999999999, 0.059959999999999986, 0.07794000000000001, 0.10836, 0.4264]
"""

#The last column contains categorical variables
col = 60
colData = []
for row in xList:      
    colData.append(row[col]) 

unique = set(colData)
sys.stdout.write("Unique Label Values \n") 
print(unique)
"""
Unique Label Values 
{'R', 'M'}
"""

#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*2 
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique)) 
print(catCount)                    
         
"""
Counts for Each Value of Categorical Label 
['R', 'M']
[97, 111]
"""

#Listing 2-4: Quantile-Quantile Plot for 4th Rocks versus Mines Attribute— qqplotAttribute.py 
#generate summary statistics for column 3 (e.g.) 
import pylab 
import scipy.stats as stats

type = [0]*3 
colCounts = [] 
#generate summary statistics for column 3 (e.g.) 
col = 3  
colData = [] 
for row in xList: 
    colData.append(float(row[col]))

stats.probplot(colData, dist="norm", plot=pylab) 
pylab.show()

# Listing 2-5: Using Python Pandas to Read and Summarize Data—pandasReadSummarize.py
import pandas as pd  
from pandas import DataFrame  
import matplotlib.pyplot as plot 

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")

#read rocks versus mines data into pandas data frame 
rocksVMines = pd.read_csv(target_url,header=None, prefix="V")
 
#print head and tail of data frame 
print(rocksVMines.head())
print(rocksVMines.tail())

#print summary of data frame  
summary = rocksVMines.describe()  
print(summary) 


# Listing 2-6: Parallel Coordinates Graph for Real Attribute Visualization—linePlots.py
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame 
rocksVMines = pd.read_csv(target_url,header=None, prefix="V") 
for i in range(208): 
    #assign color based on "M" or "R" labels   
    if rocksVMines.iat[i,60] == "M":         
        pcolor = "red" 
    else:         
        pcolor = "blue"
    #plot rows of data as if they were series data
    dataRow = rocksVMines.iloc[i,0:60]
    dataRow.plot(color=pcolor)  

plot.xlabel("Attribute Index") 
plot.ylabel(("Attribute Values")) 
plot.show() 
  
  
# Listing 2-7: Cross Plotting Pairs of Attributes—corrPlot.py
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame 
rocksVMines = pd.read_csv(target_url,header=None, prefix="V") 

#calculate correlations between real-valued attributes  
dataRow2 = rocksVMines.iloc[0:60,1] 
dataRow3 = rocksVMines.iloc[0:60,2]
plot.scatter(dataRow2, dataRow3) 
plot.xlabel("2nd Attribute") 
plot.ylabel("3rd Attribute")  
plot.show()

dataRow21 = rocksVMines.iloc[0:60,20]
plot.scatter(dataRow2, dataRow21) 
plot.xlabel("2nd Attribute")  
plot.ylabel("21st Attribute") 
plot.show()

# Listing 2-8: Correlation between Classiﬁ cation Target and Real Attributes—targetCorr.py  
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame 
rocksVMines = pd.read_csv(target_url,header=None, prefix="V") 

#change the targets to numeric values
target = [] 
for i in range(208):     
    #assign 0 or 1 target value based on "M" or "R" labels      
    if rocksVMines.iat[i,60] == "M":         
        target.append(1.0)
    else:         
        target.append(0.0)
#plot 35th attribute dataRow = rocksVMines.iloc[0:208,35] plot.scatter(dataRow, target) 
dataRow = rocksVMines.iloc[0:208,35] 
plot.scatter(dataRow, target) 

plot.xlabel("Attribute Value")  
plot.ylabel("Target Value") 
plot.show()
 

#To improve the visualization, this version dithers the points a little and makes them somewhat transparent
from random import uniform
target = [] 
for i in range(208):
#assign 0 or 1 target value based on "M" or "R" labels and add some dither
    if rocksVMines.iat[i,60] == "M":         
        target.append(1.0 + uniform(-0.1, 0.1))     
    else:         
        target.append(0.0 + uniform(-0.1, 0.1))
#plot 35th attribute with semi-opaque points dataRow = rocksVMines.iloc[0:208,35] plot.scatter(dataRow, target, alpha=0.5, s=120)
dataRow = rocksVMines.iloc[0:208,35] 
plot.scatter(dataRow, target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")  
plot.ylabel("Target Value") 
plot.show()

#Listing 2-9: Pearson’s Correlation Calculation for Attributes 2 versus 3 and 2 versus 21— corrCalc.py  
__author__ = 'mike_bowles' 
import pandas as pd  
from math import sqrt  
import sys 
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
                         "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame 
rocksVMines = pd.read_csv(target_url,header=None, prefix="V")
 
#calculate correlations between real-valued attributes  
dataRow2 = rocksVMines.iloc[1,0:60] 
dataRow3 = rocksVMines.iloc[2,0:60] 
dataRow21 = rocksVMines.iloc[20,0:60]

mean2 = 0.0; mean3 = 0.0; mean21 = 0.0 
numElt = len(dataRow2) 
for i in range(numElt):     
    mean2 += dataRow2[i]/numElt      
    mean3 += dataRow3[i]/numElt      
    mean21 += dataRow21[i]/numElt  

var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):     
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2)/numElt     
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3)/numElt     
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21)/numElt 

corr23 = 0.0; corr221 = 0.0 
for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * (dataRow3[i] - mean3) / (sqrt(var2*var3) * numElt)     
    corr221 += (dataRow2[i] - mean2) * (dataRow21[i] - mean21) / (sqrt(var2*var21) * numElt) 

sys.stdout.write("Correlation between attribute 2 and 3 \n") 
print(corr23) 
sys.stdout.write(" \n")

sys.stdout.write("Correlation between attribute 2 and 21 \n")  
print(corr221)  
sys.stdout.write(" \n") 
 
# Listing 2-10: Presenting Attribute Correlations Visually—sampleCorrHeatMap.py  
__author__ = 'mike_bowles'
import pandas as pd  
import matplotlib.pyplot as plot  
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-" 
              "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame 
rocksVMines = pd.read_csv(target_url,header=None, prefix="V") 

#calculate correlations between real-valued attributes  
corMat = DataFrame(rocksVMines.corr())

#visualize correlations using heatmap
plot.pcolor(corMat)  
plot.show() 

print("My name is QI GUO")
print("My NetID is: qiguo3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################












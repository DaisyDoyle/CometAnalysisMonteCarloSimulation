import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#
nLen = 11
lenArr = np.zeros(nLen)
mobArrA = np.zeros(nLen) 
velArrA = np.zeros(nLen) 
mobArrB = np.zeros(nLen) 
velArrB = np.zeros(nLen) 
mobArrC = np.zeros(nLen) 
velArrC = np.zeros(nLen) 
mobArrD = np.zeros(nLen) 
velArrD = np.zeros(nLen) 
mobArrE = np.zeros(nLen) 
velArrE = np.zeros(nLen) 
#
lenArr[0] = 0.31 # kbp
lenArr[1] = 0.59 # kbp
lenArr[2] = 0.86 # kbp
lenArr[3] = 1.07 # kbp
lenArr[4] = 1.28 # kbp
lenArr[5] = 1.97 # kbp
lenArr[6] = 2.24 # kbp
lenArr[7] = 4.28 # kbp
lenArr[8] = 6.59 # kbp
lenArr[9] = 9.31 # kbp
lenArr[10] = 23.01 # kbp
#
scale = 3/6.35 # convert cm to cm**2/V
print("Mobility scaling factor is {:.2f} (cm**2/V)/cm ".format(scale))
#
EA, EC, EE = 0.05, 0.3, 0.8 # V/cm
#
mobArrA[0],  mobArrC[0],  mobArrE[0]  = 5.1,  5.1,  5.1   # cm read off graph
mobArrA[1],  mobArrC[1],  mobArrE[1]  = 4.6,  4.6,  4.6   # cm
mobArrA[2],  mobArrC[2],  mobArrE[2]  = 4.25, 4.25, 4.25  # cm
mobArrA[3],  mobArrC[3],  mobArrE[3]  = 3.9,  4.0,  4.05  # cm
mobArrA[4],  mobArrC[4],  mobArrE[4]  = 3.6,  3.75, 3.9   # cm
mobArrA[5],  mobArrC[5],  mobArrE[5]  = 3.0,  3.2,  3.4   # cm
mobArrA[6],  mobArrC[6],  mobArrE[6]  = 2.8,  3.0,  3.25  # cm
mobArrA[7],  mobArrC[7],  mobArrE[7]  = 1.85, 2.1,  2.6   # cm
mobArrA[8],  mobArrC[8],  mobArrE[8]  = 1.3,  1.75, 2.2   # cm
mobArrA[9],  mobArrC[9],  mobArrE[9]  = 0.9,  1.45, 1.85  # cm
mobArrA[10], mobArrC[10], mobArrE[10] = 0.14, 1.05, 1.6   # cm
#
EB, ED = 0.15, 0.5 # V/cm
#
mobArrB[0],  mobArrD[0]  = 5.1,  5.1   # cm read off graph
mobArrB[1],  mobArrD[1]  = 4.6,  4.6   # cm
mobArrB[2],  mobArrD[2]  = 4.25, 4.25  # cm
mobArrB[3],  mobArrD[3]  = 3.9,  4.0   # cm
mobArrB[4],  mobArrD[4]  = 3.65, 3.8   # cm
mobArrB[5],  mobArrD[5]  = 3.05, 3.3   # cm
mobArrB[6],  mobArrD[6]  = 2.8,  3.15  # cm
mobArrB[7],  mobArrD[7]  = 1.9,  2.4   # cm
mobArrB[8],  mobArrD[8]  = 1.4,  2.0   # cm
mobArrB[9],  mobArrD[9]  = 1.1,  1.7   # cm
mobArrB[10], mobArrD[10] = 0.7,  1.35  # cm
#
mobArrA[0:nLen] = scale*mobArrA[0:nLen]  # cm**2/V
mobArrB[0:nLen] = scale*mobArrB[0:nLen]  # cm**2/V
mobArrC[0:nLen] = scale*mobArrC[0:nLen]  # cm**2/V
mobArrD[0:nLen] = scale*mobArrD[0:nLen]  # cm**2/V
mobArrE[0:nLen] = scale*mobArrE[0:nLen]  # cm**2/V
#
velArrA = EA*mobArrA # cm/s
velArrB = EB*mobArrB # cm/s
velArrC = EC*mobArrC # cm/s
velArrD = ED*mobArrD # cm/s
velArrE = EE*mobArrE # cm/s
#
# The code below prints out the data we have entered
print(" ")
print("Data points, E field {:.2f} V/cm".format(EA))
print("Point\t Length (kbp)\t Mobility (cm**2/V*10^4)\t Velocity (cm/s*10^4)")
for n in range(0, nLen):
    print("{:d}\t{:5.2f}\t\t{:5.2f}\t\t\t\t{:5.2f}".format(n, lenArr[n], mobArrA[n], velArrA[n]))
print(" ")
print("Data points, E field {:.2f} V/cm".format(EB))
print("Point\t Length (kbp)\t Mobility (cm**2/V*10^4)\t Velocity (cm/s*10^4)")
for n in range(0, nLen):
    print("{:d}\t{:5.2f}\t\t{:5.2f}\t\t\t\t{:5.2f}".format(n, lenArr[n], mobArrB[n], velArrB[n]))
#
print(" ")
print("Data points, E field {:.2f} V/cm".format(EC))
print("Point\t Length (kbp)\t Mobility (cm**2/V*10^4)\t Velocity (cm/s*10^4)")
for n in range(0, nLen):
    print("{:d}\t{:5.2f}\t\t{:5.2f}\t\t\t\t{:5.2f}".format(n, lenArr[n], mobArrC[n], velArrC[n]))
#
print(" ")
print("Data points, E field {:.2f} V/cm".format(ED))
print("Point\t Length (kbp)\t Mobility (cm**2/V*10^4)\t Velocity (cm/s*10^4)")
for n in range(0, nLen):
    print("{:d}\t{:5.2f}\t\t{:5.2f}\t\t\t\t{:5.2f}".format(n, lenArr[n], mobArrD[n], velArrD[n]))
#
print(" ")
print("Data points, E field {:.2f} V/cm".format(EE))
print("Point\t Length (kbp)\t Mobility (cm**2/V*10^4)\t Velocity (cm/s*10^4)")
for n in range(0, nLen):
    print("{:d}\t{:5.2f}\t\t{:5.2f}\t\t\t\t{:5.2f}".format(n, lenArr[n], mobArrE[n], velArrE[n]))
print(" ")
plt.figure(figsize = (16, 5))
plt.subplot(1, 2, 1)
plt.title("Mobility of DNA")
plt.xlabel("Length (kbp)")
plt.ylabel("Mobility (cm$^2$/V x $10^4$) ")
plt.plot(lenArr, mobArrA, linestyle = '', marker = '+', color = 'r', label = "E = " + str(EA) + " V/cm")
plt.plot(lenArr, mobArrB, linestyle = '', marker = 'o', color = 'c', label = "E = " + str(EB) + " V/cm")
plt.plot(lenArr, mobArrC, linestyle = '', marker = '.', color = 'b', label = "E = " + str(EC) + " V/cm")
plt.plot(lenArr, mobArrD, linestyle = '', marker = '+', color = 'k', label = "E = " + str(ED) + " V/cm")
plt.plot(lenArr, mobArrE, linestyle = '', marker = 'o', color = 'r', label = "E = " + str(EE) + " V/cm")
plt.grid(color = 'g')
plt.xlim(0.1, 100)
plt.ylim(0, 3)
plt.legend()
plt.xscale('log')
#
plt.subplot(1, 2, 2)
plt.title("Velocity of DNA")
plt.xlabel("Length (kbp)")
plt.ylabel("Velocity (cm/s x $10^4$) ")
plt.plot(lenArr, velArrA, linestyle = '', marker = '+', color = 'r', label = "E = " + str(EA) + " V/cm")
plt.plot(lenArr, velArrB, linestyle = '', marker = 'o', color = 'c', label = "E = " + str(EB) + " V/cm")
plt.plot(lenArr, velArrC, linestyle = '', marker = '.', color = 'b', label = "E = " + str(EC) + " V/cm")
plt.plot(lenArr, velArrD, linestyle = '', marker = '+', color = 'k', label = "E = " + str(ED) + " V/cm")
plt.plot(lenArr, velArrE, linestyle = '', marker = 'o', color = 'r', label = "E = " + str(EE) + " V/cm")
plt.grid(color = 'g')
plt.xlim(0.1, 100)
plt.ylim(-0.1, 2)
plt.xscale('log')
plt.legend()
plt.show()

import sys
#
# nLen is the number of data points in the fit
nLen = 11
#
# Define NumPy arrays to store the x and y data values
xData1 = np.zeros(nLen)
yData1 = np.zeros(nLen)
#
# Enter the x values, the field strength and the y data values.
xData1 = lenArr[0:nLen]
E1 = EB
Etest = 0.01
if abs(E1 - EA) < Etest:
    yData1 = velArrA[0:nLen]
elif abs(E1 - EB) < Etest:
    yData1 = velArrB[0:nLen]
elif abs(E1 - EC) < Etest:
    yData1 = velArrC[0:nLen]
elif abs(E1 - ED) < Etest:
    yData1 = velArrB[0:nLen]
elif abs(E1 - EE) < Etest:
    yData1 = velArrC[0:nLen]
else:
    print("Chosen field strength doesn't match any available data - stop")
    sys.exit()
#
# Define arrays to store the errors in x and y
xError1 = np.zeros(nLen)
yError1 = np.zeros(nLen)
#
# Enter the errors in the x and y values
fXerr1 = 0.01
fYerr1 = 0.01
xError1 = fXerr1*np.ones(nLen)
yError1 = fYerr1*np.ones(nLen)
#
# The code below prints out the data we have entered
print(" ")
print("Length and velocity data for electric field {:.2f} V/cm".format(E1))
print("Point\t Length (kbp)\t\t Velocity (cm/s * 10^4)\t")
for n in range(0, nLen):
    print("{:d}\t{:5.2f}\t+-{:5.2f}\t\t{:5.2f}\t+-{:5.2f}".\
          format(n, xData1[n], xError1[n], yData1[n], yError1[n]))
    
from scipy.optimize import least_squares
#
# kB = p[0]
# kC = p[1]
# kL = p[2]
# alpha = p[3]
#
def fitFunc(p, E, L):
    '''
    Mobility function
    '''
    f = p[0]*E/(1 + p[1]*L) + p[2]*E**p[3] 
    #
    return f
#
def fitFuncDiff(p, E, L):
    '''
    Differential of mobility function w.r.t. L
    '''
    df = -p[0]*p[1]*E/(1 + p[1]*L)**2
    #
    return df
#
def fitError(p, x, E, y, xerr, yerr):
    '''
    Error function for straight line fit
    '''
    e = (y - fitFunc(p, E, x))/(np.sqrt(yerr**2 + fitFuncDiff(p, E, x)**2*xerr**2))
    return e
#
# Set initial values of fit parameters, run fit
nPar = 4
pInit = [1.0,     1.0,     1.0,     1.0]
loBnd = [-np.inf, -np.inf, -np.inf, -np.inf]
upBnd = [np.inf,  np.inf,  np.inf,  np.inf]
out = least_squares(fitError, pInit, args=(xData1, E1, yData1, xError1, yError1), bounds = (loBnd, upBnd))
#
fitOK = out.success
#
# Test if fit failed
if not fitOK:
    print(" ")
    print("Fit failed")
else:
    #
    # get output
    pFinal = out.x
    kBval = pFinal[0]
    kCval = pFinal[1]
    kLval = pFinal[2]
    alpha = pFinal[3]
    #
    #   Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr = fitError(pFinal, xData1, E1, yData1, xError1, yError1)**2
    chisq = np.sum(chiarr)
    NDF = nLen - nPar
    redchisq = chisq/NDF
#
    np.set_printoptions(precision = 3)
    print(" ")
    print("Fit quality:")
    print("chisq per point = \n",chiarr)
    print("chisq = {:5.2f}, chisq/NDF = {:5.2f}.".format(chisq, redchisq))
    #
    # Compute covariance
    jMat = out.jac
    jMat2 = np.dot(jMat.T, jMat)
    detJmat2 = np.linalg.det(jMat2)
    #
    if detJmat2 < 1E-32:
        print("Value of determinat detJmat2",detJmat2)
        print("Matrix singular, error calculation failed.")
        print(" ")
        print("Parameters returned by fit at E field {:.2f}:".format(E1))
        print("kB = {:5.2f}".format(kBval))
        print("kC = {:5.2f}".format(kCval))
        print("kL = {:5.2f}".format(kLval))
        print("alpha = {:5.2f}".format(alpha))
        print(" ")
        kBerr = 0.0
        kCerr = 0.0
        kLerr = 0.0
        alperr = 0.0
    else:
        covar = np.linalg.inv(jMat2)
        kBerr = np.sqrt(covar[0, 0])
        kCerr = np.sqrt(covar[1, 1])
        kLerr = np.sqrt(covar[2, 2])
        alperr = np.sqrt(covar[3, 3])
        #
        print(" ")
        print("Parameters returned by fit at E field {:.2f}:".format(E1))
        print("kB = {:5.2f} +- {:5.2f}".format(kBval, kBerr))
        print("kC = {:5.2f} +- {:5.2f}".format(kCval, kCerr))
        print("kL = {:5.2f} +- {:5.2f}".format(kLval, kLerr))
        print("alpha = {:5.2f} +- {:5.2f}".format(alpha, alperr))
        print(" ")
    #
    # Calculate fitted function values
    #
    nPlot = 100
    xBot = 0.1
    xTop = 100.0
    xPlot = np.exp(np.linspace(np.log(xBot), np.log(xTop), nPlot))
    pPlot = pFinal
    fitPlot1 = fitFunc(pPlot, E1, xPlot)
    #
    # Plot data
    fig = plt.figure(figsize = (8, 6))
    plt.title('Data with fit for field ' + str(E1) + ' V/cm')
    plt.xlabel('Length (kbp)')
    plt.ylabel('Speed (cm/s x $10^4$)')
    plt.errorbar(xData1, yData1, xerr = xError1, yerr = yError1, fmt='r', \
                 linestyle = '', label = "Data") 
    plt.plot(xPlot, fitPlot1, color = 'b', linestyle = '-', label = "Fit")
    plt.xlim(0.1, 100)
#    plt.ylim(0, 3)
    plt.xscale('log')
    plt.grid(color = 'g')
    plt.legend()
#    plt.savefig(FitPlot.png")
    plt.show()
  
  import sys
#
# nEf is the number of data points in the fit
nEf = 5
#
# Define NumPy arrays to store the x and y data values
xData2 = np.zeros(nEf)
yData2 = np.zeros(nEf)
#
# Enter the x values, the field strength and the y data values.
Lindex = 4
L1 = lenArr[Lindex]
xData2 = np.array([EA, EB, EC, ED, EE])
yData2 = np.array([velArrA[Lindex], velArrB[Lindex], velArrC[Lindex], velArrD[Lindex], velArrE[Lindex]])
#
# Define arrays to store the errors in x and y
xError2 = np.zeros(nEf)
yError2 = np.zeros(nEf)
#
# Enter the errors in the x and y values
fXerr2 = 0.01
fYerr2 = 0.01
xError2 = fXerr2*np.ones(nEf)
yError2 = fYerr2*np.ones(nEf)
#
# The code below prints out the data we have entered
print(" ")
print("Field and velocity data for length {:.2f} kbp".format(L1))
print("Point\t Field (V/cm)\t\t Velocity (cm/s * 10^4)\t")
for n in range(0, nEf):
    print("{:d}\t{:5.2f}\t+-{:5.2f}\t\t{:5.2f}\t+-{:5.2f}".\
          format(n, xData2[n], xError2[n], yData2[n], yError2[n]))
    
from scipy.optimize import least_squares
#
# kB = p[0]
# kC = p[1]
# kL = p[2]
# alpha = p[3]
#
def fitFunc(p, E, L):
    '''
    Mobility function
    '''
    f = p[0]*E/(1 + p[1]*L) + p[2]*E**p[3]
    #
    return f
#
def fitFuncDiff(p, E, L):
    '''
    Differential of mobility function w.r.t. E
    '''
    df = p[0]/(1 + p[1]*L) + p[2]*p[3]*E**(p[3] - 1)
    #
    return df
#
def fitError(p, x, L, y, xerr, yerr):
    '''
    Error function for straight line fit
    '''
    e = (y - fitFunc(p, x, L))/(np.sqrt(yerr**2 + fitFuncDiff(p, x, L)**2*xerr**2))
    return e
#
# Set initial values of fit parameters, run fit
nPar = 4
pInit = [1.0,     1.0,     1.0,     1.0]
loBnd = [-np.inf, -np.inf, -np.inf, -np.inf]
upBnd = [np.inf,  np.inf,  np.inf,  np.inf]
out = least_squares(fitError, pInit, args=(xData2, L1, yData2, xError2, yError2), bounds = (loBnd, upBnd))
#
fitOK = out.success
#
# Test if fit failed
if not fitOK:
    print(" ")
    print("Fit failed")
else:
    #
    # get output
    pFinal = out.x
    kBval = pFinal[0]
    kCval = pFinal[1]
    kLval = pFinal[2]
    alpha = pFinal[3]
    #
    #   Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr = fitError(pFinal, xData2, L1, yData2, xError2, yError2)**2
    chisq = np.sum(chiarr)
    NDF = nEf - nPar
    redchisq = chisq/NDF
#
    np.set_printoptions(precision = 3)
    print(" ")
    print("Fit quality:")
    print("chisq per point = \n",chiarr)
    print("chisq = {:5.2f}, chisq/NDF = {:5.2f}.".format(chisq, redchisq))
    #
    # Compute covariance
    jMat = out.jac
    jMat2 = np.dot(jMat.T, jMat)
    detJmat2 = np.linalg.det(jMat2)
    #
    if detJmat2 < 1E-32:
        print("Value of determinat detJmat2",detJmat2)
        print("Matrix singular, error calculation failed.")
        print(" ")
        print("Parameters returned by fit at E field {:.2f}:".format(E1))
        print("kB = {:5.2f}".format(kBval))
        print("kC = {:5.2f}".format(kCval))
        print("kL = {:5.2f}".format(kLval))
        print("alpha = {:5.2f}".format(alpha))
        print(" ")
        kBerr = 0.0
        kCerr = 0.0
        kLerr = 0.0
        alperr = 0.0
    else:
        covar = np.linalg.inv(jMat2)
        kBerr = np.sqrt(covar[0, 0])
        kCerr = np.sqrt(covar[1, 1])
        kLerr = np.sqrt(covar[2, 2])
        alperr = np.sqrt(covar[3, 3])
        #
        print(" ")
        print("Parameters returned by fit at E field {:.2f}:".format(E1))
        print("kB = {:5.2f} +- {:5.2f}".format(kBval, kBerr))
        print("kC = {:5.2f} +- {:5.2f}".format(kCval, kCerr))
        print("kL = {:5.2f} +- {:5.2f}".format(kLval, kLerr))
        print("alpha = {:5.2f} +- {:5.2f}".format(alpha, alperr))
        print(" ")
    #
    # Calculate fitted function values
    #
    nPlot = 100
    xBot = 0.0
    xTop = 1.0
    xPlot = np.linspace(xBot, xTop, nPlot)
    fitPlot2 = fitFunc(pFinal, xPlot, L1)
    #
    # Plot data
    fig = plt.figure(figsize = (8, 6))
    plt.title('Data with fit for length ' + str(L1) + ' kbp')
    plt.xlabel('ELectric field (V/cm)')
    plt.ylabel('Speed (cm/s x $10^4$)')
    plt.errorbar(xData2, yData2, xerr = xError2, yerr = yError2, fmt='r', \
                 linestyle = '', label = "Data") 
    plt.plot(xPlot, fitPlot2, color = 'b', linestyle = '-', label = "Fit")
    plt.xlim(0.0, 1.0)
    plt.ylim(0, 3)
    plt.grid(color = 'g')
    plt.legend()
    #plt.savefig(FitPlot.png")
    plt.show()
    
    
from mpl_toolkits.mplot3d import Axes3D
#
# nY is the number of energy values, nX the number of points at each energy
nX = nLen
nY = nEf
nPoints = nX*nY
#
# Define NumPy arrays to store the x and y data values (L and E)
xData = np.zeros(nX)
yData = np.zeros(nY)
#
# Enter the x and y data
xData = lenArr
yData[0] = EA
yData[1] = EB
yData[2] = EC
yData[3] = ED
yData[4] = EE
#
# Define arrays to store the errors in x and y
xError = np.zeros(nX)
yError = np.zeros(nY)
#
# Enter the errors in the x and y values
fXerr = 0.01
fYerr = 0.01
xError = fXerr*np.ones(nX)
yError = fYerr*np.ones(nY)
#
# Set up x and y meshes
xMesh, yMesh = np.meshgrid(xData, yData)
xErrMesh, yErrMesh = np.meshgrid(xError, yError)
#
# Now z data
zMesh = np.zeros((nY, nX))
zErrMesh = np.zeros((nY, nX))
zMesh[0, :] = velArrA[:]
zMesh[1, :] = velArrB[:]
zMesh[2, :] = velArrC[:]
zMesh[3, :] = velArrD[:]
zMesh[4, :] = velArrE[:]
#
fZerr = 0.01
zErrMesh = fZerr*np.ones((nY, nX))
#
# The code below prints out the data we have entered
print(" ")
print("Check the data points:")
print("Index x\t\tIndex y\t\t Field (V/cm)\t Length (kbp)\t Velocity (cm/s * 10^4)\t")
for iX in range(0, nX):
    for iY in range(0, nY):
        print("{:d}\t\t{:d}\t\t{:5.2f}\t\t{:5.2f}\t\t{:5.2f}".\
              format(iX, iY, yMesh[iY, iX], xMesh[iY, iX], zMesh[iY, iX]))
    #
#
# Plot data
print("")
#
# Plot data
nColTab = 6
colTab = ['b', 'r', 'c', 'm', 'y', 'k']
#
fig = plt.figure(figsize = (16, 6))
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Velocity data')
ax.set_xlabel('Length (kbp)')
ax.set_ylabel('Velocity (cm/s x $10^4$)')
nCol = 0
for plot in range(0, nY):
    ax.errorbar(xMesh[plot, :], zMesh[plot, :], xerr = xErrMesh[plot, :], yerr = zErrMesh[plot, :], 
                color = colTab[nCol], linestyle = '', label = "E = " + str(yMesh[plot, 0]) + " V/cm") 
    nCol += 1
    if nCol > nColTab - 1:
        nCol = 0
ax.set_xlim(0.1, 100)
ax.set_ylim(0, 2.2)
ax.set_xscale('log')
ax.grid(color = 'g')
ax.legend()    
#   
ax = fig.add_subplot(1, 2, 2)
ax.set_title('Velocity data')
ax.set_xlabel('Field (V/cm)')
ax.set_ylabel('Velocity (cm/s x $10^4$)')
nCol = 0
for plot in range(0, nX):
    ax.errorbar(yMesh[:, plot], zMesh[:, plot], xerr = yErrMesh[:, plot], yerr = zErrMesh[:, plot], 
                color = colTab[nCol], linestyle = '', label = "Len = " + str(xMesh[0, plot]) + " kbp") 
    nCol += 1
    if nCol > nColTab - 1:
        nCol = 0
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0, 2.2)
ax.grid(color = 'g')
ax.legend()    
#
#plt.savefig(FitPlot.png")
plt.show()
#
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.set_title('Velocity data')
ax.plot_surface(xMesh, yMesh, zMesh, cmap = 'viridis', linewidth = 1, antialiased = True)
ax.set_xlabel("Length (kbp)")
ax.set_ylabel("Field (V/cm)")
ax.set_zlabel("Velocity (cm/s x $10^4$)")
plt.show()


# kB = p[0]
# kC = p[1]
# kL = p[2]
# alpha = p[3]
#
def fitFunc(p, xMesh, yMesh):
    '''
    Mobility function
    '''
    f = p[0]*yMesh/(1 + p[1]*xMesh) + p[2]*yMesh**p[3]
    #
    return f
#
def fitFuncDiff(p, xMesh, yMesh):
    '''
    Differential of mobility function
    '''
    dfx = -p[0]*p[1]*yMesh/(1 + p[1]*xMesh)**2
    dfy = (p[0] + p[0]*p[1]*xMesh)/(1 + p[1]*xMesh)**2 + p[2]*p[3]*yMesh**(p[3] - 1)
   #
    return dfx, dfy
#
def fitError(p, xMesh, yMesh, zMesh, xErrMesh, yErrMesh, zErrMesh):
    '''
    Error function for fit
    '''
    fMesh = fitFunc(p, xMesh, yMesh)
    dfx, dfy = fitFuncDiff(p, xMesh, yMesh)
    #
    RMdevSq = (zMesh - fMesh)/np.sqrt(dfx**2*xErrMesh**2 + dfy**2*yErrMesh**2 + zErrMesh**2)
    flatRMdevSq = np.ndarray.flatten(RMdevSq)
    #
    return flatRMdevSq
#
# Set initial values of fit parameters, bounds and run fit
nPar = 4
pInit = [1.0,     1.0,     1.0,     1.0]
loBnd = [-np.inf, -np.inf, -np.inf, -np.inf]
upBnd = [np.inf,  np.inf,  np.inf,  np.inf]
#
out = least_squares(fitError, pInit, args = (xMesh, yMesh, zMesh, xErrMesh, yErrMesh, zErrMesh), bounds = (loBnd, upBnd))
#
fitOK = out.success
#
# Test if fit failed
if not fitOK:
    print(" ")
    print("Fit failed")
else:
    #
    # get output
    pFinal = out.x
    kBval = pFinal[0]
    kCval = pFinal[1]
    kLval = pFinal[2]
    alpha = pFinal[3]
    #
    #   Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr = fitError(pFinal, xMesh, yMesh, zMesh, xErrMesh, yErrMesh, zErrMesh)**2
    chisq = np.sum(chiarr)
    NDF = nPoints - nPar
    redchisq = chisq/NDF
#
    np.set_printoptions(precision = 3)
    print(" ")
    print("Fit quality:")
    print("chisq per point = \n",chiarr)
    print("chisq = {:5.2f}, chisq/NDF = {:5.2f}.".format(chisq, redchisq))
    #
    # Compute covariance
    jMat = out.jac
    jMat2 = np.dot(jMat.T, jMat)
    detJmat2 = np.linalg.det(jMat2)
    #
    if detJmat2 < 1e-64:
        print("Value of determinant detJmat2",detJmat2)
        print("Matrix singular, error calculation failed.")
        print(" ")
        print("kB = {:5.2f}".format(kBval))
        print("kC = {:5.2f}".format(kCval))
        print("kL = {:5.2f}".format(kLval))
        print("alpha = {:5.2f}".format(alpha))
        print(" ")
        kBerr = 0.0
        kCerr = 0.0
        kLerr = 0.0
        alperr = 0.0
    else:
        covar = np.linalg.inv(jMat2)
        kBerr = np.sqrt(covar[0, 0])
        kCerr = np.sqrt(covar[1, 1])
        kLerr = np.sqrt(covar[2, 2])
        alperr = np.sqrt(covar[3, 3])
        #
        print(" ")
        print("Parameters returned by fit:")
        print("kB = {:5.2f} +- {:5.2f}".format(kBval, kBerr))
        print("kC = {:5.2f} +- {:5.2f}".format(kCval, kCerr))
        print("kL = {:5.2f} +- {:5.2f}".format(kLval, kLerr))
        print("alpha = {:5.2f} +- {:5.2f}".format(alpha, alperr))
        print(" ")
    #
    # Calculate fitted function values x direction
    nPlot = 100
    xBot = 0.1
    xTop = 100.0
    pPlot = pFinal
    xPlot = np.exp(np.linspace(np.log(xBot), np.log(xTop), nPlot))
    #
    # Calculate fitted function values y direction
    yBot = 0.0
    yTop = 1.0
    yPlot = np.linspace(yBot, yTop, nPlot)
    #
    # Plot data
    nColTab = 6
    colTab = ['b', 'r', 'c', 'm', 'y', 'k']
    #
    fig = plt.figure(figsize = (16, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Velocity data with fit')
    ax.set_xlabel('Length (kbp)')
    ax.set_ylabel('Velocity (cm/s x $10^4$)')
    nCol = 0
    for plot in range(0, nY):
        ax.errorbar(xMesh[plot, :], zMesh[plot, :], xerr = xErrMesh[plot, :], yerr = zErrMesh[plot, :], 
                    color = colTab[nCol], linestyle = '') 
        fitPlotX = fitFunc(pPlot, xPlot, yMesh[plot, 0]*np.ones(nPlot))
        ax.plot(xPlot, fitPlotX, linestyle = '-', color = colTab[nCol], label = "E = " + str(yMesh[plot, 0]) + " V/cm")
        nCol += 1
        if nCol > nColTab - 1:
            nCol = 0
    ax.set_xlim(xBot, xTop)
    ax.set_ylim(0, 2.2)
    ax.set_xscale('log')
    ax.grid(color = 'g')
    ax.legend()    
    #   
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Velocity data with fit')
    ax.set_xlabel('Field (V/cm)')
    ax.set_ylabel('Velocity (cm/s x $10^4$)')
    nCol = 0
    for plot in range(0, nX):
        ax.errorbar(yMesh[:, plot], zMesh[:, plot], xerr = yErrMesh[:, plot], yerr = zErrMesh[:, plot], 
                    color = colTab[nCol], linestyle = '') 
        fitPlotY = fitFunc(pPlot, xMesh[0, plot]*np.ones(nPlot), yPlot)
        ax.plot(yPlot, fitPlotY, linestyle = ':', color = colTab[nCol], label = "Len = " + str(xMesh[0, plot]) + " kbp")
        nCol += 1
        if nCol > nColTab - 1:
            nCol = 0
    ax.set_xlim(yBot, yTop)
    ax.set_ylim(0, 2.2)
    ax.grid(color = 'g')
    ax.legend()    
    #
    plt.savefig("FitPlotLengthAndField.png")
    plt.show()

#
# CALLADINE DATA
# Length and velocity data extracted from the plot using WebPlotDigitizer-WPD
ECA = 1.0 # V/cm
nLenCA = 30
lenArrCA = np.zeros(nLenCA)
#
lenArrCA[0] = 9.10 # bp
lenArrCA[1] = 75.57
lenArrCA[2] = 125.23
lenArrCA[3] = 154.18
lenArrCA[4] = 223.50
lenArrCA[5] = 300.8
lenArrCA[6] = 370.3
lenArrCA[7] = 480.18
lenArrCA[8] = 536.75
lenArrCA[9] = 582.42
lenArrCA[10] = 636.68
lenArrCA[11] = 986.51
lenArrCA[12] = 1170.14
lenArrCA[13] = 1419.20
lenArrCA[14] = 1646.16
lenArrCA[15] = 1923.69
lenArrCA[16] = 2385.39
lenArrCA[17] = 2829.18
lenArrCA[18] = 3613.40
lenArrCA[19] = 4414.20
lenArrCA[20] = 4825.03
lenArrCA[21] = 6071.25
lenArrCA[22] = 6685.40
lenArrCA[23] = 7253.23
lenArrCA[24] = 8474.72
lenArrCA[25] = 9469.38
lenArrCA[26] = 14882.10 
lenArrCA[27] = 23062.21
lenArrCA[28] = 32215.32
lenArrCA[29] = 47402.21
#
lenArrCA = lenArrCA/1000 # Convert lengths to kbp
#
# velocities extracted from graph using WPD
velArrCA = np.zeros(nLenCA) 
velArrCA[0] = 8.20 # mm/h
velArrCA[1] = 7.80
velArrCA[2] = 7.74
velArrCA[3] = 7.63
velArrCA[4] = 7.35
velArrCA[5] = 7.03
velArrCA[6] = 6.67
velArrCA[7] = 6.24
velArrCA[8] = 6.06
velArrCA[9] = 5.92
velArrCA[10] = 5.75
velArrCA[11] = 4.84
velArrCA[12] = 4.53
velArrCA[13] = 4.17
velArrCA[14] = 3.76
velArrCA[15] = 3.41
velArrCA[16] = 2.94
velArrCA[17] = 2.63
velArrCA[18] = 2.07
velArrCA[19] = 1.7
velArrCA[20] = 1.58
velArrCA[21] = 1.21
velArrCA[22] = 1.09
velArrCA[23] = 0.99
velArrCA[24] = 0.82
velArrCA[25] = 0.65
velArrCA[26] = 0.39 
velArrCA[27] = 0.35
velArrCA[28] = 0.35
velArrCA[29] = 0.35
#
# convert velocities to cm/s * 10**4
velArrCA = velArrCA/3.6
#
# Lengths extracted from plot
ECB = 2.5 # V/cm
nLenCB = 35
#nLenCB = 39
lenArrCB = np.zeros(nLenCB)
#
lenArrCB[0] = 9.11 # bp
lenArrCB[1] = 75.01
lenArrCB[2] = 124.92
lenArrCB[3] = 151.83
lenArrCB[4] = 224.25
lenArrCB[5] = 241.70
lenArrCB[6] = 298.16
lenArrCB[7] = 311.86
lenArrCB[8] = 343.79
lenArrCB[9] = 370.53
lenArrCB[10] = 393.44
lenArrCB[11] = 414.63
lenArrCB[12] = 478.09
lenArrCB[13] = 523.07
lenArrCB[14] = 572.29
lenArrCB[15] = 621.43
lenArrCB[16] = 716.44
lenArrCB[17] = 988.72
lenArrCB[18] = 1148.37
lenArrCB[19] = 1294.51
lenArrCB[20] = 1492.29
lenArrCB[21] = 1608.21
lenArrCB[22] = 1799.11
lenArrCB[23] = 1910.20
lenArrCB[24] = 2027.99
lenArrCB[25] = 2337.76
lenArrCB[26] = 2797.02
lenArrCB[27] = 3552.59
lenArrCB[28] = 4283.01
lenArrCB[29] = 4900.13
lenArrCB[30] = 5996.02
lenArrCB[31] = 6561.21
lenArrCB[32] = 7283.81
lenArrCB[33] = 8523.66
lenArrCB[34] = 9186.09
#
#lenArrCB[35] = 14624.70
#lenArrCB[36] = 23097.79
#lenArrCB[37] = 32617.70
#lenArrCB[38] = 48547.13
#
lenArrCB = lenArrCB/1000 # convert lengths to kbp
#
# Velocities extracted from plot using WPD
velArrCB = np.zeros(nLenCB) 
#
velArrCB[0] = 24.92
velArrCB[1] = 22.63
velArrCB[2] = 21.80
velArrCB[3] = 21.97
velArrCB[4] = 21.17
velArrCB[5] = 20.69
velArrCB[6] = 20.08
velArrCB[7] = 19.63
velArrCB[8] = 19.34
velArrCB[9] = 18.91
velArrCB[10] = 18.76
velArrCB[11] = 18.48
velArrCB[12] = 17.94
velArrCB[13] = 17.53
velArrCB[14] = 17.14
velArrCB[15] = 16.63
velArrCB[16] = 15.66
velArrCB[17] = 14.09
velArrCB[18] = 13.06
velArrCB[19] = 12.39
velArrCB[20] = 11.49
velArrCB[21] = 10.98
velArrCB[22] = 10.18
velArrCB[23] = 9.96
velArrCB[24] = 9.59
velArrCB[25] = 8.82
velArrCB[26] = 7.64
velArrCB[27] = 6.82
velArrCB[28] = 5.56
velArrCB[29] = 5.08
velArrCB[30] = 4.37
velArrCB[31] = 4.40
velArrCB[32] = 3.93
velArrCB[33] = 3.65
velArrCB[34] = 3.51
#
#velArrCB[35] = 3.88
#velArrCB[36] = 2.95
#velArrCB[37] = 2.93
#velArrCB[38] = 2.93
#
# Convert to cm/2 * 10**4
velArrCB = velArrCB/3.6
#
# The code below prints out the data we have entered
print(" ")
print("Data points, E field {:.2f} V/cm".format(ECA))
print("Point\t Length (kbp)\t Velocity (cm/s*10^4)")
for n in range(0, nLenCA):
    print("{:d}\t{:5.2f}\t\t{:5.2f}".format(n, lenArrCA[n], velArrCA[n]))
print(" ")
print("Data points, E field {:.2f} V/cm".format(ECB))
print("Point\t Length (kbp)\t Velocity (cm/s*10^4)")
for n in range(0, nLenCB):
    print("{:d}\t{:5.2f}\t\t{:5.2f}".format(n, lenArrCB[n], velArrCB[n]))
#
plt.figure(figsize = (12, 7))
plt.title("Velocity of DNA")
plt.xlabel("Length (kbp)")
plt.ylabel("Velocity (cm/s x $10^4$) ")
plt.plot(lenArr, velArrA, linestyle = '--', marker = '+', color = 'r', label = "E = " + str(EA) + " V/cm")
plt.plot(lenArr, velArrB, linestyle = '--', marker = 'o', color = 'c', label = "E = " + str(EB) + " V/cm")
plt.plot(lenArr, velArrC, linestyle = '--', marker = '.', color = 'b', label = "E = " + str(EC) + " V/cm")
plt.plot(lenArr, velArrD, linestyle = '--', marker = '+', color = 'k', label = "E = " + str(ED) + " V/cm")
plt.plot(lenArr, velArrE, linestyle = '--', marker = 'o', color = 'orange', label = "E = " + str(EE) + " V/cm")
plt.plot(lenArrCA, velArrCA, linestyle = '-', marker = '+', color = 'r', label = "E = " + str(ECA) + " V/cm")
plt.plot(lenArrCB, velArrCB, linestyle = '-', marker = 'o', color = 'c', label = "E = " + str(ECB) + " V/cm")
plt.grid(color = 'g')
plt.xlim(0.1, 100)
plt.ylim(-0.1, 8.0)
plt.xscale('log')
plt.legend()
plt.show()


import sys
#
# Enter the x values, the field strength and the y data values.
#E1 = ECA
E1 = ECB
Etest = 0.01
#
if abs(E1 - ECA) < Etest:
    nLen = nLenCA
    xData1 = lenArrCA[0:nLen]
    yData1 = velArrCA[0:nLen]
    fXerr1 = 0.01
    fYerr1 = 0.01
    #xError1 = fXerr1*np.ones(nLen)
    #yError1 = fYerr1*np.ones(nLen)
    xError1 = fXerr1*xData1
    yError1 = fYerr1*yData1
elif abs(E1 - ECB) < Etest:
    nLen = nLenCB
    xData1 = lenArrCB[0:nLen]
    yData1 = velArrCB[0:nLen]
    fXerr1 = 0.01
    fYerr1 = 0.01
    #xError1 = fXerr1*np.ones(nLen)
    #yError1 = fYerr1*np.ones(nLen)
    xError1 = fXerr1*xData1
    yError1 = fYerr1*yData1
else:
    print("Chosen field strength doesn't match any available data - stop")
    sys.exit()
#
# The code below prints out the data we have entered
print(" ")
print("Length and velocity data for electric field {:.2f} V/cm".format(E1))
print("Point\tLength (kbp)\t\t\tVelocity (cm/s * 10^4)\t")
for n in range(0, nLen):
    print("{:d}\t{:5.2e} +- {:5.2e}\t\t{:5.2e} +- {:5.2e}".\
          format(n, xData1[n], xError1[n], yData1[n], yError1[n]))
    
    
    
from scipy.optimize import least_squares
#
# kB = p[0]
# kC = p[1]
# kL = p[2]
# alpha = p[3]
#
def fitFunc(p, E, L):
    '''
    Mobility function
    '''
    f = p[0]*E/(1 + p[1]*L) + p[2]*E**p[3] 
    #
    return f
#
def fitFuncDiff(p, E, L):
    '''
    Differential of mobility function
    '''
    df = -p[0]*p[1]*E/(1 + p[1]*L)**2
    #
    return df
#
def fitError(p, x, E, y, xerr, yerr):
    '''
    Error function for straight line fit
    '''
    e = (y - fitFunc(p, E, x))/(np.sqrt(yerr**2 + fitFuncDiff(p, E, x)**2*xerr**2))
    return e
#
# Set initial values of fit parameters, run fit
nPar = 4
pInit = [1.0,     1.0,     1.0,     1.0]
loBnd = [-np.inf, -np.inf, -np.inf, -np.inf]
upBnd = [np.inf,  np.inf,  np.inf,  np.inf]
out = least_squares(fitError, pInit, args=(xData1, E1, yData1, xError1, yError1), bounds = (loBnd, upBnd))
#
fitOK = out.success
#
# Test if fit failed
if not fitOK:
    print(" ")
    print("Fit failed")
else:
    #
    # get output
    pFinal = out.x
    kBval = pFinal[0]
    kCval = pFinal[1]
    kLval = pFinal[2]
    alpha = pFinal[3]
    #
    #   Calculate chis**2 per point, summed chi**2 and chi**2/NDF
    chiarr = fitError(pFinal, xData1, E1, yData1, xError1, yError1)**2
    chisq = np.sum(chiarr)
    NDF = nLen - nPar
    redchisq = chisq/NDF
#
    np.set_printoptions(precision = 3)
    print(" ")
    print("Fit quality:")
    print("chisq per point = \n",chiarr)
    print("chisq = {:5.2f}, chisq/NDF = {:5.2f}.".format(chisq, redchisq))
    #
    # Compute covariance
    jMat = out.jac
    jMat2 = np.dot(jMat.T, jMat)
    detJmat2 = np.linalg.det(jMat2)
    #
    if detJmat2 < 1E-32:
        print("Value of determinat detJmat2",detJmat2)
        print("Matrix singular, error calculation failed.")
        print(" ")
        print("Parameters returned by fit at E field {:.2f}:".format(E1))
        print("kB = {:5.2f}".format(kBval))
        print("kC = {:5.2f}".format(kCval))
        print("kL = {:5.2f}".format(kLval))
        print("alpha = {:5.2f}".format(alpha))
        print(" ")
        kBerr = 0.0
        kCerr = 0.0
        kLerr = 0.0
        alperr = 0.0
    else:
        covar = np.linalg.inv(jMat2)
        kBerr = np.sqrt(covar[0, 0])
        kCerr = np.sqrt(covar[1, 1])
        kLerr = np.sqrt(covar[2, 2])
        alperr = np.sqrt(covar[3, 3])
        #
        print(" ")
        print("Parameters returned by fit at E field {:.2f}:".format(E1))
        print("kB = {:5.2f} +- {:5.2f}".format(kBval, kBerr))
        print("kC = {:5.2f} +- {:5.2f}".format(kCval, kCerr))
        print("kL = {:5.2f} +- {:5.2f}".format(kLval, kLerr))
        print("alpha = {:5.2f} +- {:5.2f}".format(alpha, alperr))
        print(" ")
    #
    # Calculate fitted function values
    #
    nPlot = 100
    xBot = 0.1
    xTop = 100.0
    xPlot = np.exp(np.linspace(np.log(xBot), np.log(xTop), nPlot))
    pPlot = pFinal
    fitPlot1 = fitFunc(pPlot, E1, xPlot)
    #
    # Plot data
    fig = plt.figure(figsize = (8, 6))
    plt.title('Data with fit for field ' + str(E1) + ' V/cm')
    plt.xlabel('Length (kbp)')
    plt.ylabel('Speed (cm/s x $10^4$)')
    plt.errorbar(xData1, yData1, xerr = xError1, yerr = yError1, color = 'r', marker = 'o', \
                 linestyle = '', label = "Data") 
    plt.plot(xPlot, fitPlot1, color = 'b', linestyle = '-', label = "Fit")
    plt.xlim(0.1, 100)
    plt.ylim(-0.1, 8)
    plt.xscale('log')
    plt.grid(color = 'g')
    plt.legend()
#    plt.savefig(FitPlot.png")
    plt.show()    
    
  
  
  
  
import datetime
now = datetime.datetime.now()
print("Date and time ",str(now))
#
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
%matplotlib inline
#
def poisson_func(k, lam):
    '''
    Poisson distribution function.
    Given number of occurences and mean, returns distribution function value.
    '''
    poiss = lam**k*np.exp(-lam)/sp.special.factorial(k)
    return poiss
#
# Number of (single) DNA strands
n_strands = 92
#
# Length of DNA strand
L_DNA = 1e8 # base pairs
#
# Average number of breaks per strand, the number of segments generated and their lengths
mean_breaks = 20 
n_segs = np.zeros(n_strands).astype(int)
seg_lengths = np.zeros((n_strands, (mean_breaks + 5*np.sqrt(mean_breaks)).astype(int)))
#
# Set up histgram of number of breaks
num_range = np.zeros(2)
num_range[0] = np.maximum(mean_breaks - 5*np.sqrt(mean_breaks), 0.0).astype(int)
num_range[1] = (mean_breaks + 5*np.sqrt(mean_breaks)).astype(int)
n_num_bins = ((num_range[1] - num_range[0])).astype(int)
num_bins, num_bin_wid = np.linspace(num_range[0], num_range[1], n_num_bins + 1, retstep = True)
num_bin_cents = 0.5*(num_bins[0:n_num_bins] + num_bins[1:n_num_bins + 1])
num_bin_err = num_bin_wid/np.sqrt(12)*np.ones(n_num_bins)
cum_num_hist = np.zeros(n_num_bins)
#
# Set up histogram of segment lengths
n_len_bins = 40
len_bins, len_bin_wid = np.linspace(0.0, 10*L_DNA/mean_breaks, n_len_bins + 1, retstep = True)
len_bin_cents = 0.5*(len_bins[0:n_len_bins] + len_bins[1:n_len_bins + 1])
len_bin_err = len_bin_wid/np.sqrt(12)*np.ones(n_len_bins)
cum_len_hist = np.zeros(n_len_bins)
#
# Initialise random number generator
rng = np.random.default_rng()
#
# Simulate breaks and segements for all strands
for n in range(0, n_strands):
    n_breaks = rng.poisson(mean_breaks)
    breaks = np.zeros(n_breaks)
    n_segs[n] = n_breaks + 1
    breaks = rng.uniform(0.0, 1.0, n_breaks)
    ends = np.zeros(n_segs[n] + 1)
    ends[1:n_segs[n]] = np.sort(breaks)
    ends[n_segs[n]] = 1.0
    seg_lengths[n, 0:n_segs[n]] = L_DNA*(ends[1:n_segs[n] + 1] - ends[0:n_segs[n]])
    #
    num_hist_here, _ = np.histogram(n_breaks, num_bins)
    cum_num_hist += num_hist_here
    #
    len_hist_here, _ = np.histogram(seg_lengths[n, 0:n_segs[n]], len_bins)
    cum_len_hist += len_hist_here
#
# Function describing distribution of number of breaks
n_poiss_plot = 100
k_poiss_plot = np.linspace(num_range[0], num_range[1], n_poiss_plot)
poiss_func_plot = n_strands*num_bin_wid*poisson_func(mean_breaks, k_poiss_plot)
#
# Function describing segment length distribution
len_func_plot = (mean_breaks*(mean_breaks + 1)*n_strands*
                 len_bin_wid/L_DNA*np.exp(-mean_breaks*len_bin_cents/L_DNA))
#
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
#
ax[0].set_title('Broken strand numbers')
ax[0].errorbar(num_bin_cents, cum_num_hist, xerr = num_bin_err, yerr = np.sqrt(cum_num_hist),
               linestyle = '', marker = '+', color = 'b')
ax[0].plot(k_poiss_plot, poiss_func_plot, linestyle = '-', marker = '', color = 'r')
ax[0].set_xlabel('Number')
ax[0].set_ylabel('Relative frequency')
ax[0].set_ylim(0.0, 1.1*(np.amax(cum_num_hist) + np.amax(np.sqrt(cum_num_hist))))
ax[0].grid(color = 'g')
#
ax[1].set_title('Broken strand lengths')
ax[1].errorbar(len_bin_cents, cum_len_hist, xerr = len_bin_err, yerr = np.sqrt(cum_len_hist),
               linestyle = '', marker = '+', color = 'b')
ax[1].plot(len_bin_cents, len_func_plot, linestyle = '-', marker = '', color = 'r')
ax[1].set_xlabel('Length (bp)')
ax[1].set_ylabel('Relative frequency')
ax[1].set_xlim(0.0, L_DNA)
ax[1].set_ylim(0.5, 1.1*(np.amax(cum_len_hist) + np.amax(np.sqrt(cum_len_hist))))
ax[1].grid(color = 'g')
#
ax[2].set_title('Broken strand lengths')
ax[2].errorbar(len_bin_cents, cum_len_hist, xerr = len_bin_err, yerr = np.sqrt(cum_len_hist),
               linestyle = '', marker = '+', color = 'b')
ax[2].plot(len_bin_cents, len_func_plot, linestyle = '-', marker = '', color = 'r')
ax[2].set_xlabel('Length (bp)')
ax[2].set_ylabel('Relative frequency')
ax[2].set_xlim(0.0, L_DNA)
ax[2].set_ylim(0.5, 1.5*(np.amax(cum_len_hist) + np.amax(np.sqrt(cum_len_hist))))
ax[2].set_yscale('log')
ax[2].grid(color = 'g')
#
plt.tight_layout()
plt.show
#
then = now
now = datetime.datetime.now()
print(" ")
print("Date and time",str(now))
print("Time since last check is",str(now - then))

from mpl_toolkits.mplot3d import Axes3D
#
use_sphere = True
#
# Radius of nucleus
rad_nuc = 2.0 # microns
rad_nuc_min = 1.5 # microns
sig_rad_nuc = 0.5 # microns
#
# Radius of cell
rad_cell = 10.0 # microns
rad_cell_min = 16.0 # microns
sig_rad_cell = 1.0 # microns
#
# Colot table for plots
n_color_tab = 8
color_tab = np.array(['r', 'orange', 'y', 'g', 'c', 'b', 'm', 'k'])
#
fig = plt.figure(figsize = (10, 8))
#
ax2d_cell = fig.add_subplot(2, 2, 1)
ax2d_cell.set_title("Initial positions of strands - cell")
ax2d_cell.set_xlabel("x")
ax2d_cell.set_ylabel("y")
#
ax3d_cell = fig.add_subplot(2, 2, 2, projection='3d')
ax3d_cell.set_title("Initial positions of strands - cell")
ax3d_cell.set_xlabel("x")
ax3d_cell.set_ylabel("y")
ax3d_cell.set_zlabel("z")
#
ax2d_pic = fig.add_subplot(2, 2, 3)
ax2d_pic.set_title("Initial positions of strands - image")
ax2d_pic.set_xlabel("x")
ax2d_pic.set_ylabel("y")
#
ax3d_pic = fig.add_subplot(2, 2, 4, projection='3d')
ax3d_pic.set_title("Initial positions of strands - image")
ax3d_pic.set_xlabel("x")
ax3d_pic.set_ylabel("y")
ax3d_pic.set_zlabel("z")
#
# Image dimensions
n_rows = 1040 # number of rows of pixels (y coord)
n_cols = 1392 # number of columns of pixels (x coord)
depth = 20
#
# Set up number of cells and their positions
n_cells = 15
x_cell = np.zeros(n_cells)
y_cell = np.zeros(n_cells)
z_cell = np.zeros(n_cells)
#
# Set location of cells
if n_cells == 1:
    x_cell[0] = n_cols/2
    y_cell[0] = n_rows/2
    z_cell[0] = depth/2
else:
    x_cell = np.random.uniform(0, n_cols, n_cells)
    y_cell = np.random.uniform(0, n_rows, n_cells)
    z_cell = np.random.uniform(0, depth, n_cells)

#
# Colors for plotting strands
colors = np.zeros(n_strands).astype(str)
#
for n_cell in range(0, n_cells):
    #
    # Simulate uniform radial and phi initial distribution of strands
    rad_arr = np.sqrt(np.random.uniform(0, max(np.random.normal(rad_cell, sig_rad_cell, 1),
                                              rad_cell_min)**2, n_strands))
    phi_arr = np.random.uniform(0, 2*np.pi, n_strands)
    #
    # Determine initial coordinates of strands (simulate uniform intial theta distribution if required)
    if use_sphere:
        theta_arr = np.arccos(np.random.uniform(-1, 1, n_strands))
        x_arr = x_cell[n_cell] + rad_arr*np.sin(theta_arr)*np.cos(phi_arr)
        y_arr = y_cell[n_cell] + rad_arr*np.sin(theta_arr)*np.sin(phi_arr)
        z_arr = z_cell[n_cell] + rad_arr*np.cos(theta_arr)
    else:
        x_arr = x_cell[n_cell] + rad_arr*np.cos(phi_arr)
        y_arr = y_cell[n_cell] + rad_arr*np.sin(phi_arr)
        z_arr = x_cell[n_cell] + np.random.uniform(0, depth, n_strands)
    #
    # Plot intial positions of all strands in image
    col_inds = (np.linspace(0, n_strands - 1, n_strands)%n_color_tab).astype(int)
    ax2d_pic.scatter(x_arr, y_arr, color = color_tab[col_inds], s = 1.0)
    ax3d_pic.scatter(x_arr, y_arr, z_arr, color = color_tab[col_inds], s = 1.0)
    #
    if n_cell > 0:
        continue
    #
    # Plot initial positions of strands in one cell 
    ax2d_cell.scatter(x_arr - x_cell[n_cell], y_arr - y_cell[n_cell], color = color_tab[col_inds], s = 10.0)
    ax3d_cell.scatter(x_arr - x_cell[n_cell], y_arr - y_cell[n_cell], 
                      z_arr - z_cell[n_cell], color = color_tab[col_inds], s = 10.0)    
#
scale_cell = 2.0
ax2d_cell.set_xlim(-scale_cell*rad_cell, scale_cell*rad_cell)
ax2d_cell.set_ylim(-scale_cell*rad_cell, scale_cell*rad_cell)
#
ax3d_cell.set_xlim(-scale_cell*rad_cell, scale_cell*rad_cell)
ax3d_cell.set_ylim(-scale_cell*rad_cell, scale_cell*rad_cell)
ax3d_cell.set_zlim(-scale_cell*rad_cell, scale_cell*rad_cell)
#
expand_pic = 0.2
ax2d_pic.set_xlim(-expand_pic*n_cols, (1 + expand_pic)*n_cols)
ax2d_pic.set_ylim(-expand_pic*n_cols, (1 + expand_pic)*n_rows)
#
ax3d_pic.set_xlim(-expand_pic*n_cols, (1 + expand_pic)*n_cols)
ax3d_pic.set_ylim(-expand_pic*n_cols, (1 + expand_pic)*n_rows)
ax3d_pic.set_zlim(-(1 + expand_pic)*rad_cell, depth + (1 + expand_pic)*rad_cell)
#
plt.tight_layout()
plt.show()

import datetime
now = datetime.datetime.now()
print("Date and time ",str(now))
#
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#
def vDagarose(L, E):
    '''
    Given length of DNA strand (in kbp) and electric field (in V/cm) returns 
    velocity in agarose (in cm/s * 10^4 or equivalently microns/s).
    '''
    if not hasattr(vDagarose, "kB"):
        vDagarose.kB, vDagarose.kC, vDagarose.kL, vDagarose.alpha = 2.24, 0.56, 0.59, 1.34
        print(" ")
        print("Parameters used to descibe vD in agarose")
        print("kB = {:.2f}".format(vDagarose.kB))
        print("kC = {:.2f}".format(vDagarose.kC))
        print("kL = {:.2f}".format(vDagarose.kL))
        print("alpha = {:.2f}".format(vDagarose.alpha))
    #
    vD = vDagarose.kB*E/(1 + vDagarose.kC*L) + vDagarose.kL*E**vDagarose.alpha
    #
    return vD
#
# Calculate fitted function values x direction
nPlot = 100
xBot = 0.1
xTop = 1000.0
xPlot = np.exp(np.linspace(np.log(xBot), np.log(xTop), nPlot))
nY = 5
yVals = np.array([0.05, 0.15, 0.3, 0.5, 0.8]) # V/cm
#
# Calculate fitted function values y direction
yBot = 0.0
yTop = 1.0
yPlot = np.linspace(yBot, yTop, nPlot)
nX = 11
xVals = np.array([0.31, 0.59, 0.86, 1.07, 1.28, 1.97, 2.24, 4.28, 6.59, 9.31, 23.01]) # kbp
#
# Plot data
nColTab = 6
colTab = ['b', 'r', 'c', 'm', 'y', 'k']
#
fig = plt.figure(figsize = (16, 6))
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Velocity model')
ax.set_xlabel('Length (kbp)')
ax.set_ylabel('Velocity (cm/s $\\times$ $10^4$)')
nCol = 0
for plot in range(nY, 0, -1):
    fitPlotX = vDagarose(xPlot, yVals[plot - 1]*np.ones(nPlot))
    ax.plot(xPlot, fitPlotX, linestyle = '-', color = colTab[nCol], label = "E = " + str(yVals[plot - 1]) + " V/cm")
    nCol += 1
    if nCol > nColTab - 1:
        nCol = 0
ax.set_xlim(xBot, xTop)
ax.set_ylim(0, 2.2)
ax.set_xscale('log')
ax.grid(color = 'g')
ax.legend()    
#   
ax = fig.add_subplot(1, 2, 2)
ax.set_title('Velocity model')
ax.set_xlabel('Field (V/cm)')
ax.set_ylabel('Velocity (cm/s $\\times$ $10^4$)')
nCol = 0
for plot in range(0, nX):
    fitPlotY = vDagarose(xVals[plot]*np.ones(nPlot), yPlot)
    ax.plot(yPlot, fitPlotY, linestyle = '-', color = colTab[nCol], label = "Len = " + str(xVals[plot]) + " kbp")
    nCol += 1
    if nCol > nColTab - 1:
        nCol = 0
ax.set_xlim(yBot, yTop)
ax.set_ylim(0, 2.2)
ax.grid(color = 'g')
ax.legend()    
#
# plt.savefig("FitPlot.png")
plt.show()
#
then = now
now = datetime.datetime.now()
print(" ")
print("Date and time",str(now))
print("Time since last check is",str(now - then))



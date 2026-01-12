###################
#EEE419 Project 4
#Caitlyn Blythe
###################

import numpy as np
import subprocess
import shutil
import string
#initialize variables to store the optimal values tphl, # of inverters, and fan
tphl_opt = 1
inv_opt = 0
fan_opt = 0
for fan in range (1,7):    #loop through fan values
    for inv in range (2,11,2):    #loop through inverter values, starting at 2 and adding 2 at a time
        shutil.copy('InvChain_base.sp', 'InvChain.sp')    #copy base file to new file
        edit_invchain = open('InvChain.sp', 'a')    #open InvChain.sp in append mode
        edit_invchain.write('\n.param fan = ' + str(fan))    #write param line for fan value
        edit_invchain.write('\nXinv1 a b inv M=1')    #hard coding the first inverter
        for i in range(1,inv+1):    #loop through the num inverters we are on in the loop to write the hspice file
            if i < inv:
                edit_invchain.write('\nXinv'+str(i+1)+' '+string.ascii_lowercase[i]+' '+string.ascii_lowercase[i+1]+' inv M=fan**'+str(i))
            elif i == inv:    #last node needs to end on z
                edit_invchain.write('\nXinv'+str(i+1)+' '+string.ascii_lowercase[i]+' z inv M=fan**'+str(i))
        edit_invchain.write('\n.end')    #writes .end in hspice file to finish it out
        edit_invchain.close()
        #opening hspice and running the file that was written
        proc = subprocess.Popen(["hspice","InvChain.sp"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output, err = proc.communicate()
        #get tphl from the csv output file
        data = np.genfromtxt('InvChain.mt0.csv', delimiter=',', comments='$', skip_header=3, dtype=None, names=True,encoding=None)
        tphl = data["tphl_inv"]
        print('N ',inv,' fan ',fan,' tphl ',tphl)
        #find best values of tphl, fan, and inv
        if tphl < tphl_opt:
            tphl_opt = tphl
            inv_opt = inv
            fan_opt = fan
print('Best values were:')
print('fan = ',fan_opt)
print('num inverters = ',inv_opt)
print('tphl = ',tphl_opt)

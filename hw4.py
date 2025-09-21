#####################
#EEE419 HW4         #
#Caitlyn Blythe     #
#####################

#importing tkinter and matplotlib
import numpy as np
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#defining inputs/constants
INPUTS = ['Mean Return (%)','Std. Dev. Return (%)','Yearly Contribution ($)','No. Years of Contribution',
          'No. Years to Retirement','Annual Retirement Spend']
INP_MNRTRN = 0    #mean return index
INP_STDDEV = 1    #std dev return index
INP_YEARLY = 2    #yearly contribution index
INP_YRCONT = 3    #years of contribution index
INP_RETIRE = 4    #years to retirement index
INP_ANNUAL = 5    #annual retirement spend index
NUM_INPUTS = 6    #number of inputs
MAX_YEARS = 70    #constant - maximum years

#####################
#function to calculate wealth
def calc_wealth(entries):    #function to calculate wealth given user inputs from entries
    mnrtrn = float(entries[INP_MNRTRN].get())    #assigning the mean return user input to variable mnrtrn
    stddev = float(entries[INP_STDDEV].get())    #assigning the standard deviation user input to variable stddev
    yearly = float(entries[INP_YEARLY].get())    #assigning the yearly contribution input to variable yearly
    yrcont = int(entries[INP_YRCONT].get())      #assigning the # of years of contribution input to variable yrcont
    retire = int(entries[INP_RETIRE].get())      #assigning the # of years in retirement input to variable retire
    annual = float(entries[INP_ANNUAL].get())    #assigning the annual spend in retirement input to variable annual
    noise = (stddev/100)*np.random.randn(MAX_YEARS)    #array of 70 random values, calculates volatility using stddev
    wealth = [yearly]    #assigns the value of yearly to variable wealth as a matrix, so we can append values later on

    for i in range(0,10):    #run analysis 10 times
        for year in range(0,MAX_YEARS):
            if wealth[year] <= 0:    #if value of wealth in current year goes to or below zero:
                wealth.append(0)     #value appended will be zero
            else:
                if year < yrcont:    #if current year is less than # of years contributed
                    wealth.append(wealth[year]*(1+(mnrtrn/100)+noise[year])+yearly)
                elif year < retire:  #else if current year is less than the # of years in retirement
                    wealth.append(wealth[year]*(1+(mnrtrn/100)+noise[year]))
                else:    #all other cases
                    wealth.append(wealth[year]*(1+(mnrtrn/100)+noise[year])-annual)
        i += 1
    avg_wealth = sum(wealth)/10    #calculating average wealth over the 10 analyses done
    return avg_wealth

#####################
#make the gui
def make_gui(root):
    entries = []
    for index in range(NUM_INPUTS):
        row = Frame(root)
        lab = Label(row, width=22, text=INPUTS[index]+": ", anchor='w')

        ent = Entry(row)
        ent.insert(0,"0")
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append(ent)
    return entries

root = Tk()
ents = make_gui(root)
avgwealth = calc_wealth(ents)
lab2 = Label(root, width=22, text="Wealth at retirement: %f" %float(avgwealth))
lab2.pack(side=TOP, anchor=W)
b1 = Button(root, text="Quit", command=root.destroy)
b1.pack(side=LEFT, padx=5, pady=5)
b2 = Button(root, text="Calculate", command=(lambda: calc_wealth(ents)))
b2.pack(side=RIGHT, padx=5, pady=5)

root.mainloop()


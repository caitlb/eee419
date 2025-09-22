#####################
#EEE419 HW4
#Caitlyn Blythe
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

    wealth = [yearly]    #initialize wealth as a matrix, with first entry being the first yearly contribution
    for year in range(1,MAX_YEARS):
        if year < yrcont:    #from start until end of contributions
            wealth.append(wealth[year-1]*(1+(mnrtrn/100)+noise[year-1])+yearly)
        elif year < retire + yrcont:    #end of contributions until retiring
            wealth.append(wealth[year-1]*(1+(mnrtrn/100)+noise[year-1]))
        else:    #from retirement until the end of 70 years
            wealth.append(wealth[year-1]*(1+(mnrtrn/100)+noise[year-1])-annual)
            if wealth[year] < 0:    #won't show negative balance
                wealth[year] = 0
    return wealth, wealth[retire-1]    #returns wealth as a matrix, as well as specific value for wealth at retirement

#####################
#simulating 10 times to find average
def calc_avgs(entries):
    wealthx10 = []    #initializing a matrix to hold 10 simulations worth of calculations
    tot_retwealth = 0    #initializing variable to hold summed wealth values at retirement
    for i in range(10):    #run 10 times
        wealthvals, ret_wealth = calc_wealth(entries)    #stores returned wealth/wealth at retirement
        wealthx10.append(wealthvals)    #append the wealth values from all 10 sims into wealthx10
        tot_retwealth += ret_wealth    #storing summed wealth values at retirement
        wealthvals = np.trim_zeros(wealthvals,'b')    #gets rid of any 0 values in matrix so they are not plotted
        plt.plot(range(MAX_YEARS),wealthvals)    #plot wealth over 70 years
    #plot stuff here
    plt.title("Wealth over 70 years")
    plt.xlabel("Years")
    plt.ylabel("Wealth")
    plt.show(block=False)

    avg_retwealth = tot_retwealth/10    #averaging wealth at retirement
    lab2.config(text=f"Wealth at retirement: {avg_retwealth:,.2f}")    #label updates when avg_retwealth changes

#####################
#make the gui
def make_gui(root):
    entries = []    #make a place for entries to go
    for index in range(NUM_INPUTS):    #take entries
        row = Frame(root)
        lab = Label(row, width=22, text=INPUTS[index]+": ", anchor='w')
        ent = Entry(row)
        ent.insert(0,"0")
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append(ent)
    return entries
#gui buttons/labels
root = Tk()
ents = make_gui(root)

lab2 = Label(root, text="")
lab2.pack(side=TOP, anchor=W)
b1 = Button(root, text="Quit", command=root.destroy)
b1.pack(side=LEFT, padx=5, pady=5)
b2 = Button(root, text="Calculate", command=(lambda: calc_avgs(ents)))
b2.pack(side=RIGHT, padx=5, pady=5)

root.mainloop()

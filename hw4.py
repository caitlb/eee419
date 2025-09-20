#####################
#EEE419 HW4         #
#Caitlyn Blythe     #
#####################

#importing tkinter and matplotlib
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#defining inputs
INPUTS = ['Mean Return (%)','Std. Dev. Return (%)','Yearly Contribution ($)','No. Years of Contribution',
          'No. Years to Retirement','Annual Retirement Spend']
INP_MNRTRN = 0    #mean return index
INP_STDDEV = 1    #std dev return index
INP_YEARLY = 2    #yearly contribution index
INP_YRCONT = 3    #years of contribution index
INP_RETIRE = 4    #years to retirement index
INP_ANNUAL = 5    #annual retirement spend index
NUM_INPUTS = 6       #number of inputs


def calc_wealth(entries):    #function to calculate wealth given user inputs from entries
    mnrtrn = float(entries[INP_MNRTRN].get())    #assigning the mean return user input to variable mnrtrn
    stddev = float(entries[INP_STDDEV].get())    #assigning the standard deviation user input to variable stddev
    yearly = float(entries[INP_YEARLY].get())    #assigning the yearly contribution input to variable yearly
    yrcont = int(entries[INP_YRCONT].get())      #assigning the # of years of contribution input to variable yrcont
    retire = int(entries[INP_RETIRE].get())      #assigning the # of years in retirement input to variable retire
    annual = float(entries[INP_ANNUAL].get())    #assigning the annual spend in retirement input to variable annual
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter.messagebox import showinfo, askyesno
import time
padxlim = 320
padylim = 240
image_count=0
w=[]
pattern_in = np.ones((padylim,padxlim))
newCl=""
Cl="dog"


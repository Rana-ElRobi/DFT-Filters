import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt
from PIL import Image as PILImg
import numpy as np
from util import convolve, read_2d_kernel, low_pass_filter, high_pass_filter, band_pass_filter,\
 dft, img_rect_mul, img_stretch, img_mirror, zero_phase, zero_magnitude, cos_multiply,\
 sin_multiply, cos_sin_multiply, rotate_img, rotate_multiply_img, ft_fov_increase, \
 img_hamm_mul, img_hamm_rect_mul
from scipy import signal

try:
    import tkinter as tk
    from tkinter import *
    from filedialog import askopenfilename
except ImportError:
    import Tkinter as tk
    from Tkinter import *
    import ttk
    from tkFileDialog import askopenfilename

LARGE_FONT= ("Verdana", 12)
# style.use("ggplot")

#plt.rcParams['image.cmap'] = 'gray'
#plt.rcParams['image.aspect'] = 'auto'

f = Figure()
a1 = f.add_subplot(221)
a2 = f.add_subplot(222)
a3 = f.add_subplot(223)
a4 = f.add_subplot(224)

fname = ''
img = None
kernel = None

task = 0
optionList = ["Select Task", "Apply Filter", "Multiply Rectangle", "Hamming", "Shrink", "Stretch", "Mirror", "Zero Phase", "Zero Magnitude", "Cos Multiply", "Sin Multiply", "Sin Cos Multiply", "Rotate", "Rotate & Multiply", "FOV"]
taskList = [0, 1, 21, 211, 221, 222, 23, 24, 25, 27, 28, 29, 30, 31, 32]
dropVar = None
def LoadImage():
    global fname, img
    fname = askopenfilename(initialdir=".", filetypes =(("Image",("*.png","*.jpg","*.jpeg","*.bmp","*.gif")),("All Files","*.*")), title = "Choose an image.")
    try:
        img = PILImg.open(fname).convert('L')
        img = np.asarray(img, dtype=np.float64)
        print 'Image:', fname.split('/')[-1], img.shape
    except:
        return
            
def LoadKernel():
    global fname, kernel
    fname = askopenfilename(initialdir=".", filetypes =(("Text",("*.txt","*.csv")),("All Files","*.*")), title = "Choose an kernel.")
    try:
        kernel = read_2d_kernel(fname)
        print 'Kernel:', fname.split('/')[-1], kernel.shape
    except:
        return
        
def Apply():
    global task, optionList, taskList, dropVar
    global conv_img, dft_result, conv_img_in, diff1, diff2, diff3
    global mul_result, fft_img, fft_kernel
    global img_str, fft_img, fft_str
    global rect_kernel, rect_inverse, hamm_kernel,hamm_inverse
    global img_mirr, fft_img, fft_mirr
    global zero_phase_img, img_dft, new_img_mag
    global zero_mag_img, img_dft, new_img_mag
    global mul, img_dft, new_img_dft
    global rotated_img, img_dft, rotated_img_dft
    global v1, v2, v3
    
    task = taskList[optionList.index(dropVar.get())]
    #high_pass_filter(5)
#    band_pass_filter()
    if task == 1:
        print "1 Apply Filter"
        half_filter_size = kernel.shape[0]/2
        conv_img = convolve(img, kernel)
        dft_result = dft(img, kernel)
        conv_img_in = signal.convolve2d(img, kernel, mode = "same", boundary="wrap")
        print conv_img.shape, dft_result.shape, conv_img_in.shape
#        diff1 = abs(conv_img[half_filter_size:img.shape[0]-half_filter_size, half_filter_size:img.shape[1]-half_filter_size] - conv_img_in)
        diff1 = abs(conv_img - conv_img_in)
        diff2 = abs(conv_img[half_filter_size:img.shape[0]-half_filter_size, half_filter_size:img.shape[1]-half_filter_size] - conv_img_in[half_filter_size:img.shape[0]-half_filter_size, half_filter_size:img.shape[1]-half_filter_size])
        diff3 = abs(conv_img - dft_result)
        diff4 = abs(conv_img[half_filter_size:img.shape[0]-half_filter_size, half_filter_size:img.shape[1]-half_filter_size] - dft_result[half_filter_size:img.shape[0]-half_filter_size, half_filter_size:img.shape[1]-half_filter_size])
        
#        diff4 = abs(conv_img[half_filter_size:img.shape[0]-half_filter_size, half_filter_size:img.shape[1]-half_filter_size] - conv_img_in)
        
        print diff1.sum()
        print diff2.sum()
        print diff3.sum()
        print diff4.sum()
        
    if task == 21:
        print "21 rect multiply"
        mul_result, fft_img, fft_kernel = img_rect_mul(img, 20) #60
    if task == 211:
        print "211 hamm rect multiply"
        rect_kernel, rect_inverse, hamm_kernel, hamm_inverse = img_hamm_rect_mul(img, rect_size=20, hamm_size=50)
    if task == 221:
        print "221 shrink"
        img_str, fft_img, fft_str = img_stretch(img, 0.5)
    if task == 222:
        print "222 stretch"
        img_str, fft_img, fft_str = img_stretch(img, 2)
    if task == 23:
        print "23 mirror"
        img_mirr, fft_img, fft_mirr = img_mirror(img)
    if task == 24:
        print "24 zero phase"
        zero_phase_img, img_dft, new_img_mag = zero_phase(img)
    if task == 25:
        # bike gonz2
        print("25 zero magnitude")
        zero_mag_img, img_dft, new_img_mag = zero_magnitude(img)
    if task == 26:
        # bike gonz2
        print("26 cos multiply")
        mul, img_dft, new_img_dft = cos_multiply(img)
    if task == 27:
        # bike gonz2
        print("27 cos multiply")
        mul, img_dft, new_img_dft = cos_multiply(img)
    if task == 28:
        # bike gonz2
        print("28 sin multiply")
        mul, img_dft, new_img_dft = sin_multiply(img)
    if task == 29:
        # bike gonz2
        print("29 cos sin multiply")
        mul, img_dft, new_img_dft = cos_sin_multiply(img)
    if task == 30:
        # bike gonz2
        print("30 rotation")
        rotated_img, img_dft, rotated_img_dft = rotate_img(img, 90)
    if task == 31:
        # slanted
        print("31 rotation multiply")
        mul, img_dft, new_img_dft = rotate_multiply_img(img, 90)
    if task == 32:
        print("32 ft_fov_increase")
        v1, v2, v3 = ft_fov_increase(img)
      
def animate(i):
    global img, kernel, task
    global conv_img, dft_result, conv_img_in, diff1, diff2, diff3
    global mul_result, fft_img, fft_kernel
    global img_str, fft_img, fft_str
    global rect_kernel, rect_inverse, hamm_kernel,hamm_inverse
    global img_mirr, fft_img, fft_mirr
    global zero_phase_img, img_dft, new_img_mag
    global zero_mag_img, img_dft, new_img_mag
    global mul, img_dft, new_img_dft
    global rotated_img, img_dft, rotated_img_dft
    global v1, v2, v3
    if img is not None:
        xList = range(10)
        yList = range(10)
        a1.clear()
        # img = plt.imread('airplane.png')
        a1.imshow(img, cmap='gray')
        if task == 1:
            #a2.clear()
            #a2.bar(range(256), bin_counts)
            a2.clear()
            a2.imshow(conv_img, cmap='gray') #, vmin=0, vmax=255
            a3.clear()
            a3.imshow(conv_img_in, cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(diff1, cmap='gray', vmin=0, vmax=255)

        if task == 21:
            a2.clear()
            a2.imshow(np.log1p(fft_img), cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(fft_kernel, cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(mul_result, cmap='gray', vmin=0, vmax=255)
            plt.imsave('50.jpg', mul_result, cmap='gray')
        if task == 211:
            a1.clear()
            a1.imshow(rect_kernel, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a2.clear()
            a2.imshow(rect_inverse, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(hamm_kernel, cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(hamm_inverse, cmap='gray', vmin=0, vmax=255)
        if task == 221:
            # vertical_lines.png
            a2.clear()
            a2.imshow(np.log1p(fft_img), cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(img_str, cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(np.log1p(fft_str), cmap='gray') # , vmin=0, vmax=255
        if task == 222:
            # vertical_lines.png
            a2.clear()
            a2.imshow(np.log1p(fft_img), cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(img_str, cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(np.log1p(fft_str), cmap='gray') # , vmin=0, vmax=255
        if task == 23:
            # slanted.gif
            a2.clear()
            a2.imshow(np.log1p(fft_img), cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(img_mirr, cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(np.log1p(fft_mirr), cmap='gray') # , vmin=0, vmax=255
        if task == 24:
            # vertical_lines.png - slanted.gif
            a2.clear()
            a2.imshow(zero_phase_img, cmap='gray' , vmin=0, vmax=255) #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(np.log1p(img_dft), cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(np.log1p(new_img_mag), cmap='gray') # , vmin=0, vmax=255
        if task == 25:
            # bike gonz2
            a2.clear()
            a2.imshow(zero_mag_img, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(np.log1p(img_dft), cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(new_img_mag, cmap='gray', vmin=0, vmax=255) # , vmin=0, vmax=255
        if task == 26:
            a2.clear()
            a2.imshow(mul, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(np.log1p(img_dft), cmap='gray') # , vmin=0, vmax=255
            # print( np.log1p(img_dft).sum() )
            a4.clear()
            a4.imshow(np.log1p(new_img_dft), cmap='gray') # , vmin=0, vmax=255
        if task == 27:
            a2.clear()
            a2.imshow(mul, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(img_dft, cmap='gray') # , vmin=0, vmax=255
            # print( np.log1p(img_dft).sum() )
            a4.clear()
            a4.imshow(new_img_dft, cmap='gray') # , vmin=0, vmax=255
        if task == 28:
            a2.clear()
            a2.imshow(mul, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(img_dft, cmap='gray') # , vmin=0, vmax=255
            # print( np.log1p(img_dft).sum() )
            a4.clear()
            a4.imshow(new_img_dft, cmap='gray') # , vmin=0, vmax=255
        if task == 29:
            a2.clear()
            a2.imshow(mul, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(img_dft, cmap='gray') # , vmin=0, vmax=255
            # print( np.log1p(img_dft).sum() )
            a4.clear()
            a4.imshow(new_img_dft, cmap='gray') # , vmin=0, vmax=255
        if task == 30:
            a2.clear()
            a2.imshow(rotated_img, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(np.log1p(img_dft), cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(np.log1p(rotated_img_dft), cmap='gray') # , vmin=0, vmax=255      
        if task == 31:
            a2.clear()
            a2.imshow(mul, cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(np.log1p(img_dft), cmap='gray') # , vmin=0, vmax=255
            a4.clear()
            a4.imshow(np.log1p(new_img_dft), cmap='gray') # , vmin=0, vmax=255
        if task == 32:
            a2.clear()
            a2.imshow(np.log1p(v1), cmap='gray') #, vmin=fft_img., vmax=fft_img.max()
            a3.clear()
            a3.imshow(np.log1p(v2), cmap='gray') # , vmin=0, vmax=255, aspect='auto'
            a4.clear()
            a4.imshow(v3, cmap='gray', interpolation='none') # , vmin=0, vmax=255
        # a.clear()
        # a.imshow(img)
    # pullData = open("f.csv","r").read()
    # dataList = pullData.split('\n')
    # xList = []
    # yList = []
    # for eachLine in dataList:
        # if len(eachLine) > 1:
            # x, y = eachLine.split(',')
            # xList.append(int(x))
            # yList.append(int(y))

    
            

class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Assignment 2")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage,):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()
          
class StartPage(tk.Frame):
    def func(self,value):
            print value
            
    def __init__(self, parent, controller):
        global optionList, taskList, dropVar
        tk.Frame.__init__(self, parent)
        # label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        # label.pack(pady=10,padx=10)
        
        ctrl_frame = Frame(self)
        button1 = ttk.Button(ctrl_frame, text="Load Image", command=LoadImage)
        button1.pack(side=LEFT)
        button2 = ttk.Button(ctrl_frame, text="Load Kernel", command=LoadKernel)
        button2.pack(side=LEFT)
        
        dropVar=StringVar()
        dropVar.set(optionList[0]) # default choice
        dropMenu1 = OptionMenu(ctrl_frame, dropVar, *optionList, command=self.func)
        dropMenu1.pack(side=LEFT)
        
        button3 = ttk.Button(ctrl_frame, text="Apply", command=Apply)
        button3.pack(side=LEFT)
        
        ctrl_frame.pack(side=TOP)
        
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        

app = SeaofBTCapp()
ani = animation.FuncAnimation(f, animate, interval=1000)
app.mainloop()
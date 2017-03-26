import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.misc import imrotate

def read_2d_kernel(fname):
    with open(fname, "r") as f:
        line = f.readline()
        delim = " "
        try:
            dim = [int(s) for s in line.split(delim)]
        except:
            delim = ","
            dim = [int(s) for s in line.split(delim)]
        kernel = np.zeros((dim[0], dim[1]))
        i = 0
        txt = f.read()
        for line in txt.split('\n'):
            if line == '':
                continue
#            print line
#            print line.split(delim)
#            print [float(s) for s in line.split(delim)]
#            print '#'
            kernel[i] = np.array([float(s) for s in line.split(delim)])
            i += 1
        print kernel
        if abs(kernel).sum() != 0:
            #kernel /= abs(kernel).sum()
            kernel = kernel
        #print kernel
        #kernel = np.fliplr( np.flipud(kernel) )
       
        return kernel
        
def convolve(img, kernel, padding = "rep"):
    kernel = np.fliplr( np.flipud(kernel) )
    new_img = np.zeros(img.shape)
    center_row = kernel.shape[0]/2
    center_col = kernel.shape[1]/2
    for img_row in range(img.shape[0]):
        for img_col in range(img.shape[1]):
            for kernel_row in range(kernel.shape[0]):
                for kernel_col in range(kernel.shape[1]):
                    i = img_row + kernel_row - center_row
                    j = img_col + kernel_col - center_col 
                    if padding == "zero":
                        if i >= 0 and i < img.shape[0] and j >= 0 and j < img.shape[1]:
                            new_img[img_row][img_col] += img[i][j] * kernel[kernel_row][kernel_col]
                    else:
                        new_img[img_row][img_col] += img[(i%img.shape[0])][(j%img.shape[1])] * kernel[kernel_row][kernel_col]
    return new_img

def dft(img, kernel):
    # dft_result = fftconvolve(img, kernel, 'same')
    # img_shape = img.shape
    padded_kernel = np.zeros(img.shape)
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    kernel_rows = kernel.shape[0]
    kernel_cols = kernel.shape[1]
    row_start = (img_rows/2 - kernel_rows/2)
    col_start = (img_cols/2 - kernel_cols/2)
    padded_kernel[row_start:(row_start + kernel_rows), col_start:(col_start + kernel_cols)] = kernel
    img_dft = np.fft.fft2(img)
    kernel_dft = np.fft.fft2(padded_kernel)
#    plt.imshow(kernel_dft.astype(np.int8))
#    plt.show()
#    plt.figure()
#    kernel_dft_shift = np.fft.fftshift(kernel_dft)
#    img_dft_shift = np.fft.fftshift(img_dft)
    f_mul = kernel_dft * img_dft
    
    
    
#    plt.imshow(kernel_dft_shift.astype(np.int8))
#    plt.show()
    inverse = np.fft.ifft2(f_mul)
    
    f_mul_shift = np.fft.ifftshift(inverse)
   # inverse_mag = np.abs(inverse)
#    return abs(f_mul_shift)
    return f_mul_shift
    
def low_pass_filter(shape = 3):
    new_filter = np.zeros( (shape, shape) )
    filter_center = shape/ 2
    for i in range(shape):
        for j in range(shape):
            i_idx = i - filter_center
            j_idx = j - filter_center
            new_filter[i][j] = np.exp(- (i_idx**2 + j_idx**2) )
    new_filter /= new_filter.sum()
    return new_filter

def high_pass_filter(shape = 3):
    new_filter = np.zeros( (shape, shape) )
    filter_center = shape/ 2
    for i in range(shape):
        for j in range(shape):
            i_idx = i - filter_center
            j_idx = j - filter_center
            new_filter[i][j] = (1 - (i_idx**2 + j_idx**2) ) * ( np.exp(- (i_idx**2 + j_idx**2) ))
    new_filter /= new_filter.sum()
    print new_filter
    return new_filter
    
def band_pass_filter(small_radius=8, large_radius=15, shape = 7):
    freq_shape = shape*100
    new_filter = np.zeros((freq_shape, freq_shape))
    kernel_center = freq_shape/2
    for i in range(freq_shape):
        for j in range(freq_shape):
            i_idx = i - kernel_center
            j_idx = j - kernel_center
            if (i_idx**2 + j_idx**2) < large_radius**2 and \
                (i_idx**2 + j_idx**2) >= small_radius**2:
                new_filter[i][j] = 1
#    plt.imshow(new_filter, cmap='gray', interpolation='none')
#    plt.show()
    spatial_filter = np.fft.ifft2(new_filter)
    spatial_filter_shift = np.fft.ifftshift(spatial_filter)
    spatial_filter_abs = abs(spatial_filter_shift)
#    plt.figure()
#    plt.imshow(spatial_filter_abs, cmap='gray')
#    plt.show()
    row_start = (freq_shape/2 - shape/2)
    col_start = (freq_shape/2 - shape/2)
    spatial_filter_cropped = spatial_filter_abs[row_start:(row_start + shape), col_start:(col_start + shape)]
#    plt.figure()
#    plt.imshow(spatial_filter_cropped, cmap='gray')
#    plt.show()
    spatial_filter_cropped /= spatial_filter_cropped.sum()
    print spatial_filter_cropped

    return spatial_filter_cropped

def img_rect_mul(img, rect_size=50):
    kernel = np.zeros(img.shape)
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    img_x_center = img_rows/ 2
    img_y_center = img_cols/ 2
    row_start = (img_x_center - rect_size/2)
    col_start = (img_y_center - rect_size/2)
    kernel[row_start:(row_start + rect_size), col_start:(col_start + rect_size)] = 1
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)

    f_mul = kernel * img_dft_shift
    inverse = np.fft.ifft2(f_mul)
    #f_mul_shift = np.fft.ifftshift(inverse)
    return (abs(inverse), abs(img_dft_shift), kernel)

def img_hamm_mul(img, hamm_size=50):
    kernel = np.zeros(img.shape)
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    img_x_center = img_rows/ 2
    img_y_center = img_cols/ 2
    row_start = (img_x_center - hamm_size/2)
    col_start = (img_y_center - hamm_size/2)
    hamming_vals = np.hamming(hamm_size)
    hamming_vals = hamming_vals.reshape((-1,1))
    hamming_vals = hamming_vals.T * hamming_vals
    kernel[row_start:(row_start + hamm_size), col_start:(col_start + hamm_size)] = hamming_vals
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)

    f_mul = kernel * img_dft_shift
    inverse = np.fft.ifft2(f_mul)
    #f_mul_shift = np.fft.ifftshift(inverse)
    return (abs(inverse), abs(img_dft_shift), kernel)

def img_hamm_rect_mul(img, rect_size=50, hamm_size=50):
    (rect_inverse, rect_img_dft_shift, rect_kernel) = img_rect_mul(img, rect_size)
    (hamm_inverse, hamm_img_dft_shift, hamm_kernel) = img_hamm_mul(img, hamm_size)
    return (rect_kernel, rect_inverse, hamm_kernel,hamm_inverse)

    
def img_stretch(img, x_scale=2):
    img_str = imresize(img, (img.shape[0], int(img.shape[1]*x_scale)), interp='nearest' )
    img_dft = np.fft.fft2(img)
    img_str_dft = np.fft.fft2(img_str)
    img_dft_shift = np.fft.fftshift(img_dft)
    img_str_dft_shift = np.fft.fftshift(img_str_dft)
    
#    img_dft_shift_phase = np.arctan2(img_dft_shift.imag,img_dft_shift.real)
#    img_str_dft_shift_phase = np.arctan2(img_str_dft_shift.imag,img_str_dft_shift.real)
    return (img_str, abs(img_dft_shift), abs(img_str_dft_shift))
    
def img_mirror(img):
    img_mirr = np.zeros(img.shape)
    img_mirr[:,:] = img[:, ::-1]
#    img_mirr[:,:] = img_mirr[::-1,:]
    img_dft = np.fft.fft2(img)
    img_mirr_dft = np.fft.fft2(img_mirr)
    img_dft_shift = np.fft.fftshift(img_dft)
    img_mirr_dft_shift = np.fft.fftshift(img_mirr_dft)
#    img_phase = np.arctan2(img_dft.imag,img_dft.real)
#    img_mirr_phase = np.arctan2(img_mirr_dft.imag,img_mirr_dft.real)
#    print abs(img_dft_shift[ (img_dft_shift.shape[0]/2) +3, :])
    return (img_mirr, abs(img_dft_shift), abs(img_mirr_dft_shift))
    
def zero_phase(img):
    img_dft = np.fft.fft2(img)
#    print(img_dft.imag)
    new_dft = img_dft.copy()
    new_dft.real = abs(img_dft)
    new_dft.imag = 0
    inverse = np.fft.ifft2(new_dft)
    inverse_shift = np.fft.ifftshift(inverse)
    
    img_phase = np.arctan2(img_dft.imag,img_dft.real)
    new_dft_phase = np.arctan2(new_dft.imag,new_dft.real)
    return (abs(inverse_shift), img_phase, new_dft_phase)

def zero_magnitude(img):
    img_dft = np.fft.fft2(img)
#    print(img_dft.imag)
    new_dft = img_dft.copy()
    phase_1 = np.arctan2(new_dft.imag,new_dft.real)
    new_dft = new_dft/ abs(img_dft) # setting magnitude to one
    phase_2 = np.arctan2(new_dft.imag,new_dft.real)
#    print( abs(phase_1-phase_2).sum() )
#    print( abs(new_dft) )
    inverse = np.fft.ifft2(new_dft)
    return (abs(inverse), abs(img_dft), abs(new_dft))    

def cos_multiply(img):
    cos_arr = np.zeros(img.shape)
    # 2*np.pi * 20should correspond to shift = 20
    v_t = np.linspace(0, 2*np.pi * 20, img.shape[1]) 
    for j in range(img.shape[0]):
        cos_arr[j, :] = (np.cos(v_t) )#+ 1)/ 2
#    cos_arr1 = np.zeros(img.shape)
#    v_t = np.linspace(0, 2*np.pi * 100, img.shape[1])
#    for j in range(img.shape[0]):
#        cos_arr1[j, :] = (np.cos(v_t) )#+ 1)/ 2
    mul = cos_arr * img
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)
#    img_phase = np.arctan2(img_dft.imag,img_dft.real)
    mul_dft = np.fft.fft2(mul)
    mul_dft_shift = np.fft.fftshift(mul_dft)
#    new_dft_phase = np.arctan2(mul_dft.imag,mul_dft.real)
    return (mul, abs(img_dft_shift), abs(mul_dft_shift))    

def sin_multiply(img):
    sin_arr = np.zeros(img.shape)
    # 2*np.pi * 20should correspond to shift = 20
    v_t = np.linspace(0, 2*np.pi * 10, img.shape[1]) 
    for j in range(img.shape[0]):
        sin_arr[:, j] = (np.cos(v_t) )#+ 1)/ 2
    mul = sin_arr * img
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)
    mul_dft = np.fft.fft2(mul)
    mul_dft_shift = np.fft.fftshift(mul_dft)
    return (mul, abs(img_dft_shift), abs(mul_dft_shift))
    
def cos_sin_multiply(img):
    sin_arr = np.zeros(img.shape)
    # 2*np.pi * 10 should correspond to shift = 10
    v_t_sin = np.linspace(0, 2*np.pi * 10, img.shape[1]) 
    for j in range(img.shape[0]):
        sin_arr[:, j] = (np.cos(v_t_sin) )#+ 1)/ 2
    cos_arr = np.zeros(img.shape)
    # 2*np.pi * 20 should correspond to shift = 20
    v_t_cos = np.linspace(0, 2*np.pi * 20, img.shape[1]) 
    for j in range(img.shape[0]):
        cos_arr[j, :] = (np.cos(v_t_cos) )#+ 1)/ 2
    mul = cos_arr * sin_arr * img
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)
    mul_dft = np.fft.fft2(mul)
    mul_dft_shift = np.fft.fftshift(mul_dft)
    return (mul, abs(img_dft_shift), abs(mul_dft_shift))

def rotate_img(img, angle=30):
    img_rotated = imrotate(img, angle=angle)
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)
    img_rotated_dft = np.fft.fft2(img_rotated)
    img_rotated_dft_shift = np.fft.fftshift(img_rotated_dft)
    return (img_rotated, abs(img_dft_shift), abs(img_rotated_dft_shift))
    
def rotate_multiply_img(img, angle):
    img_rotated = imrotate(img, angle=angle)
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)
    img_rotated_dft = np.fft.fft2(img_rotated)
    img_rotated_dft_shift = np.fft.fftshift(img_rotated_dft)
    
    cos_arr = np.zeros(img.shape)
    v_t = np.linspace(0, 2*np.pi * 5, img.shape[1]) 
    for j in range(img.shape[0]):
        cos_arr[j, :] = (np.cos(v_t) )#+ 1)/ 2
    mul = cos_arr * img_rotated
    mul_dft = np.fft.fft2(mul)
    mul_dft_shift = np.fft.fftshift(mul_dft)

    return (mul, abs(img_dft_shift), abs(mul_dft_shift))
    
def ft_fov_increase(img):
    increase_factor = 2
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)
    new_dft  = np.zeros((img_dft.shape[0]*2, img_dft.shape[1]*2), dtype=complex)
    
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    img_x_center = img_rows/ 2
    img_y_center = img_cols/ 2
    row_start = new_dft.shape[0]/2 - img_x_center
    col_start = new_dft.shape[1]/2 - img_y_center
    new_dft[row_start:(row_start + img_rows), col_start:(col_start + img_cols)] = img_dft_shift

    inverse = np.fft.ifft2(new_dft)
    return (abs(img_dft_shift), abs(new_dft), abs(inverse))
    
    
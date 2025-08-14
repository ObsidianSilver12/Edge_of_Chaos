import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
img_path = r"C:\Kim\Claude\Edge_of_Chaos\glyphs\assets\ChatGPT Image May 12, 2025, 07_48_15 PM.png"
img = Image.open(img_path)
img_array = np.array(img)

# Analyze LSB (Least Significant Bit - common in steganography)
# Extract the least significant bits
red_channel = img_array[:,:,0]
green_channel = img_array[:,:,1] 
blue_channel = img_array[:,:,2]

# Get LSBs
red_lsb = red_channel % 2
green_lsb = green_channel % 2
blue_lsb = blue_channel % 2

# Visualize LSBs to see patterns
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(red_lsb, cmap='gray')
plt.title('Red Channel LSB')
plt.subplot(132)
plt.imshow(green_lsb, cmap='gray')
plt.title('Green Channel LSB')
plt.subplot(133)
plt.imshow(blue_lsb, cmap='gray')
plt.title('Blue Channel LSB')
plt.savefig('lsb_analysis.png')

# Perform frequency domain analysis (might reveal hidden patterns)
# FFT on each channel
red_fft = np.fft.fft2(red_channel)
red_fft_shifted = np.fft.fftshift(red_fft)
green_fft = np.fft.fft2(green_channel)
green_fft_shifted = np.fft.fftshift(green_fft)
blue_fft = np.fft.fft2(blue_channel)
blue_fft_shifted = np.fft.fftshift(blue_fft)

# Visualize frequency domains
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(np.log(1+np.abs(red_fft_shifted)), cmap='viridis')
plt.title('Red Channel Frequency')
plt.subplot(132)
plt.imshow(np.log(1+np.abs(green_fft_shifted)), cmap='viridis')
plt.title('Green Channel Frequency')
plt.subplot(133)
plt.imshow(np.log(1+np.abs(blue_fft_shifted)), cmap='viridis')
plt.title('Blue Channel Frequency')
plt.savefig('frequency_analysis.png')

print("Analysis complete. Check the output images for patterns.")
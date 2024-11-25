# pip install flask numpy pandas matplotlib opencv-python

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	if 'image' not in request.files:
		return redirect(request.url)

	file = request.files['image']
	if file.filename == '':
		return redirect(request.url)

	if file:
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
		file.save(filepath)
		process_image(filepath)
		return redirect(url_for('display_results', filename=file.filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
	return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/results/<filename>')
def display_results(filename):
	return render_template('results.html', filename=filename)

def process_image(filepath):
	# Load the image using OpenCV
	img = cv2.imread(filepath)
	original_path = os.path.join(app.config['PROCESSED_FOLDER'], 'original.png')
	cv2.imwrite(original_path, img)

	def power_law_transform(image, gamma):
		# Normalize the image
		normalized_img = image / 255.0
		# Apply Power-Law transformation
		transformed_img = np.power(normalized_img, gamma)
		# Scale back to [0, 255]
		transformed_img = np.uint8(transformed_img * 255)
		return transformed_img

	# Gray-Level Slicing function
	def gray_level_slicing(image, min_val, max_val):
		# Create a copy of the image to apply gray level slicing
		output_image = np.zeros_like(image)

		# Pixels within the specified range are enhanced
		# Assign max intensity to the specified range
		output_image[(image >= min_val) & (image <= max_val)] = 255

		return output_image

	# Enhancement 1: Power-law.
	gamma = 0.5
	enhanced1 = power_law_transform(img, gamma)  # Brightness and contrast
	enhanced1_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced1.png')
	cv2.imwrite(enhanced1_path, enhanced1)

	# Enhancement 2: Graylevel slicing.

	# Define the range of gray levels to be enhanced
	min_gray = 50  # minimum gray level to enhance
	max_gray = 100  # maximum gray level to enhance
	enhanced2 = gray_level_slicing(img, min_gray, max_gray)
	enhanced2_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced2.png')
	cv2.imwrite(enhanced2_path, enhanced2)

	# Compression and segmentation placeholders
	# Save paths for where compressed and segmented images would go
	compression1_path = os.path.join(app.config['PROCESSED_FOLDER'], 'compressed1.png')
	compression2_path = os.path.join(app.config['PROCESSED_FOLDER'], 'compressed2.png')
	segmentation1_path = os.path.join(app.config['PROCESSED_FOLDER'], 'segmented1.png')
	segmentation2_path = os.path.join(app.config['PROCESSED_FOLDER'], 'segmented2.png')

	# Placeholder: You can implement compression and segmentation logic here
	# cv2.imwrite(compression1_path, ...)
	# cv2.imwrite(segmentation1_path, ...)

	# Use Matplotlib to create visualizations if needed
	# Example: Save grayscale histogram
	plt.hist(enhanced2.ravel(), bins=256, range=(0, 256), color='gray')
	plt.title('Grayscale Histogram')
	plt.xlabel('Pixel Intensity')
	plt.ylabel('Frequency')
	histogram_path = os.path.join(app.config['PROCESSED_FOLDER'], 'histogram.png')
	plt.savefig(histogram_path)
	plt.close()

if __name__ == '__main__':
	app.run(debug=True)

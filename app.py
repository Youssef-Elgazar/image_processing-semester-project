# pip install flask numpy pandas matplotlib opencv-python scikit-image

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from skimage.segmentation import clear_border
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
		transformed_img = np.power(normalized_img, gamma)
		transformed_img = np.uint8(transformed_img * 255)
		return transformed_img

	def gray_level_slicing(image, min_val, max_val):
		output_image = np.zeros_like(image)
		output_image[(image >= min_val) & (image <= max_val)] = 255
		return output_image

	# Enhancement 1: Power-law transformation
	gamma = 0.5
	enhanced1 = power_law_transform(img, gamma)
	enhanced1_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced1.png')
	cv2.imwrite(enhanced1_path, enhanced1)

	# Enhancement 2: Gray-level slicing
	min_gray = 50
	max_gray = 100
	enhanced2 = gray_level_slicing(img, min_gray, max_gray)
	enhanced2_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced2.png')
	cv2.imwrite(enhanced2_path, enhanced2)

	# Compression 1: JPEG compression (lower quality)
	compression1_path = os.path.join(app.config['PROCESSED_FOLDER'], 'compressed1.jpg')
	cv2.imwrite(compression1_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])

	# Compression 2: PCA compression
	def pca_compression(image, num_components):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		mean, eigenvectors = cv2.PCACompute(gray, mean=None, maxComponents=num_components)
		reconstructed = cv2.PCAProject(gray, mean, eigenvectors)
		return cv2.convertScaleAbs(reconstructed)

	compressed2 = pca_compression(img, num_components=50)
	compression2_path = os.path.join(app.config['PROCESSED_FOLDER'], 'compressed2.png')
	cv2.imwrite(compression2_path, compressed2)

	# Segmentation 1: Watershed Segmentation
	def watershed_segmentation(image):
		# Step1: Load and preprocess the image
		cells=image[:,:,0]

		#Step2: Apply a Binary Threshold
		ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		#Step3: Remove Noise Using Morphological Transformations
		kernel = np.ones((3,3),np.uint8)
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)


		opening = clear_border(opening) #Remove edge touching grains

		#Step4: Identify Background Area
		sure_bg = cv2.dilate(opening,kernel,iterations=1)

		#Step5: Identify Foreground Area using Distance Transform
		dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
		_, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

		#Step 6: Find Unknown Region by Subtracting Foreground from Background
		sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
		unknown = cv2.subtract(sure_bg,sure_fg)

		#Step 7: Label Markers for Watershed
		_, markers = cv2.connectedComponents(sure_fg)
		# Increment marker values by 1 so that background is 1 and other regions are 2, 3, etc.
		markers = markers+10
		#Mark unknown regions with 0
		markers[unknown==255] = 0

		#Step8: Apply Watershed and mark boundaries
		markers = cv2.watershed(image,markers)

		watershed_result = image.copy()
		# Objects’ boundaries are marked
		watershed_result[markers == -1] = [0, 0, 255]

		return watershed_result

	segmented1 = watershed_segmentation(img.copy())
	segmentation1_path = os.path.join(app.config['PROCESSED_FOLDER'], 'segmented1.png')
	cv2.imwrite(segmentation1_path, segmented1)

	# Segmentation 2: K-means Clustering
	def kmeans_segmentation(image, clusters):
		reshaped = image.reshape((-1, 3))
		reshaped = np.float32(reshaped)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
		_, labels, centers = cv2.kmeans(reshaped, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
		centers = np.uint8(centers)
		segmented = centers[labels.flatten()]
		return segmented.reshape(image.shape)

	segmented2 = kmeans_segmentation(img, clusters=3)
	segmentation2_path = os.path.join(app.config['PROCESSED_FOLDER'], 'segmented2.png')
	cv2.imwrite(segmentation2_path, segmented2)

	# Save histogram as visualization
	plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray')
	plt.title('Grayscale Histogram')
	plt.xlabel('Pixel Intensity')
	plt.ylabel('Frequency')
	histogram_path = os.path.join(app.config['PROCESSED_FOLDER'], 'histogram.png')
	plt.savefig(histogram_path)
	plt.close()

if __name__ == '__main__':
	app.run(debug=True)

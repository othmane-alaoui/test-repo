from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import base64, os
import numpy as np
import time
from timeit import default_timer as timer
import cv2
import os, random

import sys
import inspect
import colorsys
import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
import onnx
import onnxruntime

import io


route=os.path.dirname(os.path.abspath(__file__))
des=os.path.join(route,"static","test.png")


app = Flask(__name__)

def letterbox_image(image, size):
	'''resize image with unchanged aspect ratio using padding'''
	iw, ih = image.size
	w, h = size
	scale = min(w/iw, h/ih)
	nw = int(iw*scale)
	nh = int(ih*scale)

	image = image.resize((nw,nh), Image.BICUBIC)
	new_image = Image.new('RGB', size, (128,128,128))
	new_image.paste(image, ((w-nw)//2, (h-nh)//2))
	return new_image


class YOLO(object):
	def __init__(self,classes_path):
		self.classes_path = classes_path

		self.class_names = self._get_class()
		self.sess = K.get_session()
		self.session = None
		self.final_model = None

		print("classes "+str(len(self.class_names)))

		
		# Generate colors for drawing bounding boxes.
		hsv_tuples = [(x / len(self.class_names), 1., 1.)
					  for x in range(len(self.class_names))]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		self.colors = list(
			map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
				self.colors))
		np.random.seed(10101)  # Fixed seed for consistent colors across runs.
		np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
		np.random.seed(None)  # Reset seed to default.
		K.set_learning_phase(0)

	@staticmethod
	def _get_data_path(name):
		path = os.path.expanduser(name)
		
		return "./"+path

	def _get_class(self):
		classes_path = self._get_data_path(self.classes_path)
		with open(classes_path) as f:
			class_names = f.readlines()
		class_names = [c.strip() for c in class_names]
		return class_names

	def predict(self, image):
		start = timer()
		model_image_size = (416, 416)
		boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
		image_data = np.array(boxed_image, dtype='float32')
		image_data /= 255.
		image_data = np.transpose(image_data, [2, 0, 1])

		image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
		feed_f = dict(zip(['input_1', 'image_shape'],
						  (image_data, np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2))))
		all_boxes, all_scores, indices = self.session.run(None, input_feed=feed_f)

		out_boxes, out_scores, out_classes = [], [], []

		for idx_ in indices:
			out_classes.append(idx_[1])
			out_scores.append(all_scores[tuple(idx_)])
			idx_1 = (idx_[0], idx_[2])
			out_boxes.append(all_boxes[idx_1])
		end = timer()
		print("[INFO] inference took {:.4f} seconds".format(end - start))

		#create images with bounding boxes
		font = ImageFont.truetype(font=self._get_data_path('font/FiraMono-Medium.otf'),
								  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
		thickness = (image.size[0] + image.size[1]) // 300

		for i, c in reversed(list(enumerate(out_classes))):
			predicted_class = self.class_names[c]
			box = out_boxes[i]
			score = out_scores[i]

			label = '{} {:.2f}'.format(predicted_class, score)
			draw = ImageDraw.Draw(image)
			label_size = draw.textsize(label, font)

			top, left, bottom, right = box
			top = max(0, np.floor(top + 0.5).astype('int32'))
			left = max(0, np.floor(left + 0.5).astype('int32'))
			bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
			right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

			inference_text = 'Inference time : %.1f fps' % (1.0/(end-start))

			if top - label_size[1] >= 0:
				text_origin = np.array([left, top - label_size[1]])
			else:
				text_origin = np.array([left, top + 1])

			for i in range(thickness):
				draw.rectangle(
					[left + i, top + i, right - i, bottom - i],
					outline=self.colors[c])
			draw.rectangle(
				[tuple(text_origin), tuple(text_origin + label_size)],
				fill=self.colors[c])
			draw.text(text_origin, label, fill=(0, 0, 0), font=font)
			draw.text((5,5),fill=(0,0,255) ,text=inference_text)
			del draw

		return image


@app.route('/')
@app.route('/home')
def home():
	#remove images
	folder_path = (r"static")
	test = os.listdir(folder_path)
	for images in test:
		if images.endswith(".png"):
			os.remove(os.path.join(folder_path, images))
	return render_template('home.html')

@app.route('/cam')
def Camera():
	#remove images
	folder_path = (r"static")
	test = os.listdir(folder_path)
	for images in test:
		if images.endswith(".png"):
			os.remove(os.path.join(folder_path, images))
		
	return render_template('cam.html')

@app.route('/saveimage',methods=['POST'])
def saveImage():
	data_url = request.values['imageBase64']
	image_encoded = data_url.split(',')[1]
	body = base64.b64decode(image_encoded.encode('utf-8'))
	file=open(des,"wb")
	file.write(body)
	return "ok"


@app.route('/process')
def process():
	return render_template('process.html')


@app.route('/showimage')
def showImage():
	print("in showImage")
	return render_template('output.html')

def detect_img(yolo, img_url, output_url, model_file_name):
	image = Image.open(img_url)
	
	yolo.session = onnxruntime.InferenceSession(model_file_name)
	r_image = yolo.predict(image)
	
	return r_image




@app.route('/output')
def output():
	# arguments
	model_file_name = "mask_bb2.onnx"
	classes = "custom_classes.txt"
	output_file = "static/output.png"
	target_opset = 10
	image = "static/test.png"

	r_img = detect_img(YOLO(classes), image, output_file, model_file_name)
	img_pred = r_img.save("static/output.png")
	imgval="../static/{}".format(img_pred)
	return render_template('output.html', imgval=imgval)


if __name__ == '__main__':
	app.debug = True
	app.run(host='0.0.0.0', port=8088)
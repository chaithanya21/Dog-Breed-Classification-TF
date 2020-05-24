# Dog-Breed-Classification-TF
This Project is all about building a Deep Learning Pipe Line to process the real world , user supplied Images. Given an Image of a dog the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.(TensorFlow Version) 

## Project Overview

[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

Welcome to the Convolutional Neural Networks (CNN) project. The project focuses on building a **dog breed identification**  pipeline that can be used within a web or mobile app to process real-world, user-supplied images.Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.

```	
git clone https://github.com/chaithanya21/Dog-Breed-Classification-TF.git
cd Dog-Breed-Classification-TF
```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/Dog-Breed-Classification-TF/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/Dog-Breed-Classification-TF/lfw`.If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/Dog-Breed-Classification-TF/bottleneck_features`.

5.  __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.


	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate Dog-Breed-Classification-TF
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate Dog-Breed-Classification-TF
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate Dog-Breed-Classification-TF
	```
  7.**If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.
  
  
	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name Dog-Breed-Classification-TF python=3.5
	source activate Dog-Breed-Classification-TF
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name Dog-Breed-Classification-TF python=3.5
	activate Dog-Breed-Classification-TF
	pip install -r requirements/requirements.txt
	```
8. (Optional) **If you are using AWS**, install Tensorflow.

```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```
10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 

```
python -m ipykernel install --user --name Dog-Breed-Classification-TF --display-name "Dog-Breed-Classification-TF"
```
11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```
12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

# Results

The Model acheived a Test accuracy of 1.3% when Trained Using a Convolutional Neural Network From Scratch , Using Data Augmentation and Batch Normalization Techniques with Much Deeper Architrecture Can help to improve the model performance.

The Model acheived a Test Accuracy of 83% when Trained using a Pre-Trained Resnet50 Model for 25 epochs, The Performance of the model can be improved using othet pre-trained models such as Xception and Inception Networks.

## Some of the Results obtained After Testing on Real world Images

<img src='https://github.com/chaithanya21/Dog-Breed-Classification-TF/blob/master/Results/Result1.png' >

# Blog Post 


[<h2>How Deep Learning Helps Identify Dog Breeds</h2>](https://medium.com/@chaithanyakumar_91513/how-deep-learning-helps-identify-dog-breeds-ec6dc6575e87?source=your_stories_page---------------------------)





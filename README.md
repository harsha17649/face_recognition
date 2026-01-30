# Face Verification Project (Siamese Network)

## What is this project?
Hi! This is a project I built to create a custom Face Verification system. Think of it like the FaceID on your phone. It uses a special type of AI called a **Siamese Network** to look at two faces and decide if they are the same person or not.

I didn't just want it to recognize *a* face; I wanted it to verify *my* face specifically.

## How does a "Siamese Network" work?
Most AI just looks at one picture and says "That's a cat." But a Siamese Network is like a "Twin" network.
1. It takes two images at the same time.
2. It runs both images through the exact same "brain" (embedding layer) to turn the pictures into a list of numbers.
3. Then, it compares those numbers.
   * If the numbers are close, the faces are the **same**.
   * If the numbers are far apart, the faces are **different**.

To teach it, I used three types of images:
* **Anchor:** A reference photo of me.
* **Positive:** Another photo of me (matches the Anchor).
* **Negative:** A photo of a stranger (doesn't match).

## Tools I Used
I used Python and some specific libraries to make this work:
* **TensorFlow/Keras:** This is the main library for building the deep learning model.
* **OpenCV:** I used this to access my webcam and grab images in real-time.
* **Matplotlib:** This is just to help me draw graphs and see the images while I was coding.
* **NumPy:** This handles all the math and number crunching behind the scenes.

## Project Structure
To keep things organized, the code creates these folders automatically:
* `data/anchor`: Stores the main reference photos of me.
* `data/positive`: Stores other photos of me to train the model.
* `data/negative`: Stores photos of random strangers.
* `training_checkpoints`: Saves the model progress so I don't lose it if my computer crashes.
* `application_data`: Stores images used during the final real-time test.

## Step-by-Step: What the Code Does

### 1. Getting the Data
First, I had to get the pictures.
* **For Strangers (Negatives):** I downloaded the "Labelled Faces in the Wild" (LFW) dataset. This gave me thousands of random faces to use as "negative" examples.
* **For Me (Positives/Anchors):** I wrote a script using OpenCV to turn on my webcam.
    * I press **'a'** to take an Anchor photo.
    * I press **'p'** to take a Positive photo.
    * I also did some data augmentation (creating modified copies of my images) to make the dataset bigger and the model smarter.

### 2. Preprocessing (Cleaning up)
The computer can't read a raw photo easily. So, I wrote a `preprocess` function that:
1.  Loads the image file.
2.  Resizes it to **100x100 pixels**.
3.  Scales the colors down to be between 0 and 1 (instead of 0 to 255). This helps the AI learn faster.

### 3. Building the Model
I built the network using three main parts:
* **Embedding Layer:** This is a Convolutional Neural Network (CNN). It scans the face for shapes, eyes, and noses and turns them into a "feature vector" (a list of numbers).
* **Distance Layer:** This is a custom layer I wrote called `L1Dist`. It basically subtracts the numbers from the two images to measure the distance between them.
* **Classifier:** The final layer decides the result. It outputs `1` if the faces match and `0` if they don't.

### 4. Training
I set up a custom training loop using `BinaryCrossentropy` (a fancy way of calculating error) and the `Adam` optimizer.
* I feed the model batches of data (Anchor + Positive vs. Anchor + Negative).
* It calculates the loss (mistakes) and updates the weights using `GradientTape`.
* I also save checkpoints during training so I can resume later.

### 5. Evaluation
To check if it was working, I used metrics like **Precision** (how accurate it is) and **Recall** (how many correct matches it found). I plotted these on a graph to watch the model get smarter over time.

## Real-Time Test
This is the fun part! After training, I wrote a `verify` function.
1.  I open the webcam again.
2.  A generic "verification" image is saved in the background.
3.  When I press **'v'**, the model compares my live webcam face against 50 "input" images of me.
4.  If enough of them match (based on a detection threshold), it says **Verified!**

## How to Run This
If you want to try this code:
1.  Install the requirements: `pip install tensorflow opencv-python matplotlib numpy`.
2.  Download the **LFW dataset** (for the negative images) and extract it.
3.  Run the notebook cells in order.
4.  Use the webcam collection cell to grab your own face data (press 'a' for anchor, 'p' for positive).
5.  Train the model (this might take a while if you don't have a GPU!).
6.  Run the final section to test it with your webcam in real-time.

---
*Note: I set up the code to limit GPU memory usage at the start so it doesn't crash my system.*
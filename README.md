# Face Verification Project (Siamese Network)

## What is this project?
Hi! This is a project I made to teach a computer how to recognize faces. It is called a **Siamese Network**. Basically, it works like the FaceID unlock on a phone. The goal is to give the computer two pictures and have it guess: "Is this the same person?" or "Are these different people?"

I wrote this code in a Python Jupyter Notebook.

## How does it actually work?
To make this work, I used a special method. Instead of just showing the computer one face, I show it three things at once:
1.  **Anchor:** A photo of me.
2.  **Positive:** Another photo of me (so it matches the Anchor).
3.  **Negative:** A photo of a random stranger.

The computer looks at these and learns that the **Anchor** and **Positive** should be close friends (similar), but the **Anchor** and **Negative** should be far apart (different).

## The Tools I Used
To build this, I had to install some Python tools:
* **TensorFlow:** This is the main brain for learning.
* **OpenCV:** This helps the computer see images through the webcam.
* **Matplotlib:** This acts like a graph paper to draw and show the pictures.
* **NumPy:** This handles all the math and number lists.

## Step-by-Step: What I Did in the Code

### 1. Setting up the folders
First, I wrote code to make three folders on the computer: `data/positive`, `data/negative`, and `data/anchor`. This keeps everything organized so the photos don't get mixed up.

### 2. Getting the pictures
* **The Strangers (Negatives):** I can't take photos of thousands of strangers myself, so I used a famous dataset called "Labelled Faces in the Wild" (LFW). I unzip this file to get the negative images.
* **My Photos (Positives & Anchors):** I wrote a script to use my webcam.
    * If I press the **'a'** key, it takes an Anchor photo.
    * If I press the **'p'** key, it takes a Positive photo.

### 3. Fixing the images (Preprocessing)
Computers like things to be neat. So, I made a function called `preprocess`. It takes the photo, shrinks it down to a square (100x100 pixels), and changes the colors to numbers between 0 and 1. This makes it easier for the model to read.

### 4. Building the "Brain" (The Model)
This is the cool part. I built a deep learning model.
* **The Embedding Layer:** This takes the photo and crushes it down into a list of numbers (features). It uses things called "Conv2D" and "MaxPooling" to find shapes and lines in the face.
* **The Distance Layer:** I made a custom layer called `L1Dist`. It basically subtracts the numbers from the two pictures to see how different they are.
* **The Classifier:** Finally, there is a dense layer with a "sigmoid" activation. This just spits out a score: 1 means it's a match, and 0 means it's not.

## How to Run It
Since I don't have the dataset downloaded yet, I haven't run the training part. But if you want to use it:
1.  Make sure you have the libraries installed (`pip install tensorflow opencv-python matplotlib`).
2.  Download the **LFW dataset** and put it in the folder.
3.  Run the cells to open your webcam and collect your own photos.
4.  Run the model cells to start the learning process.

## Final Note
I am using a GPU setup in the code to make it run faster because training on just a regular CPU takes forever!.
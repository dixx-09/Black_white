import numpy as np
import cv2
import os

#--------Model file paths--------#
proto_file = r'D:/new projects/black and white/model/colorization_deploy_v2.prototxt'
model_file = r'D:/new projects/black and white/model/colorization_release_v2.caffemodel'
hull_pts = r'D:/new projects/black and white/model/pts_in_hull.npy'
image_directory = r'D:/new projects/black and white/images'  # Directory containing images
#--------------#--------------#

# Check if model files exist
for file in [proto_file, model_file, hull_pts]:
    if not os.path.isfile(file):
        print(f"Error: {file} not found.")
        exit()

#--------Reading the model params--------#
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)
#-----------------------------------#---------------------#

# List all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image in the directory
for image_file in image_files:
    # Construct full image path
    img_path = os.path.join(image_directory, image_file)

    #-----Reading and preprocessing image--------#
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image from {img_path}.")
        continue

    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    #-----------------------------------#---------------------#

    # Add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    #-----------------------------------#---------------------#

    # Resize the image for the network
    resized = cv2.resize(lab_img, (224, 224))
    # Split the L channel
    L = cv2.split(resized)[0]
    # Mean subtraction
    L -= 50
    #-----------------------------------#---------------------#

    # Predicting the ab channels from the input L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted 'ab' volume to the same dimensions as our input image
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    # Take the L channel from the image
    L = cv2.split(lab_img)[0]
    # Join the L channel with predicted ab channel
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    # Then convert the image from Lab to BGR 
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Change the image to 0-255 range and convert it from float32 to int
    colorized = (255 * colorized).astype("uint8")

    # Let's resize the images and show them together
    img = cv2.resize(img, (640, 640))
    colorized = cv2.resize(colorized, (640, 640))

    result = cv2.hconcat([img, colorized])

    # Display the result
    cv2.imshow("Grayscale -> Colour", result)

    # Wait for a key press for a specified duration (e.g., 2000 ms)
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Clean up
cv2.destroyAllWindows()
This code analyzes and visualizes Discrete Cosine Transform (DCT) coefficients of two 
randomly selected images—one from a folder containing AI-generated images and another from a folder with photos.


  Key steps:

1. Load grayscale images from specified folders.
2. Compute the DCT for 8×8 blocks of each image.
3. Extract high-frequency components, representing fine details and potential compression artifacts.
4. Calculate metrics, including the standard deviation of DCT coefficients and the proportion of high-frequency energy.
5. Visualize results using Matplotlib: DCT coefficient heatmaps, DCT coefficient histograms, High-frequency component heatmaps, High-frequency energy as a textual annotation


  Files description:

.\generated -> images 1024x1024 made with GAN-powered service https://magicstudio.com/ai-art-generator/

.\original_compressed -> photos taken by camera. Cut to 1:1 rate, and compressed to 1024x1024 pixels from RAW format

.\original_cropped -> photos taken by camera. The pieces 1024x1024 pixels taken directly from the RAW image

photo_original.jpg -> image converted to JPEG from RAW file without processing

photo_processed.jpg -> processed photo_original (Adobe Photoshop color and white balance auto correction applied)



  Run instruction:

run the DCT_analyze.py
make sure you have all necessary libraries installed

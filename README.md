### K-Means Example ###
This K-Means Example was written for a presentation given at RIT-AI.
RIT's awesome AI club!
Simply clone the repo - then run
 
`python3 kMeans.py`
Then type the path to any image, and your desired number of clusters.

We'll take every pixel of the image, graph each as RGB values, then run k-means on the pixels.
Once grouped, we set the value of every pixel to the average of its associated group.
In this way, we end up with exactly k colors in the entire image.

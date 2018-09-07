import numpy as np
from scipy import misc

def main():

     # get input
    imageName = input('Please enter image name: ')
    points = misc.imread(imageName) # WidthxHeightx3 array
    k = int(input("Please enter k: "))
    
    # get new matrix, save it away
    newPoints = kMeans(points, k)
    misc.imsave('outfile.jpg', np.hstack((newPoints, points)))

def kMeans(points, k, maxIter=10):
    """returns k means clustered points"""
    # init centroids randomly
    centroids = initialize_centroids(points, k)
    
    # iterate as many times as we should
    for i in range(0,maxIter):
        print("Iteration: " + str(i + 1))
        closestCentroids = closest_centroids(points, centroids)
        centroids = move_centroids(points, closestCentroids, centroids)
    
    # return our new matrix
    finalPoints = set_to_centroids(points, centroids, closestCentroids)
    return finalPoints

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    (x, y, z) = points.shape
    centroids = points.copy().reshape(x*y, z)
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroids(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    # Get differences between each point and all centroids (elementwise subtraction)
    # Each row here will be an array of differences between a 
    differences = points - centroids[:, np.newaxis, np.newaxis]

    # Square every value of differences
    squareDifferences = differences**2

    # sum rgb differences (x1-c1 => sum(x1-c1))
    summedSquaredDifferences = squareDifferences.sum(axis=3)

    # square root distances to finish distance formula
    finalDistances = np.sqrt(summedSquaredDifferences)

    # return vector of which centroids are closest
    return np.argmin(finalDistances, axis=0)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    newCentroids = []

    # iterate over all k
    for i in range(centroids.shape[0]):
        
        # find which points are in group i
        # returns a matrix of True/False for all points
        indices = (closest==i)
        
        # gets all points that are in group i
        correspondingPoints = points[indices]

        # get the average rgb values for the points
        average = correspondingPoints.mean(axis=0)

        # add it to our new centroids list
        newCentroids.append(average)

    # return our new centroids
    return np.array(newCentroids)

def set_to_centroids(points, centroids, closestCentroids):
    """returns matrix of  all points set to the value of thir corresponding centroids"""
    # Make a matrix to hold new points
    newPoints = np.zeros(points.shape)
    
    # set all points to corresponding centroids
    (x, y) = closestCentroids.shape
    for i in range(x):
        for j in range(y):
            newPoints[i][j] = centroids[closestCentroids[i][j]]
    # newPoints = centroids[closestCentroids]
    return newPoints

main()

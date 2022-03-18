import numpy

class ClassifiedPoint:
    """ Class for storing information about classified points."""

    classMap = {
        "plus" : 0,
        "circle" : 1
    }

    classMapInvers = {
        0 : "plus",
        1 : "circle"
    }
    def __init__(self, x, y, classification):
        self.point = numpy.array([x,y])
        self.classification = ClassifiedPoint.classMap[classification]

    def l2normToPoint(self, point):
        return numpy.linalg.norm(self.point - point, 2)

    def __str__(self)->str:
        return f"{self.point}, {ClassifiedPoint.classMapInvers[self.classification]}"

    def __eq__(self, other):
        return (self.point[0] == other.point[0] 
                and self.point[1] == other.point[1]
                and self.classification == other.classification)
    
def knn(point, classifiedPoints, k = 1):
    """Find the classifcation of input point compared to classified points."""
    # current max in list of closest points
    currentMax = 0
    # list of closest points
    closestPoints = k*[None]
    for testPindex, testPoint in enumerate(classifiedPoints):
        distance = testPoint.l2normToPoint(point)
        if testPindex < k:
            closestPoints[testPindex] = (testPoint, distance)
            currentMax = max(currentMax, distance)
        elif distance < currentMax:
            index = -1
            for maxIndex, maxVal in enumerate(closestPoints):
                if maxVal[1] == currentMax:
                    index = maxIndex
                    break
            closestPoints[index] = (testPoint, distance)
            currentMax = 0
            for x in closestPoints:
                currentMax = max(currentMax, x[1])

    plusScore = 0
    circleScore = 0
    for closePoint in closestPoints:
        if ClassifiedPoint.classMapInvers[closePoint[0].classification] == "plus":
            plusScore += 1
        if ClassifiedPoint.classMapInvers[closePoint[0].classification] == "circle":
            circleScore += 1
    
    if plusScore > circleScore:
        return "plus"
    else:
        return "circle"

if __name__ == "__main__":
    # Classified Points
    classifiedPoints = (
        ClassifiedPoint(-1, 1, "circle"),
        ClassifiedPoint( 0, 2, "circle"),
        ClassifiedPoint( 3, 0, "circle"),
        ClassifiedPoint( 1, 3, "circle"),
        ClassifiedPoint( 2,-1, "circle"),
        ClassifiedPoint( 0, 3, "plus"),
        ClassifiedPoint( 1,-1, "plus"),
        ClassifiedPoint( 2, 0, "plus"),
        ClassifiedPoint(-1, 2, "plus"),
        ClassifiedPoint( 3, 1, "plus"),
    )
    classifiedPointsList = [
        ClassifiedPoint(-1, 1, "circle"),
        ClassifiedPoint( 0, 2, "circle"),
        ClassifiedPoint( 3, 0, "circle"),
        ClassifiedPoint( 1, 3, "circle"),
        ClassifiedPoint( 2,-1, "circle"),
        ClassifiedPoint( 0, 3, "plus"),
        ClassifiedPoint( 1,-1, "plus"),
        ClassifiedPoint( 2, 0, "plus"),
        ClassifiedPoint(-1, 2, "plus"),
        ClassifiedPoint( 3, 1, "plus"),
    ]


    # leave out error classfication
    totalPointCnt = len(classifiedPoints)
    for k in range(1,10,2):
        incorrectLabels = 0
        for leaveOutPoint in classifiedPoints:
            classifiedPointsList.remove(leaveOutPoint)
            classification = knn(leaveOutPoint.point, classifiedPointsList, k)
            classifiedPointsList.append(leaveOutPoint)

            if ClassifiedPoint.classMap[classification] != leaveOutPoint.classification:
                incorrectLabels += 1
        
        # Results
        print(f"k:{k} Error:{incorrectLabels/totalPointCnt}")



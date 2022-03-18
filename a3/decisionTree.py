

def decisionTree(A1, A2, A3):
    """
            root
            /    \
          A2=0  A2=1
         /      /   \
        0     A1=0 A1=1
              /       \
             0         1
    """

    if A2 == 0:
        return 0
    elif A1 == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    dataPoints = (
                (1,0,0),
                (1,0,1),
                (0,1,0),
                (1,1,1),
                (1,1,0),
            )
    outputValues = (0,0,0,1,1)

    for i, point in enumerate(dataPoints):
        if decisionTree(point[0], point[1], point[2]) != outputValues[i]:
            print(f"incorrect {point}")
        else:
            print(f"correct {point}")

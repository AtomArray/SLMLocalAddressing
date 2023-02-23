import numpy as np

class ThorCam:
    def __init__(self):
        pass

    def gaussian2d(self, x, y, x0, y0, sigma, A):
        return A * np.exp(-((x-x0)**2.0 + (y-y0)**2.0)/(2.0*sigma**2.0))

    def getImage(self):
        width = 256
        height = 256

        image = np.zeros((width, height))

        sigma = 3.0
        A = 100.0

        positions = self.getTweezerPositionsGrid()
        #positions = self.getTweezerPositionsRandom()
        for pos in positions:
            x0 = pos[0]
            y0 = pos[1]

            xs = np.arange(x0-10, x0+10)
            ys = np.arange(y0-10, y0+10)
            xx, yy = np.meshgrid(xs, ys)

            image[xx, yy] += self.gaussian2d(xx, yy, x0, y0, sigma, A)


        return image

    def getTweezerPositionsGrid(self):
        pos = []
        for i in range(12):
            for j in range(12):
                x0 = i * 20 + 10
                y0 = j * 20 + 10
                pos.append([x0, y0])
        return pos

    def getTweezerPositionsRandom(self):
        random_positions = np.random.randint(10, 240, size=(150, 2))
        valid_positions = []
        for p in random_positions:
            valid = True
            for x in valid_positions:
                if np.sqrt((p[0]-x[0])**2.0 + (p[1]-x[1])**2.0) < 15:
                    valid = False
                    break
            if valid:
                valid_positions.append(p)

        return valid_positions
    





def main():
    thorCam = ThorCam()

    print(thorCam.getImage())

if __name__ == "__main__":
    main()

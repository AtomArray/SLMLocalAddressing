import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import pyqtgraph as pg
import numpy as np
from ThorCamTest import ThorCam

        

class TrapGeography(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.thorCam = ThorCam()

        self.map = pg.ImageView()
        self.map.ui.roiBtn.hide()
        self.map.ui.menuBtn.hide()

        self.mapView = self.map.getView()

        self.mapImageData = self.thorCam.getImage()
        self.map.setImage(self.mapImageData)

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.map)

        self.button = QtWidgets.QPushButton("Next")
        self.button.clicked.connect(self.clickButton)

        self.layout.addWidget(self.button)

        self.layer = 0
        self.arrows = None
        self.justShowedArrows = False


        self.setGeometry(100, 100, 800, 800)


    def clickButton(self):
        if self.arrows:
            for a in self.arrows:
                self.map.removeItem(a)

            self.arrows = None

        if self.justShowedArrows:
            # Update highlighted circles
            self.justShowedArrows = False
            pass
        else:
            self.arrows = self.plotTrajectories(self.sorted_slmTweezers, self.trajectories, layer=self.layer)
            self.justShowedArrows = True

            self.layer += 1

    def identifySLMTweezers(self):
        blockedPixels = np.zeros(self.mapImageData.shape, dtype=np.bool)

        backgroundThreshold = 20
        blockoutRadius = 5

        viablePixels = np.where(self.mapImageData > backgroundThreshold)
        values = self.mapImageData[viablePixels]
        #print(viablePixels)
        sorted_indices = np.argsort(values)[::-1]

        slmTweezers = []

        for i in range(len(sorted_indices)):
            x, y = viablePixels[0][sorted_indices[i]], viablePixels[1][sorted_indices[i]]
            if not blockedPixels[x,y]:
                slmTweezers.append([x,y])
                blockedPixels[x-blockoutRadius:x+blockoutRadius+1, y-blockoutRadius:y+blockoutRadius+1] = True

        return np.array(slmTweezers)

    def sortSLMTweezers1(self, slmTweezers):
        # Sorting tweezers into rows from the top down

        # Begin by sorting Y pixels
        sortedIndices = np.argsort(slmTweezers[:, 1]) # Sort by Y pixel

        min_x_separation = 15

        rows = [[]]
        for i in range(len(slmTweezers)):
            j = len(rows)-1
            while j >= 0:
                row_j_available = True
                for tweezer in rows[j]:
                    if np.abs(slmTweezers[tweezer][0] - slmTweezers[sortedIndices[i]][0]) < min_x_separation:
                        row_j_available = False
                        break
                if not row_j_available:
                    break

                j -= 1

            j += 1
            # We should now add tweezer slmTweezers[sortedIndices[i]] to row j
            if j == len(rows):
                rows.append([])

            rows[j].append(sortedIndices[i])

        x_sorted_rows = []
        for row in rows:
            sorted_row = []
            x_vals = [slmTweezers[i, 0] for i in row]
            sorted_indices = np.argsort(x_vals)
            x_sorted_rows.append(np.array(row)[sorted_indices])

        
        sorted_slm_tweezers = [[slmTweezers[i] for i in sorted_row] for sorted_row in x_sorted_rows]
        return sorted_slm_tweezers


    def sortSLMTweezers2(self, slmTweezers):
        # Sorting tweezers into columns from left to right

        # Begin by sorting X pixels
        sortedIndices = np.argsort(slmTweezers[:, 0]) # Sort by X pixel

        col_width = 15

        cols = []
        lastNewRowStartX = 0
        for i in range(len(slmTweezers)):
            if len(cols) == 0:
                cols.append([])
            else:
                previousCol = cols[-1]
                xStart = slmTweezers[previousCol[0]][0]
                if slmTweezers[sortedIndices[i]][0] > xStart + col_width:
                    cols.append([])
            cols[-1].append(sortedIndices[i])

        y_sorted_cols = []
        for col in cols:
            sorted_col = []
            y_vals = [slmTweezers[i, 1] for i in col]
            sorted_indices = np.argsort(y_vals)
            y_sorted_cols.append(np.array(col)[sorted_indices])

        
        sorted_slm_tweezers = [[slmTweezers[i] for i in sorted_col] for sorted_col in y_sorted_cols]
        return sorted_slm_tweezers
            



    def labelSLMTweezers(self, slmTweezers):
        count = 0
        for c in range(len(slmTweezers)):
            
            if c % 3 == 0:
                color = (0,255,0)
            elif c % 3 == 1:
                color = (100, 100, 255)
            elif c % 3 == 2:
                color = (255, 100, 100)

            #color = (0, 255, 0)

            for i in range(len(slmTweezers[c])):
                t = slmTweezers[c][i]
                label = pg.TextItem("%d"%(count),color=color, anchor=(0.5,0.5))
                label.setPos(t[0], t[1]-5)
                self.map.addItem(label)

                count += 1

            # Mark rows
            #maxX = np.max([slmTweezers[c][i][0] for i in range(len(slmTweezers[c]))])
            #minX = np.min([slmTweezers[c][i][0] for i in range(len(slmTweezers[c]))])
            #self.map.addItem(pg.InfiniteLine(maxX, angle=90, pen=color))
            #self.map.addItem(pg.InfiniteLine(minX, angle=90, pen=color))


    def getRandomFilling(self, sorted_slmTweezers):
        num_tweezers = sum([len(r) for r in sorted_slmTweezers])

        random_filling = np.random.randint(2, size=num_tweezers)
        return random_filling

    def highlightRandomFilling(self, sorted_slmTweezers, random_filling):
        index = 0
        for r in sorted_slmTweezers:
            for i in range(len(r)):
                circle = pg.QtGui.QGraphicsEllipseItem(-7, -7, 15, 15)
                if random_filling[index]:
                    c = 'g'
                else:
                    c = 'r'
                circle.setPen(pg.mkPen(c, width=2))
                circle.setBrush(pg.mkBrush(None))
                circle.setPos(r[i][0], r[i][1])

                self.map.addItem(circle)


                index += 1

    def determineTrajectories(self, sorted_slmTweezers, random_filling):
        destinations = []

        global_index = 0
        for c in sorted_slmTweezers:
            target_indices = np.arange(len(c))
            next_target_index = 0
            destinations_for_column = []

            for i in range(len(c)):
                if random_filling[global_index]:
                    destinations_for_column.append(target_indices[next_target_index])
                    next_target_index += 1
                else:
                    destinations_for_column.append(-1)
                global_index += 1

            destinations.append(destinations_for_column)
        return destinations

    def plotTrajectories(self, sorted_slmTweezers, trajectories, layer=-1):
        arrows = []
        for c_index in range(len(sorted_slmTweezers)):
            c = sorted_slmTweezers[c_index]
            trajectories_for_column = trajectories[c_index]

            current_layer = 0
            for i in range(len(c)):
                if trajectories_for_column[i] >= 0:
                    if current_layer == layer or layer == -1:
                        source = c[i]
                        dest = c[trajectories_for_column[i]]

                        arrow = pg.PlotCurveItem([source[0], dest[0]], [source[1], dest[1]], pen=pg.mkPen('b', width=5))

                        self.map.addItem(arrow)

                        arrows.append(arrow)

                    current_layer += 1

        return arrows


def main():
    app = QtWidgets.QApplication(sys.argv)

    geography = TrapGeography()
    geography.show()

    slmTweezers = geography.identifySLMTweezers()
    sorted_slmTweezers = geography.sortSLMTweezers2(slmTweezers)
    #geography.labelSLMTweezers(sorted_slmTweezers)

    random_filling = geography.getRandomFilling(sorted_slmTweezers)
    geography.highlightRandomFilling(sorted_slmTweezers, random_filling)

    trajectories = geography.determineTrajectories(sorted_slmTweezers, random_filling)

    geography.random_filling = random_filling
    geography.trajectories = trajectories
    geography.sorted_slmTweezers = sorted_slmTweezers

    app.exec_()


if __name__ == "__main__":
    main()

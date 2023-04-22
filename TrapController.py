import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from SLMController import *
from LoadSave import LoadSavePanel
from Characters import characterDict
from network_server import NetworkServer



class Column:
    def __init__(self):
        self.xPos = 0.0

        self.yPositions = []

    def __str__(self):
        return "Column: xPos = %d\tyPositions = %s" %(self.xPos, self.yPositions)

class InputField(QtWidgets.QWidget):
    def __init__(self, description, initialVal):
        super().__init__()

        self.description = description

        label = QtWidgets.QLabel("%s:" %description)
        self.textBox = QtWidgets.QLineEdit("%s" %initialVal)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.textBox)

    def setVal(self, val):
        self.textBox.setText(str(val))
    
    def getVal(self):
        return self.textBox.text()

    def keyValuePair(self):
        return self.description, self.getVal()


class IntegerInput(InputField):
    def __init__(self, description, initialVal):
        super().__init__(description, initialVal)

        self.textBox.editingFinished.connect(self.edited)


    def edited(self):
        try:
            val = int(self.textBox.text())
            if val <= 0:
                self.textBox.setText("10")
        except:
            self.textBox.setText("10")

    def val(self):
        return int(self.textBox.text())



    def keyValuePair(self):
        return self.description, self.val()



class SubIntegerInput(InputField):
    def __init__(self, description, initialVal, subfactor=10):
        super().__init__(description, initialVal)

        self.subfactor = subfactor

        self.textBox.editingFinished.connect(self.edited)


    def edited(self):
        try:
            val = float(self.textBox.text())

            val = np.round(val * self.subfactor)
            val /= self.subfactor

            self.textBox.setText("%.2f" %val)
        except Exception as e:
            print("Error:", e)
            self.textBox.setText("10")

    def val(self):
        return float(self.textBox.text())



    def keyValuePair(self):
        return self.description, self.val()




class StringInput(InputField):
    def __init__(self, description, initialVal):
        super().__init__(description, initialVal)

        self.textBox.setMaxLength(100)#used to be 13 changed 10/20/2022
        self.textBox.editingFinished.connect(self.edited)

    def edited(self):
        # print("edited")
        try:
            text = self.textBox.text().strip()
            if not text.isupper():
                text = text.upper()
            self.textBox.setText(text)
        except:
            self.textBox.setText("ATOM")

        if text == "":
            self.textBox.setText("ATOM")

    def val(self):
        return self.textBox.text()

    def text(self):
        return self.textBox.text()

    def keyValuePair(self):
        return self.description, self.val()

class TargetPicker(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.oversamplingFactor = 10 #12/17/2022, was prev 1

        self.allInputs = [] # Subclasses should update

    def getCurrentParameters(self):
        parameters = {}
        for i in self.allInputs:
            key, value = i.keyValuePair()

            parameters[key] = value

        return parameters

    def setCurrentParameters(self, params):
        for i in self.allInputs:
            if i.description in params.keys():
                i.setVal(params[i.description])


class RectangularLatticePicker(TargetPicker):
    def __init__(self):
        super().__init__()

        self.oversamplingFactor = 1 #CHANGE THIS TO 10 TO GET 0.1 SPACING RESOLUTION

        self.oversamplingLabel = QtWidgets.QLabel("Oversampling factor: %d" %self.oversamplingFactor)

        self.numRowsInput = IntegerInput("# Rows", 5)
        self.numColsInput = IntegerInput("# Cols", 5)
        self.horizontalSpacing = SubIntegerInput("Horizontal spacing", 5, subfactor=self.oversamplingFactor)
        self.verticalSpacing = SubIntegerInput("Vertical spacing", 5, subfactor=self.oversamplingFactor)

        self.trapMaskFilename = InputField("Mask Filename", "")


        self.allInputs = [
            self.numRowsInput,
            self.numColsInput,
            self.horizontalSpacing,
            self.verticalSpacing,
            self.trapMaskFilename
        ]


        self.layout = QtWidgets.QGridLayout(self)

        self.layout.addWidget(self.oversamplingLabel, 0, 1)
        self.layout.addWidget(self.numColsInput, 1, 1)
        self.layout.addWidget(self.numRowsInput, 1, 2)

        self.layout.addWidget(self.horizontalSpacing, 2, 1)
        self.layout.addWidget(self.verticalSpacing, 2, 2)

        self.layout.addWidget(self.trapMaskFilename, 3, 1)



    def getColumns(self):
        numCols = self.numColsInput.val()
        numRows = self.numRowsInput.val()

        try:
            mask_filename = self.trapMaskFilename.getVal()

            mask = np.loadtxt("TrapArrangementMasks/%s" %mask_filename).T.astype(np.bool)
            print("Loaded mask:", mask.T.astype(np.int))

            if mask.shape[0] != numCols or mask.shape[1] != numRows:
                print("Loaded mask has wrong shape! Using default mask.")
                mask = np.ones((numCols, numRows), dtype=np.bool)
        except Exception as e:
            print("Unable to load trap mask:", e)
            mask = np.ones((numCols, numRows), dtype=np.bool)


        columns = []

        for i in range(numCols):
            c = Column()
            c.xPos = i * self.horizontalSpacing.val()

            for j in range(numRows):
                if mask[i, j]:
                    c.yPositions.append(j * self.verticalSpacing.val())

            columns.append(c)

        return columns


        
class RectangularPerimeterPicker(TargetPicker):
    def __init__(self):
        super().__init__()

        self.numRowsInput = IntegerInput("# Rows", 5)
        self.numColsInput = IntegerInput("# Cols", 5)
        self.spacing = IntegerInput("Spacing", 5)

        self.allInputs = [
            self.numRowsInput,
            self.numColsInput,
            self.spacing
        ]


        self.layout = QtWidgets.QGridLayout(self)

        self.layout.addWidget(self.numColsInput, 1, 1)
        self.layout.addWidget(self.numRowsInput, 1, 2)

        self.layout.addWidget(self.spacing, 2, 1)


    def getColumns(self):
        columns = []

        for i in range(self.numColsInput.val()):
            c = Column()


            c.xPos = i * self.spacing.val()

            if i == 0 or i == self.numColsInput.val() - 1:
                for j in range(self.numRowsInput.val()):
                    c.yPositions.append(j * self.spacing.val())

            else:
                c.yPositions.append(0)
                c.yPositions.append((self.numRowsInput.val() - 1) * self.spacing.val())

            columns.append(c)

        return columns
    

class TriangularLatticePicker(TargetPicker):
    def __init__(self):
        super().__init__()

        self.edgeLengthInput = IntegerInput("Edge length", 5)
        self.spacing = IntegerInput("Spacing", 8)

        self.allInputs = [
            self.edgeLengthInput,
            self.spacing
        ]

        self.layout = QtWidgets.QGridLayout(self)

        self.layout.addWidget(self.edgeLengthInput, 1, 1)
        self.layout.addWidget(self.spacing, 2, 1)

    def getColumns(self):

        horizontalOffset = int(self.spacing.val() / 2)
        verticalOffset = int(np.round(self.spacing.val() / 2 * np.sqrt(3)))
        latticeSpacing = self.spacing.val()

        numTrapsPerColumn = int((self.edgeLengthInput.val() + 1) // 2)

        print("Horizontal offset:", horizontalOffset)
        print("Vertical offset:", verticalOffset)
        print("Lattice spacing:", latticeSpacing)

        columns = []

        for i in range(2*self.edgeLengthInput.val()):
            c = Column()

            if i % 2 == 0:
                c.xPos = (i//2) * latticeSpacing

                for j in range(numTrapsPerColumn):
                    c.yPositions.append(j * verticalOffset * 2)
            else:
                c.xPos = horizontalOffset + (i//2) * latticeSpacing

                for j in range(numTrapsPerColumn):
                    c.yPositions.append(verticalOffset + j * 2 * verticalOffset)

            print("Column y positions:", c.yPositions)
            columns.append(c)

        return columns


class RandomArrangementPicker(TargetPicker):
    def __init__(self):
        super().__init__()


        self.numTrapsInput = IntegerInput("# Traps", 100)

        # Force atoms into a grid of columns, but with random
        # vertical position within each column
        self.confineColumnsToGridCheckbox = QtWidgets.QCheckBox("Confine columns to grid")
        self.confineColumnsToGridCheckbox.setChecked(True)
        self.confineColumnsToGridCheckbox.setEnabled(False)

        self.horizontalSpacing = IntegerInput("Horizontal spacing", 5)
        self.numColumns = IntegerInput("# Columns", 10)


        # TODO: Need to incorporate 'confine columns to grid' boolean to this
        self.allInputs = [
            self.numTrapsInput,
            self.horizontalSpacing,
            self.numColumns
        ]



        self.layout = QtWidgets.QGridLayout(self)

        self.layout.addWidget(self.numTrapsInput, 1, 1)
        self.layout.addWidget(self.confineColumnsToGridCheckbox, 2, 1)
        self.layout.addWidget(self.horizontalSpacing, 3, 1)
        self.layout.addWidget(self.numColumns, 4, 1)


    def getColumns(self):
        columns = []

        for i in range(self.numColumns.val()):
            c = Column()
            c.xPos = i * self.horizontalSpacing.val()

            columns.append(c)

        n = 0
        while n < self.numTrapsInput.val():
            y = np.random.randint(100)
            i = np.random.randint(len(columns))
            c = columns[i]

            insertionIndex = np.searchsorted(c.yPositions, y)
            minSeparation = 4

            distances = np.abs(np.array(c.yPositions) - y)
            if len(distances) >= 1 and np.min(distances) < minSeparation:
                continue # Too close to adjacent traps in the column

            n += 1
            c.yPositions.append(y)

        return columns


class AtomicaPicker(TargetPicker):
    def __init__(self):
        super().__init__()

        self.wordInput = StringInput("Word", "ATOM")
        self.horizontalSpacing = IntegerInput("Horizontal spacing", 5)
        self.verticalSpacing = IntegerInput("Vertical spacing", 5)

        self.allInputs = [
            self.wordInput,
            self.horizontalSpacing,
            self.verticalSpacing
        ]


        self.layout = QtWidgets.QGridLayout(self)

        self.layout.addWidget(self.wordInput, 1, 1)

        self.layout.addWidget(self.horizontalSpacing, 2, 1)
        self.layout.addWidget(self.verticalSpacing, 2, 2)

    def getColumns_OLD(self):
        columns = []
        CHARACTER_HEIGHT = 6
        CHARACTER_WIDTH = 5
        WORD_LENGTH = len(self.wordInput.text())

        for i in range(CHARACTER_WIDTH * WORD_LENGTH):
        # for i in range(CHARACTER_WIDTH):
            characterNum = i // CHARACTER_WIDTH # Integer division
            letter = self.wordInput.text()[characterNum]
            if letter != " ":
                characterMap = characterDict[letter]
                c = Column()

                c.xPos = (i + characterNum) * self.horizontalSpacing.val()

                for j in range(CHARACTER_HEIGHT):
                    if letter != " " and characterMap[CHARACTER_HEIGHT - j - 1][i%CHARACTER_WIDTH] != 0:
                        c.yPositions.append(j * self.verticalSpacing.val())

                columns.append(c)

        return columns


    def getColumns(self):
        columns = []
        CHARACTER_HEIGHT = 6
        CHARACTER_WIDTH = 5
        SPACE_BETWEEN_LINES = 3
        
        print(self.wordInput.text())
        LINES = (self.wordInput.text()).split("\\")
        LINES.reverse()
        print(LINES)
        # LEN_OF_LINES = np.array([len(LINE) for LINE in LINES])
        # MAX_LEN_OF_LINE = np.max(LEN_OF_LINES)

        line_counter = 0
        for LINE in LINES:
            y0 = line_counter*(CHARACTER_HEIGHT+SPACE_BETWEEN_LINES)
            WORD_LENGTH = len(LINE)
            for i in range(CHARACTER_WIDTH * WORD_LENGTH):
            # for i in range(CHARACTER_WIDTH):
                characterNum = i // CHARACTER_WIDTH # Integer division
                letter = LINE[characterNum]
                if not (letter == " " or letter == "_"):
                    characterMap = characterDict[letter]
                    c = Column()

                    c.xPos = (i + characterNum) * self.horizontalSpacing.val()

                    for j in range(CHARACTER_HEIGHT):
                        if letter != " " and characterMap[CHARACTER_HEIGHT - j - 1][i%CHARACTER_WIDTH] != 0:
                            c.yPositions.append((y0+j)* self.verticalSpacing.val())

                    columns.append(c)

            line_counter += 1

        return columns

class ArrangementPicker(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.picker = QtWidgets.QComboBox()
        self.picker.addItem("Rectangular lattice")
        self.picker.addItem("Random")
        self.picker.addItem("Rectangular perimeter")
        self.picker.addItem("Triangular lattice")
        self.picker.addItem("Atomica")
        
        self.picker.currentIndexChanged.connect(self.changedArrangementType)


        self.allPickers = [
            RectangularLatticePicker(),
            RandomArrangementPicker(),
            RectangularPerimeterPicker(),
            TriangularLatticePicker(),
            AtomicaPicker()
        ]

        for i in range(len(self.allPickers)):
            if i == 0:
                self.allPickers[i].show()
            else:
                self.allPickers[i].hide()



        self.reservoirCheckbox = QtWidgets.QCheckBox("With Reservoir")
        self.reservoirCheckbox.setChecked(False)
        self.reservoirGapInput = IntegerInput("Reservoir gap", 15)



        self.layout = QtWidgets.QVBoxLayout(self)

        self.layout.addWidget(self.picker)

        for p in self.allPickers:
            self.layout.addWidget(p)

        self.layout.addWidget(self.reservoirCheckbox)
        self.layout.addWidget(self.reservoirGapInput)

    def generateTrapPositions(self):
        currentPicker = self.allPickers[self.picker.currentIndex()]
        columns = currentPicker.getColumns()


        # Add reservoir
        if self.reservoirCheckbox.isChecked():
            maxYPos = max([max(c.yPositions) for c in columns])
            maxNumRows = max([len(c.yPositions) for c in columns])
            startingReservoirYPos = maxYPos + self.reservoirGapInput.val() # Offset from main array
            numReservoirRows = int(maxNumRows)
            reservoirSpacing = 5

            for c in columns:
                for i in range(numReservoirRows):
                    c.yPositions.append(startingReservoirYPos + i * reservoirSpacing)


        trapPositions = []
        for c in columns:
            for i in range(len(c.yPositions)):
                trapPositions.append([c.xPos, c.yPositions[i]])


        return currentPicker.oversamplingFactor, trapPositions


    def changedArrangementType(self, index):
        for i in range(len(self.allPickers)):
            if i == index:
                self.allPickers[i].show()
            else:
                self.allPickers[i].hide()


    def getCurrentArrangementParameters(self):
        curIndex = self.picker.currentIndex()
        targetParameters = self.allPickers[curIndex].getCurrentParameters()
        
        return curIndex, targetParameters


    def setCurrentArrangementParameters(self, parameters):
        index = parameters[0]
        params = parameters[1]

        self.picker.setCurrentIndex(index)
        self.allPickers[index].setCurrentParameters(params)


class TrapArrangement(QtWidgets.QWidget):

    COM_changed = QtCore.pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        
        self.setFixedWidth(600) #change to 800 on laptop, 400 on beast



        titleLabel = QtWidgets.QLabel("Trap Arrangement")
        titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        titleLabel.setFont(QtGui.QFont('Arial', 20))

        self.arrangementPicker = ArrangementPicker()
        self.generateButton = QtWidgets.QPushButton("Generate arrangement")

        self.numTrapsLabel = QtWidgets.QLabel("# Traps:")

        self.targetArrangementView = pg.ImageView()
        self.targetArrangementView.ui.roiBtn.hide()
        self.targetArrangementView.ui.menuBtn.hide()
        self.targetArrangementView.ui.histogram.hide()

        self.simulatedTrapView = pg.ImageView()
        self.simulatedTrapView.ui.roiBtn.hide()
        self.simulatedTrapView.ui.menuBtn.hide()
        self.simulatedTrapView.ui.histogram.hide()

        self.viewWidget = QtWidgets.QWidget()
        viewLayout = QtWidgets.QHBoxLayout(self.viewWidget)
        viewLayout.addWidget(self.targetArrangementView)
        #viewLayout.addWidget(self.simulatedTrapView)


        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(titleLabel)
        self.layout.addWidget(self.arrangementPicker)
        self.layout.addWidget(self.generateButton)
        self.layout.addWidget(self.viewWidget)
        self.layout.addWidget(self.numTrapsLabel)

    def generateArrangement(self):
        oversamplingFactor, positions = self.arrangementPicker.generateTrapPositions()
        trapPositions = np.array(positions)

        self.updateNumTraps(len(trapPositions))

        trapCOM = self.getTrapCOM(trapPositions)
        self.COM_changed.emit(trapCOM[0], trapCOM[1])

        maxX = np.round(max(trapPositions[:, 0]) * oversamplingFactor).astype(np.int)
        maxY = np.round(max(trapPositions[:, 1]) * oversamplingFactor).astype(np.int)


        padding = 1

        image = np.zeros((maxX + padding, maxY + padding))

        for t in trapPositions:
            image[np.round(t[0] * oversamplingFactor).astype(np.int), np.round(t[1] * oversamplingFactor).astype(np.int)] = 1

        self.targetArrangementView.setImage(image[:, ::-1])

        return oversamplingFactor, trapPositions

    def updateNumTraps(self, num):
        self.numTrapsLabel.setText("# Traps: %d" %num)

    def getTrapCOM(self, trapPositions):

        picker = self.arrangementPicker.allPickers[self.arrangementPicker.picker.currentIndex()]

        if hasattr(picker, "horizontalSpacing"):
            horizontalSpacing = picker.horizontalSpacing.val()
        elif hasattr(picker, "spacing"):
            horizontalSpacing = picker.spacing.val()
        else:
            print("Could not establish horizontal spacing, defaulting to 5!")
            horizontalSpacing = 5


        if hasattr(picker, "verticalSpacing"):
            verticalSpacing = picker.verticalSpacing.val()
        elif hasattr(picker, "spacing"):
            verticalSpacing = picker.spacing.val()
        else:
            print("Could not establish vertical spacing, defaulting to 5!")
            verticalSpacing = 5
            

        numTraps = len(trapPositions)
        COMx = 1./numTraps * np.sum(trapPositions[:,0])
        COMy = 1./numTraps * np.sum(trapPositions[:,1])

        # Round to closest grid point defined by horizontal and vertical spacings
        COM = np.array([roundTo(COMx,horizontalSpacing), roundTo(COMy, verticalSpacing)])

        return COM

    def getCurrentArrangementParameters(self):
        return self.arrangementPicker.getCurrentArrangementParameters()
    
    def setCurrentArrangementParameters(self, parameters):
        self.arrangementPicker.setCurrentArrangementParameters(parameters)

        self.generateArrangement()
        

class TrapControllerInterface(QtWidgets.QWidget):
    def __init__(self, app, shouldRunNetworkServer=True, shouldEnableSLMDisplay=True, shouldEnableThorcam=True):
        super().__init__()

        self.app = app

        self.trapArrangement = TrapArrangement()
        print("Done creating trap arrangement widget.")

        self.slmController = SLMController(app, shouldEnableSLMDisplay, shouldEnableThorcam)
        print("Done creating SLM Controller.")

        self.loadSavePanel = LoadSavePanel()
        self.loadSavePanel.setCurrentConfigurationCallback(self.getCurrentConfiguration)
        self.loadSavePanel.setRestoreConfigurationCallback(self.restoreConfiguration)

        # self.slmZernikeCoefficients = SLMZernikeCoefficients(app) ####ADDED 10/13/2022

        #self.setWidgetBackgroundColor(self.trapArrangement)
        #self.setWidgetBackgroundColor(self.slmController)

        if shouldRunNetworkServer:        
            self.start_server()
            print("Done starting network server.")


        self.trapArrangement.COM_changed.connect(self.slmController.slmSettings.setZeroOrderOffset)


        self.trapArrangement.generateButton.clicked.connect(self.generateTrapArrangement)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.trapArrangement, 0, 0, 1, 1)
        self.layout.addWidget(self.slmController, 0, 1, 2, 1)


        # Not fully implemented yet
        self.layout.addWidget(self.loadSavePanel, 1, 0, 1, 1)


        self.setGeometry(100, 100, 900, 800)


    def start_server(self):
        self.port = 2000
        self.local_ip = "192.168.10.68"
        self.network_thread = NetworkServer(self.local_ip, self.port)

        self.network_thread.data_received_signal.connect(self.process_remote_command)
        self.network_thread.start()

    def process_remote_command(self, string):
        if "ZERNIKE" in string.upper():
            self.slmController.updateZernikeFromString(string)
        if "LOCAL_CALIBRATION" in string.upper():
            print("Reciveved local calibration command") 
            self.slmController.runCalibrateWithTrap()
        if "SET_BLAZE_GRATING" in string.upper(): 
            print("Recieved local Blaze grating")
            self.slmController.setCalibrationBlazeGrating(string)
        if "LOCAL_CORNERS" in string.upper():
            print("Recieved local corners")
            self.slmController.saveLocalCorners()
        if "TRAP_CORNERS" in string.upper(): 
            self.slmController.saveTrapCorners()
        # elif not string.upper().startswith("ZERNIKE"): #if this isn't just a zernike polynomial command
        #     target_config = string.strip().split("\n")[0].split(" ")[0]
        #     print('target_config', target_config)
        #     self.loadSavePanel.setConfiguration(target_config)
        #     self.loadSavePanel.load()
        else: 
            print("Not a valid network command")

        # if "LA_CALIBRATION" in string.upper(): 



    def getCurrentConfiguration(self):
        # try:
        trapArrangementParameters = self.trapArrangement.getCurrentArrangementParameters()
        slmParameters = self.slmController.getCurrentParameters()

        return trapArrangementParameters, slmParameters
        # except:
            # print("Exception", e)
            # return None

    def restoreConfiguration(self, arrangementSettings, slmSettings):
        print("Attempting to restore configurations:")


        print("Setting arrangement parameters...")
        self.trapArrangement.setCurrentArrangementParameters(arrangementSettings)
        
        print("Done!")
        print("Setting SLM parameters...")
        self.slmController.setCurrentParameters(slmSettings)
        print("Done!")
        # print(arrangementSettings)
        # print("SLMSETTINGS", slmSettings)




    def closeEvent(self, event):
        self.slmController.close()


    def setWidgetBackgroundColor(self, widget):
        widget.setAutoFillBackground(True)
        palette = QtGui.QPalette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#dddddd"))
        widget.setPalette(palette)


    def generateTrapArrangement(self):
        oversamplingFactor, trapPositions = self.trapArrangement.generateArrangement()

        self.slmController.setTargetTrapPositions(trapPositions, oversamplingFactor)








def main():
    app = QtWidgets.QApplication(sys.argv)

    shouldRunNetworkServer = True
    shouldEnableSLMDisplay = True
    shouldEnableThorcam = False

    trapControllerInterface = TrapControllerInterface(app, shouldRunNetworkServer, shouldEnableSLMDisplay, shouldEnableThorcam)
    trapControllerInterface.show()

    app.exec_()


if __name__ == "__main__":
    main()

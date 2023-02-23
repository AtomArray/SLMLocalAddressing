import os
import sys
import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class LoadSavePanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.directory = "SavedConfigurations/"

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        self.currentConfigurationCallback = None
        self.restoreConfigurationCallback = None

        self.dropdownMenu = QtWidgets.QComboBox()
        self.refreshButton = QtWidgets.QPushButton("Refresh")
        self.loadButton = QtWidgets.QPushButton("Load")
        # self.saveButton = QtWidgets.QPushButton("Overwrite")
        self.saveAsNewButton = QtWidgets.QPushButton("Save new")

        self.layout.addWidget(self.refreshButton, 0, 2, 1, 1)
        self.layout.addWidget(self.dropdownMenu, 0, 0, 1, 2)
        self.layout.addWidget(self.loadButton, 1, 0)
        # self.layout.addWidget(self.saveButton, 1, 1)
        self.layout.addWidget(self.saveAsNewButton, 1, 2)


        self.reloadDropdownMenu()

        self.refreshButton.clicked.connect(self.reloadDropdownMenu)
        self.saveAsNewButton.clicked.connect(self.saveAsNew)
        self.loadButton.clicked.connect(self.load)


    def setCurrentConfigurationCallback(self, callback):
        self.currentConfigurationCallback = callback

    def setRestoreConfigurationCallback(self, callback):
        self.restoreConfigurationCallback = callback

    def reloadDropdownMenu(self):
        self.dropdownMenu.clear()

        available_files = os.listdir(self.directory)

        for f in available_files:
            self.dropdownMenu.addItem(f)

    def saveAsNew(self):
        if self.currentConfigurationCallback == None:
            print("ERROR: Unable to retrieve configuration to save.")
            return

        config = self.currentConfigurationCallback()
        if type(config) == type(None):
            print("ERROR: Unable to retrieve configuration to save. (2)")
            return

        arrangementSettings, slmSettings = config

        filename, ok = QtGui.QInputDialog.getText(self, "Save new configuration", "Enter filename:")
        
        if not ok:
            return

        if len(filename) == 0:
            return

        currently_available_files = os.listdir(self.directory)
        if filename in currently_available_files:
            print("Unable to save, since this file already exists.")
            sys.stdout.flush()
            return

        path = "%s/%s" %(self.directory, filename)

        np.savez(path,
            arrangementSettings=arrangementSettings,
            slmSettings=slmSettings)

        self.reloadDropdownMenu()

    def load(self):
        if self.restoreConfigurationCallback == None:
            print("ERROR: Unable to restore configuration.")
            return

        path = "%s/%s" %(self.directory, self.dropdownMenu.currentText())

        data = np.load(path, allow_pickle=True)
        arrangementSettings = data["arrangementSettings"]
        slmSettings = data["slmSettings"]

        self.restoreConfigurationCallback(arrangementSettings, slmSettings)

    def setConfiguration(self, config):
        """Find an entry in the dropdown menu by name 'config'.
        If not found, don't change anything"""
        index = self.dropdownMenu.findText("%s.npz"%config, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.dropdownMenu.setCurrentIndex(index)

        ####IF WANT TO CHANGE ZERNIKE, DO SAME THING WITH MENU BUT WITH ZERNIKE TABLE AND ADD IT

    # def readCurrentFile(self):
    #     path = "%s/%s" %(self.directory, self.dropdownMenu.currentText())

    #     data = np.loadtxt(path)
    #     if len(data.shape) == 1:
    #         data = np.array([data])

    #     # Try to read channel amplitude from file as well:
    #     channelAmplitude = 100
    #     try:
    #         with open(path) as f:
    #             first_line = f.readline()
                
    #             if "Channel Amplitude" in first_line:
    #                 channelAmplitude = int(first_line.split("=")[1].strip())
    #     except:
    #         pass
    #     return data, channelAmplitude

    # def overwriteCurrentFile(self, data, channelAmplitude):
    #     curFile = self.dropdownMenu.currentText()
    #     if len(curFile) == 0:
    #         return


    #     ok = QtGui.QMessageBox.question(self, "Are you sure?", "Are you sure you want to overwrite '%s'?" %curFile)
    #     if ok != QtGui.QMessageBox.Yes:
    #         return 

    #     path = "%s/%s" %(self.directory, curFile)

    #     np.savetxt(path, data, header="Channel Amplitude = %d" %channelAmplitude)


    # def saveTempFile(self, phases, channel):
    #     path = "tmp/tmp%d.txt"%(channel)
    #     np.savetxt(path, phases)


def main():
    app = QtWidgets.QApplication(sys.argv)

    x = LoadSavePanel()
    x.show()

    app.exec_()


if __name__ == "__main__":
    main()

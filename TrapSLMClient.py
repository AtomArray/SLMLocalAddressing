import numpy as np
import socket
import time

class TrapSLMClient:
	def __init__(self):
		HOST = "192.168.10.7"
		PORT = 1235

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		try:
			self.socket.connect((HOST, PORT))

			self.connected = True
		except:
			self.connected = False
			pass
	
	def setStaticWaveform(self, xFreq, yFreq, xAmp, yAmp):
		string = "STATIC %f %f %f %f" %(xFreq, yFreq, xAmp, yAmp)
		self.sendString(string)

	def sendString(self, string):
		if not self.connected:
			return
		self.socket.send((string + "\n").encode())


	def __del__(self):
		if self.connected:
			self.socket.close()


class AODClient_DualChannel:
	def __init__(self):
		HOST = "192.168.10.7"
		PORT = 3001

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		try:
			self.socket.connect((HOST, PORT))

			self.connected = True
		except:
			self.connected = False
			pass
		# self.setAmp(0, 0, 0.8)
		# self.setAmp(1, 0, 0.8)
		# self.setFreq(0, 0, 95.0)
		# self.setFreq(1, 0, 95.0)
		
		
		

	def setAmp(self, channel, index, amp):
		string = "SET %d %d AMPLITUDE %.3f" %(channel, index, amp)
		self.sendString(string)

	def setFreq(self, channel, index, freq):
		string = "SET %d %d FREQUENCY %.3f" %(channel, index, freq)
		self.sendString(string)

	def sendString(self, string):
		if not self.connected:
			return
		self.socket.send((string + "\n").encode())





	def __del__(self):
		if self.connected:
			self.socket.close()


def main():
	client = AODClient()
	# client.setStaticWaveform(100, 101, 1, 1)

if __name__ == "__main__":
	main()
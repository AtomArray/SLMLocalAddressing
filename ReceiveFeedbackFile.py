import socket
import sys
import numpy as np



def main():
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	server_address = ('192.168.10.7', 12345)
	sock.bind(server_address)

	sock.listen(1)

	print("Waiting for connection...")

	connection, client_address = sock.accept()

	data = connection.recv(10000)
	if data:
		data = data.decode()
		print("Received data:", data)
		print("Saving...")

		tokens = data.strip().split(" ")

		float_tokens = []
		for t in tokens:
			if len(t.strip()) == 0:
				continue

			try:
				float_tokens.append(float(t))
			except:
				pass # skip

		np.save("FeedbackFile.npy", float_tokens)
		print("Saved!")
	connection.close()
	sock.close()



if __name__ == "__main__":
	main()
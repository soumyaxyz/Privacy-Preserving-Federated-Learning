import socket
import threading

def server_function(port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific address and port
    host = 'localhost'#socket.gethostname()

    server_socket.bind((host,   port))

    # Listen for incoming connections
    server_socket.listen(5)

    print(f"Server listening on {host}:{port}")

    while True:
        # Accept a connection from a client
        client_socket, addr = server_socket.accept()
        print(f"Got connection from {addr}")

        # Receive data from the client
        data = client_socket.recv(1024).decode()
        print(f"Received data from client: {data}")

        # Close the connection
        client_socket.close()

def client_function(server_address):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect(server_address)

    # Send information to the server
    message = "Hello, server!"
    client_socket.send(message.encode())

    # Close the connection
    client_socket.close()

def process_for_information(data):
    initialization_details = None
    return initialization_details

def create_socket_and_listen(port, debug=False):
    host = socket.gethostname()
    server_socket = socket.socket()
    server_socket.bind((host, port))

    # # Create a socket object for broadcasting
    # server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # Bind the socket to a specific address and port


    # # Receive broadcast message
    # broadcast_message, _ = server_socket.recvfrom(1024)
    # print(f"Received broadcast message: {broadcast_message.decode()}")

    # configure how many client the server can listen simultaneously
      # close the connection

    if debug:
        # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        print(f"Listening for broadcast messages on {host}:{server_socket.getsockname()[1]}")
    conn, address = server_socket.accept()  # accept new connection
    if debug:
        print("Connection from: " + str(address))

    server_socket.listen(2)
    print(f"Server listening on {host}:{port}")
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))

    initialization_details = None

    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        if not data:
            # if data is not received break
            break
        print("from connected user: " + str(data))
        ack = "ACK"
        conn.send(ack.encode())  # send data to the client

        initialization_details = process_for_information(data)
        if initialization_details is not None:
            break
    

    conn.close()
    # # Close the broadcast socket
    # server_socket.close()

    return initialization_details

def main():
    
    # Use a specific port for communication
    port = 12345

    initialization_details = None
    while initialization_details is None:
        initialization_details = create_socket_and_listen(port, debug=False)

    





    

    # Check the broadcast message to decide whether to act as client or server
    if "server" in broadcast_message.decode().lower():
        # Start the server thread
        server_thread = threading.Thread(target=server_function, args=(port,))
        server_thread.start()
    else:
        # Act as a client
        server_ip = input("Enter the server IP address: ")
        client_function((server_ip, port))

if __name__ == "__main__":
    main()




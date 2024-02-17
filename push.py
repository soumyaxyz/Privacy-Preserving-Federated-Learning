# import socket

# def client_function(server_address):
#     # Create a socket object
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     # Connect to the server
#     client_socket.connect(server_address)

#     while True:
#         # Get user input
#         message = input("Enter a message (or 'exit' to quit): ")

#         # Send the message to the server
#         client_socket.send(message.encode())

#         # Check for exit condition
#         if message.lower() == 'exit':
#             break

#     # Close the connection
#     client_socket.close()

# def main():
#     # Use the same port and server IP address as the running server
#     server_ip = input("Enter the server IP address: ")

    
#     port = 2000

#     server_address = (server_ip, port)

#     # Call the client function
#     client_function(server_address)

# if __name__ == "__main__":
#     main()


import socket


def client_program():
    host = socket.gethostname()  # as both code is running on same pc
    port = 12345  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    message = input(" -> ")  # take input

    while message.lower().strip() != 'bye':
        client_socket.send(message.encode())  # send message
        data = client_socket.recv(1024).decode()  # receive response

        print('Received from server: ' + data)  # show in terminal

        message = input(" -> ")  # again take input

    client_socket.close()  # close the connection


if __name__ == '__main__':
    client_program()
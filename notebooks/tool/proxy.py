import socket, threading

def tunnel(source, destination):
    try:
        while True:
            data = source.recv(4096)
            if not data: break
            destination.sendall(data)
    except: pass
    finally:
        source.close()
        destination.close()

def handle_client(client_socket):
    try:
        request = client_socket.recv(4096).decode('utf-8', 'ignore')
        if not request or 'CONNECT' not in request:
            client_socket.close()
            return
        
        # Extract host and port
        line = request.split('\n')[0]
        url = line.split(' ')[1]
        host, port = url.split(':')
        
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote_socket.connect((host, int(port)))
        client_socket.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        
        threading.Thread(target=tunnel, args=(client_socket, remote_socket), daemon=True).start()
        tunnel(remote_socket, client_socket)
    except Exception: client_socket.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 8080))
server.listen(10)
print("Proxy is running on port 8080... (Press Ctrl+C to stop)")

while True:
    client, addr = server.accept()
    threading.Thread(target=handle_client, args=(client,), daemon=True).start()
import socket
import pickle
import argparse
import torch
from torch import nn, optim
from data_loader import get_datasets
from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        train_data, test_data = get_datasets(client_id)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
        self.model = create_model().to(device)
        self.sample_count = len(train_data)

    def local_train(self, global_params, local_epochs=2):
        self.model.load_state_dict(global_params)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                total_samples += data.size(0)

            total_loss += epoch_loss
            total_correct += epoch_correct

            epoch_train_loss = epoch_loss / len(self.train_loader.dataset)
            epoch_train_acc = 100.0 * epoch_correct / len(self.train_loader.dataset)
            print(f"Client {self.client_id} - Epoch {epoch + 1}: "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")

        avg_loss = total_loss / total_samples
        avg_accuracy = 100.0 * total_correct / total_samples
        return self.model.state_dict(), avg_loss, avg_accuracy

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('192.168.1.203', 12345))

        while True:
            try:
                command = sock.recv(5)
                if not command:
                    continue

                command = command.decode()
                if command == 'EXIT':
                    break
                elif command != 'TRAIN':
                    continue

                # Receive global model
                data_len = int.from_bytes(sock.recv(4), 'big')
                data = b''
                while len(data) < data_len:
                    packet = sock.recv(data_len - len(data))
                    if not packet:
                        break
                    data += packet

                if len(data) != data_len:
                    print("Received incomplete model parameters!")
                    continue

                global_params = pickle.loads(data)

                # Local training
                updated_params, avg_loss, avg_acc = self.local_train(global_params)

                # Send update parameters and metrics
                data = pickle.dumps({
                    'params': updated_params,
                    'loss': avg_loss,
                    'accuracy': avg_acc,
                    'sample_count': self.sample_count
                })
                sock.sendall(len(data).to_bytes(4, 'big'))
                sock.sendall(data)

            except ConnectionResetError:
                print("Connection to server lost!")
                break
            except KeyboardInterrupt:
                break

        sock.close()
        print(f"Client {self.client_id} exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    args = parser.parse_args()

    client = FederatedClient(args.client_id)
    client.run()
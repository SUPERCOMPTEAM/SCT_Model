#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

int main() {
    int sock;
    struct sockaddr_in server;
    int data[] = {2, 1, 3, 2, 4, 4, 12, 12, 7, 7};  // Пример данных
    int data_length = sizeof(data);
    float response[5];  // Ожидаем такой же размер ответа

    // Создание сокета
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        printf("Could not create socket\n");
        return 1;
    }
    printf("Socket created\n");

    server.sin_addr.s_addr = inet_addr("127.0.0.1");
    server.sin_family = AF_INET;
    server.sin_port = htons(7998);

    // Подключение к серверу
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        printf("Connection failed\n");
        return 1;
    }
    printf("Connected to server\n");

    // Отправка длины данных
    uint32_t len = htonl(data_length);
    if (send(sock, &len, sizeof(len), 0) < 0) {
        printf("Send length failed\n");
        return 1;
    }
    printf("Data length sent: %d\n", data_length);

    // Отправка данных
    if (send(sock, data, data_length, 0) < 0) {
        printf("Send data failed\n");
        return 1;
    }
    printf("Data sent\n");

    // Прием обработанных данных
    int bytes_received = recv(sock, response, sizeof(response), 0);
    if (bytes_received < 0) {
        printf("Receive failed\n");
        return 1;
    }
    printf("Received %d bytes\n", bytes_received);

    // Закрытие сокета
    close(sock);
    printf("Socket closed\n");

    // Вывод полученных данных
    printf("Received data:\n");
    for (int i = 0; i < sizeof(response)/sizeof(response[0]); i++) {
        printf("%f\n", response[i]);
    }

    return 0;
}

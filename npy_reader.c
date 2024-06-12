#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Функция для считывания double массива из .npy файла
double* read_npy_double(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return NULL;
    }

    // Пропускаем заголовок (здесь предполагается, что заголовок имеет фиксированную длину 128 байт)
    fseek(file, 128, SEEK_SET);

    // Вычисляем размер файла
    fseek(file, 0, SEEK_END);
    long filesize = ftell(file);
    fseek(file, 128, SEEK_SET); // Снова устанавливаем позицию после заголовка

    // Определяем количество элементов
    *length = (filesize - 128) / sizeof(double);

    // Выделяем память для массива
    double* array = (double*)malloc(*length * sizeof(double));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // Считываем данные
    fread(array, sizeof(double), *length, file);
    fclose(file);

    return array;
}

int main(int argc, char const *argv[])
{
    const char* filename = argv[1];
    size_t length;
    double* array = read_npy_double(filename, &length);

    if (array == NULL) {
        return 1;
    }

    printf("Array length: %zu\n", length);
    for (size_t i = 0; i < length; ++i) {
        printf("%f\n", array[i]);
    }

    free(array);
    return 0;
}

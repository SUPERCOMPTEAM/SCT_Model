#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void write_npy_int(const char* filename, int* array, size_t length) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return;
    }

    const char* magic_string = "\x93NUMPY";
    uint8_t major_version = 1;
    uint8_t minor_version = 0;
    fwrite(magic_string, 1, 6, file);
    fwrite(&major_version, 1, 1, file);
    fwrite(&minor_version, 1, 1, file);

    char header[256];
    int header_len = snprintf(header, sizeof(header),
                              "{'descr': '<i4', 'fortran_order': False, 'shape': (%zu,)}", length);

    int padding = 64 - (10 + header_len + 1) % 64;
    if (padding == 64) padding = 0;

    uint16_t header_total_len = header_len + padding + 1;
    fwrite(&header_total_len, sizeof(uint16_t), 1, file);

    char full_header[256];
    snprintf(full_header, sizeof(full_header), "%s%*c", header, padding + 1, '\n');
    fwrite(full_header, 1, header_total_len, file);

    fwrite(array, sizeof(int), length, file);
    fclose(file);
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int *array = (int *) malloc(sizeof(int) * n);
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    printf("Enter %d integers:\n", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &array[i]);
    }

    write_npy_int(filename, array, n);
    printf("Array written to %s\n", filename);
    free(array);

    return 0;
}

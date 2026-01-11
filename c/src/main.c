/**
 * Hello World WASI CLI in C
 *
 * Simple main() that prints to stdout - runs directly with wasmtime.
 */

#include <stdio.h>

int main(void) {
    printf("Hello wasm component world from C!\n");
    return 0;
}

/**
 * Hello World WASI Component in C
 *
 * Exports a greet function that returns a greeting string.
 */

#include <stdio.h>
#include "hello_c.h"  // Generated WIT bindings

// Implementation of the greeter interface
void exports_hello_c_greeter_greet(hello_c_string_t *ret) {
    const char *greeting = "Hello wasm component world from C!";

    // Print to stdout (WASI handles this automatically)
    printf("%s\n", greeting);

    // Return the greeting string using the helper function
    hello_c_string_set(ret, greeting);
}

/**
 * Hello World WASI Component in C++
 *
 * Exports a greet function that returns a greeting string.
 */

#include <cstdio>
#include "hello_cpp_cpp.h"  // Generated C++ WIT bindings

// Implementation of the greeter interface
namespace exports {
namespace hello {
namespace cpp {
namespace greeter {

wit::string Greet() {
    const char *greeting = "Hello wasm component world from C++!";

    // Print to stdout
    printf("%s\n", greeting);

    // Return the greeting string
    return wit::string::from_view(std::string_view(greeting));
}

}}}}  // namespace exports::hello::cpp::greeter

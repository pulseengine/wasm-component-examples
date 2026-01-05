// Simple Go hello world WASI component
//
// This demonstrates building a Go component using TinyGo and rules_wasm_component.
// The component exports a simple greeter interface.

package main

import "fmt"

// GreeterImpl implements the greeter interface exported by this component
type GreeterImpl struct{}

// Greet returns a greeting message
func (g *GreeterImpl) Greet() string {
	return "Hello from Go!"
}

func main() {
	// For reactor components, main() can be empty
	// The exports are handled through the WIT bindings
	fmt.Println("Hello from Go WASM component!")
}

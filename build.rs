// Workspace build script
// This script is a simple passthrough for the mpsgraph-rs build
// It allows the workspace to be built with cargo build at the root

fn main() {
    println!("Workspace build script running");
    
    // Link Metal Performance Shaders framework for both macOS and iOS
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
/// Represents the deployment platform for a graph
#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeploymentPlatform {
    /// macOS platform
    MacOS = 0,
    /// iOS platform
    IOS = 1,
    /// tvOS platform
    TVOS = 2,
    /// visionOS platform
    VisionOS = 3,
}

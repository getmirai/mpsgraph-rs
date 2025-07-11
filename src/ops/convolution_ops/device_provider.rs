use ::core::ffi::c_double;
use objc2::Encoding;
use objc2::{Encode, RefEncode};
use objc2_foundation::{NSArray, NSInteger, NSNumber, NSUInteger};

extern_protocol!(
    /// A way of extending a NSCoder to enable the setting of MTLDevice for unarchived objects
    ///
    /// When a object is initialized by a NSCoder, it calls -initWithCoder:, which is
    /// missing the necessary MTLDevice to correctly initialize the MPSKernel, or MPSNNGraph.
    /// If the coder does not conform to MPSDeviceProvider, the system default device
    /// will be used.  If you would like to specify which device to use, subclass the
    /// NSCoder (NSKeyedUnarchiver, etc.) to conform to MPSDeviceProvider so that
    /// the device can be gotten from the NSCoder.
    ///
    /// See MPSKeyedUnarchiver for one implementation of this protocol. It reads files
    /// prepared with the NSKeyedArchiver and allows you to set the MTLDevice that the
    /// unarchived objects use.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsdeviceprovider?language=objc)
    pub unsafe trait MPSDeviceProvider {
        /// Return the device to use when making MPSKernel subclasses from the NSCoder
        #[unsafe(method(mpsMTLDevice))]
        #[unsafe(method_family = none)]
        unsafe fn mpsMTLDevice(&self) -> Option<Retained<ProtocolObject<dyn MTLDevice>>>;
    }
);

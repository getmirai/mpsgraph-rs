<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Writing Modern NSObject Subclasses in Rust with objc2 and objc2_foundation

This guide provides a comprehensive approach to creating Objective-C-compatible classes in Rust using the `objc2` and `objc2_foundation` crates. We'll cover class definition, memory management, thread safety, and framework integration.

---

## 1. Project Setup

### Dependencies

Add these to your `Cargo.toml`:

```toml
[dependencies]
objc2 = "0.3"
objc2-foundation = "0.3"
objc2-app-kit = "0.3"  # For AppKit integration
```


### Basic Structure

```rust
#![deny(unsafe_op_in_unsafe_fn)]
use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{define_class, sel, msg_send};
```

---

## 2. Class Definition

### Basic NSObject Subclass

```rust
define_class!(
    #[unsafe(super = NSObject)]
    #[name = "RustDataModel"]
    #[ivars = ModelIvars]
    struct RustDataModel;

    unsafe impl NSObjectProtocol for RustDataModel {}
);
```


### Instance Variables (ivars)

```rust
#[derive(Default)]
struct ModelIvars {
    counter: AtomicI32,
    data: Mutex&lt;Vec&lt;u8&gt;&gt;,
}
```

---

## 3. Method Implementation

### Initialization Pattern

```rust
extern_methods! {
    unsafe impl RustDataModel {
        #[method(initWithCount:)]
        fn init_with_count(this: *mut Self, count: i32) -&gt; Option&lt;Retained&lt;Self&gt;&gt; {
            let this = unsafe { msg_send![super(this), init] };
            this.map(|mut obj| {
                obj.ivars().counter.store(count, Ordering::Relaxed);
                obj
            })
        }
    }
}
```


### Property Accessors

```rust
#[method(count)]
fn count(&amp;self) -&gt; i32 {
    self.ivars().counter.load(Ordering::Relaxed)
}

#[method(setCount:)]
fn set_count(&amp;self, new_value: i32) {
    self.ivars().counter.store(new_value, Ordering::Relaxed);
}
```

---

## 4. Memory Management

### Ownership Patterns

```rust
fn create_instance() -&gt; Retained&lt;RustDataModel&gt; {
    let cls = RustDataModel::class();
    unsafe { msg_send![cls, new] }
}

fn use_instance(obj: &amp;RustDataModel) {
    let count = unsafe { msg_send![obj, count] };
    println!("Current count: {}", count);
}
```


### Autorelease Pools

```rust
use objc2::rc::autoreleasepool;

autoreleasepool(|_| {
    let temp_obj = RustDataModel::new();
    // Temporary object automatically released here
});
```

---

## 5. Thread Safety

### Thread Confinement

```rust
define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[name = "MainThreadModel"]
    struct MainThreadModel {
        ui_elements: MainThreadCell&lt;Vec&lt;Retained&lt;NSObject&gt;&gt;&gt;,
    }
);
```


### Concurrent Access

```rust
#[derive(Default)]
struct ThreadSafeIvars {
    cache: RwLock&lt;HashMap&lt;String, Retained&lt;NSData&gt;&gt;&gt;,
}

#[method(cacheValueForKey:)]
fn cache_value(&amp;self, key: &amp;NSString) -&gt; Option&lt;Retained&lt;NSData&gt;&gt; {
    self.ivars().cache.read().unwrap().get(key).cloned()
}
```

---

## 6. Framework Integration

### AppKit Delegate

```rust
define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[name = "AppDelegate"]
    #[ivars = AppDelegateIvars]
    struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}
    unsafe impl NSApplicationDelegate for AppDelegate {
        #[method(applicationDidFinishLaunching:)]
        fn did_finish_launching(&amp;self, notification: &amp;NSNotification) {
            // App initialization logic
        }
    }
);
```


### Metal Performance Shaders

```rust
extern_class!(
    #[unsafe(super(NSObject))]
    pub struct MPSGraph;
);

define_class!(
    #[unsafe(super = MPSGraph)]
    #[name = "CustomGraph"]
    struct CustomGraph {
        pipeline: Mutex&lt;Option&lt;Retained&lt;MTLComputePipelineState&gt;&gt;&gt;,
    }

    unsafe impl NSObjectProtocol for CustomGraph {}
    
    extern_methods! {
        #[method(encodeToCommandBuffer:)]
        fn encode(&amp;self, buffer: &amp;MTLCommandBuffer) {
            let guard = self.ivars().pipeline.lock().unwrap();
            // Encoding logic
        }
    }
);
```

---

## 7. Best Practices

### Type Safety

- Prefer concrete `NSObject` subclasses over `AnyObject`
- Use `Retained&lt;T&gt;` for ownership tracking
- Leverage Rust's type system for compile-time checks


### Error Handling

```rust
#[method(performOperationWithError:_)]
fn perform_operation(&amp;self) -&gt; Result&lt;(), Retained&lt;NSError&gt;&gt; {
    // Implementation
    Ok(())
}
```


### Protocol Conformance

```rust
unsafe impl NSCopying for RustDataModel {
    type Output = Retained&lt;Self&gt;;
    
    fn copy(&amp;self) -&gt; Retained&lt;Self&gt; {
        // Copy implementation
    }
}
```

---

## 8. Complete Example: Custom UI Component

```rust
define_class!(
    #[unsafe(super = NSView)]
    #[thread_kind = MainThreadOnly]
    #[name = "CustomView"]
    #[ivars = ViewIvars]
    struct CustomView;

    unsafe impl NSObjectProtocol for CustomView {}
    
    extern_methods! {
        #[method(drawRect:)]
        fn draw_rect(&amp;self, rect: NSRect) {
            let color = NSColor::systemRedColor();
            color.setFill();
            unsafe { NSRectFill(rect) };
        }
    }
);

impl CustomView {
    fn new(mtm: MainThreadMarker) -&gt; Retained&lt;Self&gt; {
        let view: Retained&lt;Self&gt; = unsafe { msg_send![Self::class(), new] };
        view.setWantsLayer(true);
        view
    }
}
```

---

## Key Considerations

1. **Lifecycle Management**:
    - Use `Retained&lt;T&gt;` for owned object references
    - Implement `NSCopying` and `NSMutableCopying` when needed
    - Handle autorelease pools appropriately
2. **Thread Affinity**:
    - Use `MainThreadOnly` for UI components
    - `AnyThread` for compute-intensive classes
    - Combine with Rust's synchronization primitives
3. **Performance**:
    - Minimize Objective-C message sends in hot paths
    - Use atomic operations for simple state
    - Prefer Rust-side data processing when possible
4. **Interoperability**:
    - Use `ns_string!` macro for string conversions
    - Implement standard protocols (`NSCoding`, `NSSecureCoding`)
    - Handle framework-specific memory management rules

This guide provides the foundation for building robust Objective-C compatible classes in Rust. Always consult framework-specific documentation and leverage Rust's type system to ensure memory safety and thread correctness.

<div>⁂</div>

[^1]: https://docs.rs/objc2/

[^2]: https://stackoverflow.com/questions/78715945/calling-nsapplication-from-rust-using-objc

[^3]: https://www.reddit.com/r/rust/comments/sm2ztn/inheritance_in_rust/

[^4]: https://www.macscripter.net/t/how-to-create-an-objective-c-subclass/45121

[^5]: https://kylewlacy.github.io/posts/cocoa-apps-in-rust-eventually/

[^6]: https://www.objc.io/issues/13-architecture/subclassing

[^7]: https://en.wikipedia.org/wiki/Objective-C

[^8]: https://docs.rs/objc2/latest/objc2/runtime/type.Object.html

[^9]: https://rubicon-objc.readthedocs.io/en/stable/tutorial/tutorial-2.html

[^10]: https://stackoverflow.com/questions/24997536/runtime-error-when-using-corefoundation-objects-in-a-swift-nsobject-subclass

[^11]: https://lib.rs/crates/objc2-foundation

[^12]: https://learn.microsoft.com/en-us/dotnet/api/foundation.nsobject?view=xamarin-ios-sdk-12

[^13]: https://stackoverflow.com/questions/19358079/importing-foundation-h-but-using-nsobject

[^14]: https://stackoverflow.com/questions/32490684/initialize-subclass-of-nsobject

[^15]: https://developer.apple.com/documentation/objectivec/nsobject

[^16]: https://crates.io/crates/objc2/0.3.0-beta.4

[^17]: https://docs.rs/objc2/latest/objc2/runtime/trait.NSObjectProtocol.html

[^18]: https://crates.io/crates/objc2/0.2.7

[^19]: https://www.reddit.com/r/swift/comments/17v5ama/guide_to_nsobject/

[^20]: https://lib.rs/crates/objc2

[^21]: https://github.com/SSheldon/rust-objc/blob/master/examples/example.rs

[^22]: https://lib.rs/crates/objc2-foundation

[^23]: https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjectiveC/Chapters/ocObjectsClasses.html

[^24]: https://www.thecodedmessage.com/posts/oop-3-inheritance/

[^25]: https://stackoverflow.com/questions/32490684/initialize-subclass-of-nsobject

[^26]: https://github.com/madsmtm/objc2

[^27]: https://github.com/rust-windowing/winit/issues/4015

[^28]: https://github.com/madsmtm/objc2/issues/30

[^29]: https://www.hackingwithswift.com/read/10/5/custom-subclasses-of-nsobject

[^30]: https://crates.io/crates/objrs

[^31]: https://www.objc.io/issues/13-architecture/subclassing

[^32]: http://sasheldon.com/rust-objc/objc/declare/index.html

[^33]: https://crates.io/crates/objc

[^34]: https://stackoverflow.com/questions/39943371/swift-3-subclassing-nsobject-or-not

[^35]: https://crates.io/crates/objc2/0.3.0-beta.3.patch-leaks.3


//! In-process Parakeet transcription through parakeet.cpp's stable C API.
//!
//! The native library owns model loading and inference. Voxtype passes its
//! captured mono 16 kHz samples directly; this module does not start a server,
//! write a temporary WAV, or spawn an adapter process.

use crate::error::TranscribeError;
use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::Mutex;

type AbiVersionFn = unsafe extern "C" fn() -> c_int;
type LoadFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type LoadErrorFn = unsafe extern "C" fn() -> *const c_char;
type LastErrorFn = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type TranscribePcmFn =
    unsafe extern "C" fn(*mut c_void, *const f32, c_int, c_int, c_int) -> *mut c_char;
type FreeContextFn = unsafe extern "C" fn(*mut c_void);
type FreeStringFn = unsafe extern "C" fn(*mut c_char);

const REQUIRED_ABI: c_int = 10;

/// Loaded model and native library. A mutex serializes inference through the
/// single model context and makes the transcriber safe to share with Voxtype.
pub struct ParakeetCppTranscriber {
    runtime: Mutex<Runtime>,
}

struct Runtime {
    // Keep the shared object loaded until after its context has been freed.
    _library: Library,
    context: *mut c_void,
    transcribe_pcm: TranscribePcmFn,
    last_error: LastErrorFn,
    free_context: FreeContextFn,
    free_string: FreeStringFn,
}

// `context` is owned by Runtime, and every call that uses it is protected by
// ParakeetCppTranscriber::runtime's mutex. parakeet.cpp documents this model
// context as reusable for repeated transcription calls.
unsafe impl Send for Runtime {}

impl Drop for Runtime {
    fn drop(&mut self) {
        if !self.context.is_null() {
            // SAFETY: context came from this library's load function, and the
            // library is still alive in `_library` while Drop is running.
            unsafe { (self.free_context)(self.context) };
            self.context = std::ptr::null_mut();
        }
    }
}

impl ParakeetCppTranscriber {
    pub fn new(model_path: &Path) -> Result<Self, TranscribeError> {
        if !model_path.is_file() {
            return Err(TranscribeError::ModelNotFound(format!(
                "Parakeet GGUF model file not found: {}",
                model_path.display()
            )));
        }

        let library = load_library()?;
        // SAFETY: each symbol is loaded from the same library and its type is
        // copied from the published parakeet_capi.h ABI declarations.
        let abi_version = unsafe { library.get::<AbiVersionFn>(b"parakeet_capi_abi_version\0") }
            .map(|symbol| *symbol)
            .map_err(|e| {
                init_error(format!(
                    "libparakeet.so is missing its ABI version symbol: {e}"
                ))
            })?;
        let load = unsafe { library.get::<LoadFn>(b"parakeet_capi_load\0") }
            .map(|symbol| *symbol)
            .map_err(|e| {
                init_error(format!("libparakeet.so is missing parakeet_capi_load: {e}"))
            })?;
        let load_error = unsafe { library.get::<LoadErrorFn>(b"parakeet_capi_load_error\0") }
            .map(|symbol| *symbol)
            .map_err(|e| {
                init_error(format!(
                    "libparakeet.so is missing parakeet_capi_load_error: {e}"
                ))
            })?;
        let transcribe_pcm =
            unsafe { library.get::<TranscribePcmFn>(b"parakeet_capi_transcribe_pcm\0") }
                .map(|symbol| *symbol)
                .map_err(|e| {
                    init_error(format!(
                        "libparakeet.so is missing parakeet_capi_transcribe_pcm: {e}"
                    ))
                })?;
        let last_error = unsafe { library.get::<LastErrorFn>(b"parakeet_capi_last_error\0") }
            .map(|symbol| *symbol)
            .map_err(|e| {
                init_error(format!(
                    "libparakeet.so is missing parakeet_capi_last_error: {e}"
                ))
            })?;
        let free_context = unsafe { library.get::<FreeContextFn>(b"parakeet_capi_free\0") }
            .map(|symbol| *symbol)
            .map_err(|e| {
                init_error(format!("libparakeet.so is missing parakeet_capi_free: {e}"))
            })?;
        let free_string = unsafe { library.get::<FreeStringFn>(b"parakeet_capi_free_string\0") }
            .map(|symbol| *symbol)
            .map_err(|e| {
                init_error(format!(
                    "libparakeet.so is missing parakeet_capi_free_string: {e}"
                ))
            })?;

        let version = unsafe { abi_version() };
        if version < REQUIRED_ABI {
            return Err(init_error(format!(
                "libparakeet.so ABI {version} is too old; Voxtype requires ABI {REQUIRED_ABI} or newer"
            )));
        }

        let path = CString::new(model_path.to_string_lossy().as_bytes()).map_err(|_| {
            init_error("Parakeet model path contains an embedded NUL byte".to_string())
        })?;
        let context = unsafe { load(path.as_ptr()) };
        if context.is_null() {
            let reason = unsafe { c_string(load_error()) };
            return Err(init_error(format!(
                "could not load Parakeet GGUF model {}: {reason}",
                model_path.display()
            )));
        }

        Ok(Self {
            runtime: Mutex::new(Runtime {
                _library: library,
                context,
                transcribe_pcm,
                last_error,
                free_context,
                free_string,
            }),
        })
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }
        let sample_count = c_int::try_from(samples.len()).map_err(|_| {
            TranscribeError::AudioFormat("Audio buffer exceeds the native API limit".to_string())
        })?;
        let runtime = self.runtime.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to lock Parakeet C API: {e}"))
        })?;

        // SAFETY: samples points to `sample_count` valid f32 values for this
        // synchronous call; Voxtype audio is mono, 16 kHz; the context remains
        // locked and alive for the duration of the call.
        let output = unsafe {
            (runtime.transcribe_pcm)(runtime.context, samples.as_ptr(), sample_count, 16000, 0)
        };
        if output.is_null() {
            let reason = unsafe { c_string((runtime.last_error)(runtime.context)) };
            return Err(TranscribeError::InferenceFailed(format!(
                "Parakeet Redux inference failed: {reason}"
            )));
        }

        // SAFETY: the C API returns a NUL-terminated UTF-8 string allocated by
        // its own allocator. Copy it before freeing through the matching API.
        let text = unsafe { CStr::from_ptr(output) }
            .to_string_lossy()
            .trim()
            .to_string();
        unsafe { (runtime.free_string)(output) };
        Ok(text)
    }
}

fn load_library() -> Result<Library, TranscribeError> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("VOXTYPE_PARAKEET_CPP_LIBRARY") {
        candidates.push(path.into());
    }
    candidates.push(
        dirs::home_dir()
            .unwrap_or_default()
            .join(".local/lib/voxtype-parakeet/libparakeet.so"),
    );
    candidates.push("libparakeet.so".into());
    candidates.push("/usr/local/lib/libparakeet.so".into());
    candidates.push("/usr/lib/libparakeet.so".into());

    let mut errors = Vec::new();
    for path in candidates {
        // SAFETY: libparakeet is a native shared library with the stable C API;
        // the handle is retained by Runtime for as long as its function pointers
        // and context are in use.
        match unsafe { Library::new(&path) } {
            Ok(library) => return Ok(library),
            Err(error) => errors.push(format!("{}: {error}", path.display())),
        }
    }

    Err(init_error(format!(
        "could not load libparakeet.so (tried {}); install the shared parakeet.cpp library or set VOXTYPE_PARAKEET_CPP_LIBRARY",
        errors.join("; ")
    )))
}

unsafe fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "unknown native error".to_string();
    }
    // SAFETY: C API error strings are NUL-terminated and valid for the current
    // thread until the next API operation.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn init_error(message: String) -> TranscribeError {
    TranscribeError::InitFailed(message)
}

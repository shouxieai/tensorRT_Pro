#pragma once
#ifndef __UNION_PYBIND11_HPP
#define __UNION_PYBIND11_HPP

//========= begin of #include "detail/common.h" ============

/*
    pybind11/detail/common.h -- Basic macros

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#define PYBIND11_VERSION_MAJOR 2
#define PYBIND11_VERSION_MINOR 6
#define PYBIND11_VERSION_PATCH dev0

#define PYBIND11_NAMESPACE_BEGIN(name) namespace name {
#define PYBIND11_NAMESPACE_END(name) }

#if defined(__INTEL_COMPILER)
#  pragma warning push
#  pragma warning disable 68    // integer conversion resulted in a change of sign
#  pragma warning disable 186   // pointless comparison of unsigned integer with zero
#  pragma warning disable 878   // incompatible exception specifications
#  pragma warning disable 1334  // the "template" keyword used for syntactic disambiguation may only be used within a template
#  pragma warning disable 1682  // implicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem)
#  pragma warning disable 1786  // function "strdup" was declared deprecated
#  pragma warning disable 1875  // offsetof applied to non-POD (Plain Old Data) types is nonstandard
#  pragma warning disable 2196  // warning #2196: routine is both "inline" and "noinline"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4512) // warning C4512: Assignment operator was implicitly defined as deleted
#  pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#  pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#  pragma warning(disable: 4702) // warning C4702: unreachable code
#  pragma warning(disable: 4522) // warning C4522: multiple assignment operators specified
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#  pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#  pragma GCC diagnostic ignored "-Wattributes"
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wnoexcept-type"
#  endif
#endif

// Robust support for some features and loading modules compiled against different pybind versions
// requires forcing hidden visibility on pybind code, so we enforce this by setting the attribute on
// the main `pybind11` namespace.
#if !defined(PYBIND11_NAMESPACE)
#  ifdef __GNUG__
#    define PYBIND11_NAMESPACE pybind11 __attribute__((visibility("hidden")))
#  else
#    define PYBIND11_NAMESPACE pybind11
#  endif
#endif

#if !(defined(_MSC_VER) && __cplusplus == 199711L) && !defined(__INTEL_COMPILER)
#  if __cplusplus >= 201402L
#    define PYBIND11_CPP14
#    if __cplusplus >= 201703L
#      define PYBIND11_CPP17
#    endif
#  endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
// MSVC sets _MSVC_LANG rather than __cplusplus (supposedly until the standard is fully implemented)
// Unless you use the /Zc:__cplusplus flag on Visual Studio 2017 15.7 Preview 3 or newer
#  if _MSVC_LANG >= 201402L
#    define PYBIND11_CPP14
#    if _MSVC_LANG > 201402L && _MSC_VER >= 1910
#      define PYBIND11_CPP17
#    endif
#  endif
#endif

// Compiler version assertions
#if defined(__INTEL_COMPILER)
#  if __INTEL_COMPILER < 1700
#    error pybind11 requires Intel C++ compiler v17 or newer
#  endif
#elif defined(__clang__) && !defined(__apple_build_version__)
#  if __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 3)
#    error pybind11 requires clang 3.3 or newer
#  endif
#elif defined(__clang__)
// Apple changes clang version macros to its Xcode version; the first Xcode release based on
// (upstream) clang 3.3 was Xcode 5:
#  if __clang_major__ < 5
#    error pybind11 requires Xcode/clang 5.0 or newer
#  endif
#elif defined(__GNUG__)
#  if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
#    error pybind11 requires gcc 4.8 or newer
#  endif
#elif defined(_MSC_VER)
// Pybind hits various compiler bugs in 2015u2 and earlier, and also makes use of some stl features
// (e.g. std::negation) added in 2015u3:
#  if _MSC_FULL_VER < 190024210
#    error pybind11 requires MSVC 2015 update 3 or newer
#  endif
#endif

#if !defined(PYBIND11_EXPORT)
#  if defined(WIN32) || defined(_WIN32)
#    define PYBIND11_EXPORT __declspec(dllexport)
#  else
#    define PYBIND11_EXPORT __attribute__ ((visibility("default")))
#  endif
#endif

#if defined(_MSC_VER)
#  define PYBIND11_NOINLINE __declspec(noinline)
#else
#  define PYBIND11_NOINLINE __attribute__ ((noinline))
#endif

#if defined(PYBIND11_CPP14)
#  define PYBIND11_DEPRECATED(reason) [[deprecated(reason)]]
#else
#  define PYBIND11_DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif

#if defined(PYBIND11_CPP17)
#  define PYBIND11_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#  define PYBIND11_MAYBE_UNUSED
#else
#  define PYBIND11_MAYBE_UNUSED __attribute__ ((__unused__))
#endif

/* Don't let Python.h #define (v)snprintf as macro because they are implemented
   properly in Visual Studio since 2015. */
#if defined(_MSC_VER) && _MSC_VER >= 1900
#  define HAVE_SNPRINTF 1
#endif

/// Include Python header, disable linking to pythonX_d.lib on Windows in debug mode
#if defined(_MSC_VER)
#  if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 4)
#    define HAVE_ROUND 1
#  endif
#  pragma warning(push)
#  pragma warning(disable: 4510 4610 4512 4005)
#  if defined(_DEBUG) && !defined(Py_DEBUG)
#    define PYBIND11_DEBUG_MARKER
#    undef _DEBUG
#  endif
#endif

#include <Python.h>
#include <frameobject.h>
#include <pythread.h>

/* Python #defines overrides on all sorts of core functions, which
   tends to weak havok in C++ codebases that expect these to work
   like regular functions (potentially with several overloads) */
#if defined(isalnum)
#  undef isalnum
#  undef isalpha
#  undef islower
#  undef isspace
#  undef isupper
#  undef tolower
#  undef toupper
#endif

#if defined(copysign)
#  undef copysign
#endif

#if defined(_MSC_VER)
#  if defined(PYBIND11_DEBUG_MARKER)
#    define _DEBUG
#    undef PYBIND11_DEBUG_MARKER
#  endif
#  pragma warning(pop)
#endif

#include <cstddef>
#include <cstring>
#include <forward_list>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include <type_traits>

#if PY_MAJOR_VERSION >= 3 /// Compatibility macros for various Python versions
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyInstanceMethod_New(ptr)
#define PYBIND11_INSTANCE_METHOD_CHECK PyInstanceMethod_Check
#define PYBIND11_INSTANCE_METHOD_GET_FUNCTION PyInstanceMethod_GET_FUNCTION
#define PYBIND11_BYTES_CHECK PyBytes_Check
#define PYBIND11_BYTES_FROM_STRING PyBytes_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyBytes_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyBytes_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyBytes_AsString
#define PYBIND11_BYTES_SIZE PyBytes_Size
#define PYBIND11_LONG_CHECK(o) PyLong_Check(o)
#define PYBIND11_LONG_AS_LONGLONG(o) PyLong_AsLongLong(o)
#define PYBIND11_LONG_FROM_SIGNED(o) PyLong_FromSsize_t((ssize_t) o)
#define PYBIND11_LONG_FROM_UNSIGNED(o) PyLong_FromSize_t((size_t) o)
#define PYBIND11_BYTES_NAME "bytes"
#define PYBIND11_STRING_NAME "str"
#define PYBIND11_SLICE_OBJECT PyObject
#define PYBIND11_FROM_STRING PyUnicode_FromString
#define PYBIND11_STR_TYPE ::pybind11::str
#define PYBIND11_BOOL_ATTR "__bool__"
#define PYBIND11_NB_BOOL(ptr) ((ptr)->nb_bool)
// Providing a separate declaration to make Clang's -Wmissing-prototypes happy.
// See comment for PYBIND11_MODULE below for why this is marked "maybe unused".
#define PYBIND11_PLUGIN_IMPL(name) \
    extern "C" PYBIND11_MAYBE_UNUSED PYBIND11_EXPORT PyObject *PyInit_##name(); \
    extern "C" PYBIND11_EXPORT PyObject *PyInit_##name()

#else
#define PYBIND11_INSTANCE_METHOD_NEW(ptr, class_) PyMethod_New(ptr, nullptr, class_)
#define PYBIND11_INSTANCE_METHOD_CHECK PyMethod_Check
#define PYBIND11_INSTANCE_METHOD_GET_FUNCTION PyMethod_GET_FUNCTION
#define PYBIND11_BYTES_CHECK PyString_Check
#define PYBIND11_BYTES_FROM_STRING PyString_FromString
#define PYBIND11_BYTES_FROM_STRING_AND_SIZE PyString_FromStringAndSize
#define PYBIND11_BYTES_AS_STRING_AND_SIZE PyString_AsStringAndSize
#define PYBIND11_BYTES_AS_STRING PyString_AsString
#define PYBIND11_BYTES_SIZE PyString_Size
#define PYBIND11_LONG_CHECK(o) (PyInt_Check(o) || PyLong_Check(o))
#define PYBIND11_LONG_AS_LONGLONG(o) (PyInt_Check(o) ? (long long) PyLong_AsLong(o) : PyLong_AsLongLong(o))
#define PYBIND11_LONG_FROM_SIGNED(o) PyInt_FromSsize_t((ssize_t) o) // Returns long if needed.
#define PYBIND11_LONG_FROM_UNSIGNED(o) PyInt_FromSize_t((size_t) o) // Returns long if needed.
#define PYBIND11_BYTES_NAME "str"
#define PYBIND11_STRING_NAME "unicode"
#define PYBIND11_SLICE_OBJECT PySliceObject
#define PYBIND11_FROM_STRING PyString_FromString
#define PYBIND11_STR_TYPE ::pybind11::bytes
#define PYBIND11_BOOL_ATTR "__nonzero__"
#define PYBIND11_NB_BOOL(ptr) ((ptr)->nb_nonzero)
// Providing a separate PyInit decl to make Clang's -Wmissing-prototypes happy.
// See comment for PYBIND11_MODULE below for why this is marked "maybe unused".
#define PYBIND11_PLUGIN_IMPL(name) \
    static PyObject *pybind11_init_wrapper();                           \
    extern "C" PYBIND11_MAYBE_UNUSED PYBIND11_EXPORT void init##name(); \
    extern "C" PYBIND11_EXPORT void init##name() {                      \
        (void)pybind11_init_wrapper();                                  \
    }                                                                   \
    PyObject *pybind11_init_wrapper()
#endif

#if PY_VERSION_HEX >= 0x03050000 && PY_VERSION_HEX < 0x03050200
extern "C" {
    struct _Py_atomic_address { void *value; };
    PyAPI_DATA(_Py_atomic_address) _PyThreadState_Current;
}
#endif

#define PYBIND11_TRY_NEXT_OVERLOAD ((PyObject *) 1) // special failure return code
#define PYBIND11_STRINGIFY(x) #x
#define PYBIND11_TOSTRING(x) PYBIND11_STRINGIFY(x)
#define PYBIND11_CONCAT(first, second) first##second
#define PYBIND11_ENSURE_INTERNALS_READY \
    pybind11::detail::get_internals();

#define PYBIND11_CHECK_PYTHON_VERSION \
    {                                                                          \
        const char *compiled_ver = PYBIND11_TOSTRING(PY_MAJOR_VERSION)         \
            "." PYBIND11_TOSTRING(PY_MINOR_VERSION);                           \
        const char *runtime_ver = Py_GetVersion();                             \
        size_t len = std::strlen(compiled_ver);                                \
        if (std::strncmp(runtime_ver, compiled_ver, len) != 0                  \
                || (runtime_ver[len] >= '0' && runtime_ver[len] <= '9')) {     \
            PyErr_Format(PyExc_ImportError,                                    \
                "Python version mismatch: module was compiled for Python %s, " \
                "but the interpreter version is incompatible: %s.",            \
                compiled_ver, runtime_ver);                                    \
            return nullptr;                                                    \
        }                                                                      \
    }

#define PYBIND11_CATCH_INIT_EXCEPTIONS \
        catch (pybind11::error_already_set &e) {                               \
            PyErr_SetString(PyExc_ImportError, e.what());                      \
            return nullptr;                                                    \
        } catch (const std::exception &e) {                                    \
            PyErr_SetString(PyExc_ImportError, e.what());                      \
            return nullptr;                                                    \
        }                                                                      \

/** \rst
    ***Deprecated in favor of PYBIND11_MODULE***

    This macro creates the entry point that will be invoked when the Python interpreter
    imports a plugin library. Please create a `module` in the function body and return
    the pointer to its underlying Python object at the end.

    .. code-block:: cpp

        PYBIND11_PLUGIN(example) {
            pybind11::module m("example", "pybind11 example plugin");
            /// Set up bindings here
            return m.ptr();
        }
\endrst */
#define PYBIND11_PLUGIN(name)                                                  \
    PYBIND11_DEPRECATED("PYBIND11_PLUGIN is deprecated, use PYBIND11_MODULE")  \
    static PyObject *pybind11_init();                                          \
    PYBIND11_PLUGIN_IMPL(name) {                                               \
        PYBIND11_CHECK_PYTHON_VERSION                                          \
        PYBIND11_ENSURE_INTERNALS_READY                                        \
        try {                                                                  \
            return pybind11_init();                                            \
        } PYBIND11_CATCH_INIT_EXCEPTIONS                                       \
    }                                                                          \
    PyObject *pybind11_init()

/** \rst
    This macro creates the entry point that will be invoked when the Python interpreter
    imports an extension module. The module name is given as the fist argument and it
    should not be in quotes. The second macro argument defines a variable of type
    `py::module` which can be used to initialize the module.

    The entry point is marked as "maybe unused" to aid dead-code detection analysis:
    since the entry point is typically only looked up at runtime and not referenced
    during translation, it would otherwise appear as unused ("dead") code.

    .. code-block:: cpp

        PYBIND11_MODULE(example, m) {
            m.doc() = "pybind11 example module";

            // Add bindings here
            m.def("foo", []() {
                return "Hello, World!";
            });
        }
\endrst */
#define PYBIND11_MODULE(name, variable)                                        \
    PYBIND11_MAYBE_UNUSED                                                      \
    static void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &);     \
    PYBIND11_PLUGIN_IMPL(name) {                                               \
        PYBIND11_CHECK_PYTHON_VERSION                                          \
        PYBIND11_ENSURE_INTERNALS_READY                                        \
        auto m = pybind11::module(PYBIND11_TOSTRING(name));                    \
        try {                                                                  \
            PYBIND11_CONCAT(pybind11_init_, name)(m);                          \
            return m.ptr();                                                    \
        } PYBIND11_CATCH_INIT_EXCEPTIONS                                       \
    }                                                                          \
    void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &variable)


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

using ssize_t = Py_ssize_t;
using size_t  = std::size_t;

/// Approach used to cast a previously unknown C++ instance into a Python object
enum class return_value_policy : uint8_t {
    /** This is the default return value policy, which falls back to the policy
        return_value_policy::take_ownership when the return value is a pointer.
        Otherwise, it uses return_value::move or return_value::copy for rvalue
        and lvalue references, respectively. See below for a description of what
        all of these different policies do. */
    automatic = 0,

    /** As above, but use policy return_value_policy::reference when the return
        value is a pointer. This is the default conversion policy for function
        arguments when calling Python functions manually from C++ code (i.e. via
        handle::operator()). You probably won't need to use this. */
    automatic_reference,

    /** Reference an existing object (i.e. do not create a new copy) and take
        ownership. Python will call the destructor and delete operator when the
        object’s reference count reaches zero. Undefined behavior ensues when
        the C++ side does the same.. */
    take_ownership,

    /** Create a new copy of the returned object, which will be owned by
        Python. This policy is comparably safe because the lifetimes of the two
        instances are decoupled. */
    copy,

    /** Use std::move to move the return value contents into a new instance
        that will be owned by Python. This policy is comparably safe because the
        lifetimes of the two instances (move source and destination) are
        decoupled. */
    move,

    /** Reference an existing object, but do not take ownership. The C++ side
        is responsible for managing the object’s lifetime and deallocating it
        when it is no longer used. Warning: undefined behavior will ensue when
        the C++ side deletes an object that is still referenced and used by
        Python. */
    reference,

    /** This policy only applies to methods and properties. It references the
        object without taking ownership similar to the above
        return_value_policy::reference policy. In contrast to that policy, the
        function or property’s implicit this argument (called the parent) is
        considered to be the the owner of the return value (the child).
        pybind11 then couples the lifetime of the parent to the child via a
        reference relationship that ensures that the parent cannot be garbage
        collected while Python is still using the child. More advanced
        variations of this scheme are also possible using combinations of
        return_value_policy::reference and the keep_alive call policy */
    reference_internal
};

PYBIND11_NAMESPACE_BEGIN(detail)

inline static constexpr int log2(size_t n, int k = 0) { return (n <= 1) ? k : log2(n >> 1, k + 1); }

// Returns the size as a multiple of sizeof(void *), rounded up.
inline static constexpr size_t size_in_ptrs(size_t s) { return 1 + ((s - 1) >> log2(sizeof(void *))); }

/**
 * The space to allocate for simple layout instance holders (see below) in multiple of the size of
 * a pointer (e.g.  2 means 16 bytes on 64-bit architectures).  The default is the minimum required
 * to holder either a std::unique_ptr or std::shared_ptr (which is almost always
 * sizeof(std::shared_ptr<T>)).
 */
constexpr size_t instance_simple_holder_in_ptrs() {
    static_assert(sizeof(std::shared_ptr<int>) >= sizeof(std::unique_ptr<int>),
            "pybind assumes std::shared_ptrs are at least as big as std::unique_ptrs");
    return size_in_ptrs(sizeof(std::shared_ptr<int>));
}

// Forward declarations
struct type_info;
struct value_and_holder;

struct nonsimple_values_and_holders {
    void **values_and_holders;
    uint8_t *status;
};

/// The 'instance' type which needs to be standard layout (need to be able to use 'offsetof')
struct instance {
    PyObject_HEAD
    /// Storage for pointers and holder; see simple_layout, below, for a description
    union {
        void *simple_value_holder[1 + instance_simple_holder_in_ptrs()];
        nonsimple_values_and_holders nonsimple;
    };
    /// Weak references
    PyObject *weakrefs;
    /// If true, the pointer is owned which means we're free to manage it with a holder.
    bool owned : 1;
    /**
     * An instance has two possible value/holder layouts.
     *
     * Simple layout (when this flag is true), means the `simple_value_holder` is set with a pointer
     * and the holder object governing that pointer, i.e. [val1*][holder].  This layout is applied
     * whenever there is no python-side multiple inheritance of bound C++ types *and* the type's
     * holder will fit in the default space (which is large enough to hold either a std::unique_ptr
     * or std::shared_ptr).
     *
     * Non-simple layout applies when using custom holders that require more space than `shared_ptr`
     * (which is typically the size of two pointers), or when multiple inheritance is used on the
     * python side.  Non-simple layout allocates the required amount of memory to have multiple
     * bound C++ classes as parents.  Under this layout, `nonsimple.values_and_holders` is set to a
     * pointer to allocated space of the required space to hold a sequence of value pointers and
     * holders followed `status`, a set of bit flags (1 byte each), i.e.
     * [val1*][holder1][val2*][holder2]...[bb...]  where each [block] is rounded up to a multiple of
     * `sizeof(void *)`.  `nonsimple.status` is, for convenience, a pointer to the
     * beginning of the [bb...] block (but not independently allocated).
     *
     * Status bits indicate whether the associated holder is constructed (&
     * status_holder_constructed) and whether the value pointer is registered (&
     * status_instance_registered) in `registered_instances`.
     */
    bool simple_layout : 1;
    /// For simple layout, tracks whether the holder has been constructed
    bool simple_holder_constructed : 1;
    /// For simple layout, tracks whether the instance is registered in `registered_instances`
    bool simple_instance_registered : 1;
    /// If true, get_internals().patients has an entry for this object
    bool has_patients : 1;

    /// Initializes all of the above type/values/holders data (but not the instance values themselves)
    void allocate_layout();

    /// Destroys/deallocates all of the above
    void deallocate_layout();

    /// Returns the value_and_holder wrapper for the given type (or the first, if `find_type`
    /// omitted).  Returns a default-constructed (with `.inst = nullptr`) object on failure if
    /// `throw_if_missing` is false.
    value_and_holder get_value_and_holder(const type_info *find_type = nullptr, bool throw_if_missing = true);

    /// Bit values for the non-simple status flags
    static constexpr uint8_t status_holder_constructed  = 1;
    static constexpr uint8_t status_instance_registered = 2;
};

static_assert(std::is_standard_layout<instance>::value, "Internal error: `pybind11::detail::instance` is not standard layout!");

/// from __cpp_future__ import (convenient aliases from C++14/17)
#if defined(PYBIND11_CPP14) && (!defined(_MSC_VER) || _MSC_VER >= 1910)
using std::enable_if_t;
using std::conditional_t;
using std::remove_cv_t;
using std::remove_reference_t;
#else
template <bool B, typename T = void> using enable_if_t = typename std::enable_if<B, T>::type;
template <bool B, typename T, typename F> using conditional_t = typename std::conditional<B, T, F>::type;
template <typename T> using remove_cv_t = typename std::remove_cv<T>::type;
template <typename T> using remove_reference_t = typename std::remove_reference<T>::type;
#endif

/// Index sequences
#if defined(PYBIND11_CPP14)
using std::index_sequence;
using std::make_index_sequence;
#else
template<size_t ...> struct index_sequence  { };
template<size_t N, size_t ...S> struct make_index_sequence_impl : make_index_sequence_impl <N - 1, N - 1, S...> { };
template<size_t ...S> struct make_index_sequence_impl <0, S...> { typedef index_sequence<S...> type; };
template<size_t N> using make_index_sequence = typename make_index_sequence_impl<N>::type;
#endif

/// Make an index sequence of the indices of true arguments
template <typename ISeq, size_t, bool...> struct select_indices_impl { using type = ISeq; };
template <size_t... IPrev, size_t I, bool B, bool... Bs> struct select_indices_impl<index_sequence<IPrev...>, I, B, Bs...>
    : select_indices_impl<conditional_t<B, index_sequence<IPrev..., I>, index_sequence<IPrev...>>, I + 1, Bs...> {};
template <bool... Bs> using select_indices = typename select_indices_impl<index_sequence<>, 0, Bs...>::type;

/// Backports of std::bool_constant and std::negation to accommodate older compilers
template <bool B> using bool_constant = std::integral_constant<bool, B>;
template <typename T> struct negation : bool_constant<!T::value> { };

template <typename...> struct void_t_impl { using type = void; };
template <typename... Ts> using void_t = typename void_t_impl<Ts...>::type;

/// Compile-time all/any/none of that check the boolean value of all template types
#if defined(__cpp_fold_expressions) && !(defined(_MSC_VER) && (_MSC_VER < 1916))
template <class... Ts> using all_of = bool_constant<(Ts::value && ...)>;
template <class... Ts> using any_of = bool_constant<(Ts::value || ...)>;
#elif !defined(_MSC_VER)
template <bool...> struct bools {};
template <class... Ts> using all_of = std::is_same<
    bools<Ts::value..., true>,
    bools<true, Ts::value...>>;
template <class... Ts> using any_of = negation<all_of<negation<Ts>...>>;
#else
// MSVC has trouble with the above, but supports std::conjunction, which we can use instead (albeit
// at a slight loss of compilation efficiency).
template <class... Ts> using all_of = std::conjunction<Ts...>;
template <class... Ts> using any_of = std::disjunction<Ts...>;
#endif
template <class... Ts> using none_of = negation<any_of<Ts...>>;

template <class T, template<class> class... Predicates> using satisfies_all_of = all_of<Predicates<T>...>;
template <class T, template<class> class... Predicates> using satisfies_any_of = any_of<Predicates<T>...>;
template <class T, template<class> class... Predicates> using satisfies_none_of = none_of<Predicates<T>...>;

/// Strip the class from a method type
template <typename T> struct remove_class { };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...)> { typedef R type(A...); };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...) const> { typedef R type(A...); };

/// Helper template to strip away type modifiers
template <typename T> struct intrinsic_type                       { typedef T type; };
template <typename T> struct intrinsic_type<const T>              { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T*>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&>                   { typedef typename intrinsic_type<T>::type type; };
template <typename T> struct intrinsic_type<T&&>                  { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<const T[N]> { typedef typename intrinsic_type<T>::type type; };
template <typename T, size_t N> struct intrinsic_type<T[N]>       { typedef typename intrinsic_type<T>::type type; };
template <typename T> using intrinsic_t = typename intrinsic_type<T>::type;

/// Helper type to replace 'void' in some expressions
struct void_type { };

/// Helper template which holds a list of types
template <typename...> struct type_list { };

/// Compile-time integer sum
#ifdef __cpp_fold_expressions
template <typename... Ts> constexpr size_t constexpr_sum(Ts... ns) { return (0 + ... + size_t{ns}); }
#else
constexpr size_t constexpr_sum() { return 0; }
template <typename T, typename... Ts>
constexpr size_t constexpr_sum(T n, Ts... ns) { return size_t{n} + constexpr_sum(ns...); }
#endif

PYBIND11_NAMESPACE_BEGIN(constexpr_impl)
/// Implementation details for constexpr functions
constexpr int first(int i) { return i; }
template <typename T, typename... Ts>
constexpr int first(int i, T v, Ts... vs) { return v ? i : first(i + 1, vs...); }

constexpr int last(int /*i*/, int result) { return result; }
template <typename T, typename... Ts>
constexpr int last(int i, int result, T v, Ts... vs) { return last(i + 1, v ? i : result, vs...); }
PYBIND11_NAMESPACE_END(constexpr_impl)

/// Return the index of the first type in Ts which satisfies Predicate<T>.  Returns sizeof...(Ts) if
/// none match.
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_first() { return constexpr_impl::first(0, Predicate<Ts>::value...); }

/// Return the index of the last type in Ts which satisfies Predicate<T>, or -1 if none match.
template <template<typename> class Predicate, typename... Ts>
constexpr int constexpr_last() { return constexpr_impl::last(0, -1, Predicate<Ts>::value...); }

/// Return the Nth element from the parameter pack
template <size_t N, typename T, typename... Ts>
struct pack_element { using type = typename pack_element<N - 1, Ts...>::type; };
template <typename T, typename... Ts>
struct pack_element<0, T, Ts...> { using type = T; };

/// Return the one and only type which matches the predicate, or Default if none match.
/// If more than one type matches the predicate, fail at compile-time.
template <template<typename> class Predicate, typename Default, typename... Ts>
struct exactly_one {
    static constexpr auto found = constexpr_sum(Predicate<Ts>::value...);
    static_assert(found <= 1, "Found more than one type matching the predicate");

    static constexpr auto index = found ? constexpr_first<Predicate, Ts...>() : 0;
    using type = conditional_t<found, typename pack_element<index, Ts...>::type, Default>;
};
template <template<typename> class P, typename Default>
struct exactly_one<P, Default> { using type = Default; };

template <template<typename> class Predicate, typename Default, typename... Ts>
using exactly_one_t = typename exactly_one<Predicate, Default, Ts...>::type;

/// Defer the evaluation of type T until types Us are instantiated
template <typename T, typename... /*Us*/> struct deferred_type { using type = T; };
template <typename T, typename... Us> using deferred_t = typename deferred_type<T, Us...>::type;

/// Like is_base_of, but requires a strict base (i.e. `is_strict_base_of<T, T>::value == false`,
/// unlike `std::is_base_of`)
template <typename Base, typename Derived> using is_strict_base_of = bool_constant<
    std::is_base_of<Base, Derived>::value && !std::is_same<Base, Derived>::value>;

/// Like is_base_of, but also requires that the base type is accessible (i.e. that a Derived pointer
/// can be converted to a Base pointer)
template <typename Base, typename Derived> using is_accessible_base_of = bool_constant<
    std::is_base_of<Base, Derived>::value && std::is_convertible<Derived *, Base *>::value>;

template <template<typename...> class Base>
struct is_template_base_of_impl {
    template <typename... Us> static std::true_type check(Base<Us...> *);
    static std::false_type check(...);
};

/// Check if a template is the base of a type. For example:
/// `is_template_base_of<Base, T>` is true if `struct T : Base<U> {}` where U can be anything
template <template<typename...> class Base, typename T>
#if !defined(_MSC_VER)
using is_template_base_of = decltype(is_template_base_of_impl<Base>::check((intrinsic_t<T>*)nullptr));
#else // MSVC2015 has trouble with decltype in template aliases
struct is_template_base_of : decltype(is_template_base_of_impl<Base>::check((intrinsic_t<T>*)nullptr)) { };
#endif

/// Check if T is an instantiation of the template `Class`. For example:
/// `is_instantiation<shared_ptr, T>` is true if `T == shared_ptr<U>` where U can be anything.
template <template<typename...> class Class, typename T>
struct is_instantiation : std::false_type { };
template <template<typename...> class Class, typename... Us>
struct is_instantiation<Class, Class<Us...>> : std::true_type { };

/// Check if T is std::shared_ptr<U> where U can be anything
template <typename T> using is_shared_ptr = is_instantiation<std::shared_ptr, T>;

/// Check if T looks like an input iterator
template <typename T, typename = void> struct is_input_iterator : std::false_type {};
template <typename T>
struct is_input_iterator<T, void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>>
    : std::true_type {};

template <typename T> using is_function_pointer = bool_constant<
    std::is_pointer<T>::value && std::is_function<typename std::remove_pointer<T>::type>::value>;

template <typename F> struct strip_function_object {
    using type = typename remove_class<decltype(&F::operator())>::type;
};

// Extracts the function signature from a function, function pointer or lambda.
template <typename Function, typename F = remove_reference_t<Function>>
using function_signature_t = conditional_t<
    std::is_function<F>::value,
    F,
    typename conditional_t<
        std::is_pointer<F>::value || std::is_member_pointer<F>::value,
        std::remove_pointer<F>,
        strip_function_object<F>
    >::type
>;

/// Returns true if the type looks like a lambda: that is, isn't a function, pointer or member
/// pointer.  Note that this can catch all sorts of other things, too; this is intended to be used
/// in a place where passing a lambda makes sense.
template <typename T> using is_lambda = satisfies_none_of<remove_reference_t<T>,
        std::is_function, std::is_pointer, std::is_member_pointer>;

/// Ignore that a variable is unused in compiler warnings
inline void ignore_unused(const int *) { }

/// Apply a function over each element of a parameter pack
#ifdef __cpp_fold_expressions
#define PYBIND11_EXPAND_SIDE_EFFECTS(PATTERN) (((PATTERN), void()), ...)
#else
using expand_side_effects = bool[];
#define PYBIND11_EXPAND_SIDE_EFFECTS(PATTERN) (void)pybind11::detail::expand_side_effects{ ((PATTERN), void(), false)..., false }
#endif

PYBIND11_NAMESPACE_END(detail)

/// C++ bindings of builtin Python exceptions
class builtin_exception : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    /// Set the error using the Python C API
    virtual void set_error() const = 0;
};

#define PYBIND11_RUNTIME_EXCEPTION(name, type) \
    class name : public builtin_exception { public: \
        using builtin_exception::builtin_exception; \
        name() : name("") { } \
        void set_error() const override { PyErr_SetString(type, what()); } \
    };

PYBIND11_RUNTIME_EXCEPTION(stop_iteration, PyExc_StopIteration)
PYBIND11_RUNTIME_EXCEPTION(index_error, PyExc_IndexError)
PYBIND11_RUNTIME_EXCEPTION(key_error, PyExc_KeyError)
PYBIND11_RUNTIME_EXCEPTION(value_error, PyExc_ValueError)
PYBIND11_RUNTIME_EXCEPTION(type_error, PyExc_TypeError)
PYBIND11_RUNTIME_EXCEPTION(buffer_error, PyExc_BufferError)
PYBIND11_RUNTIME_EXCEPTION(import_error, PyExc_ImportError)
PYBIND11_RUNTIME_EXCEPTION(cast_error, PyExc_RuntimeError) /// Thrown when pybind11::cast or handle::call fail due to a type casting error
PYBIND11_RUNTIME_EXCEPTION(reference_cast_error, PyExc_RuntimeError) /// Used internally

[[noreturn]] PYBIND11_NOINLINE inline void pybind11_fail(const char *reason) { throw std::runtime_error(reason); }
[[noreturn]] PYBIND11_NOINLINE inline void pybind11_fail(const std::string &reason) { throw std::runtime_error(reason); }

template <typename T, typename SFINAE = void> struct format_descriptor { };

PYBIND11_NAMESPACE_BEGIN(detail)
// Returns the index of the given type in the type char array below, and in the list in numpy.h
// The order here is: bool; 8 ints ((signed,unsigned)x(8,16,32,64)bits); float,double,long double;
// complex float,double,long double.  Note that the long double types only participate when long
// double is actually longer than double (it isn't under MSVC).
// NB: not only the string below but also complex.h and numpy.h rely on this order.
template <typename T, typename SFINAE = void> struct is_fmt_numeric { static constexpr bool value = false; };
template <typename T> struct is_fmt_numeric<T, enable_if_t<std::is_arithmetic<T>::value>> {
    static constexpr bool value = true;
    static constexpr int index = std::is_same<T, bool>::value ? 0 : 1 + (
        std::is_integral<T>::value ? detail::log2(sizeof(T))*2 + std::is_unsigned<T>::value : 8 + (
        std::is_same<T, double>::value ? 1 : std::is_same<T, long double>::value ? 2 : 0));
};
PYBIND11_NAMESPACE_END(detail)

template <typename T> struct format_descriptor<T, detail::enable_if_t<std::is_arithmetic<T>::value>> {
    static constexpr const char c = "?bBhHiIqQfdg"[detail::is_fmt_numeric<T>::index];
    static constexpr const char value[2] = { c, '\0' };
    static std::string format() { return std::string(1, c); }
};

#if !defined(PYBIND11_CPP17)

template <typename T> constexpr const char format_descriptor<
    T, detail::enable_if_t<std::is_arithmetic<T>::value>>::value[2];

#endif

/// RAII wrapper that temporarily clears any Python error state
struct error_scope {
    PyObject *type, *value, *trace;
    error_scope() { PyErr_Fetch(&type, &value, &trace); }
    ~error_scope() { PyErr_Restore(type, value, trace); }
};

/// Dummy destructor wrapper that can be used to expose classes with a private destructor
struct nodelete { template <typename T> void operator()(T*) { } };

PYBIND11_NAMESPACE_BEGIN(detail)
template <typename... Args>
struct overload_cast_impl {
    constexpr overload_cast_impl() {} // MSVC 2015 needs this

    template <typename Return>
    constexpr auto operator()(Return (*pf)(Args...)) const noexcept
                              -> decltype(pf) { return pf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...), std::false_type = {}) const noexcept
                              -> decltype(pmf) { return pmf; }

    template <typename Return, typename Class>
    constexpr auto operator()(Return (Class::*pmf)(Args...) const, std::true_type) const noexcept
                              -> decltype(pmf) { return pmf; }
};
PYBIND11_NAMESPACE_END(detail)

// overload_cast requires variable templates: C++14
#if defined(PYBIND11_CPP14)
#define PYBIND11_OVERLOAD_CAST 1
/// Syntax sugar for resolving overloaded function pointers:
///  - regular: static_cast<Return (Class::*)(Arg0, Arg1, Arg2)>(&Class::func)
///  - sweet:   overload_cast<Arg0, Arg1, Arg2>(&Class::func)
template <typename... Args>
static constexpr detail::overload_cast_impl<Args...> overload_cast = {};
// MSVC 2015 only accepts this particular initialization syntax for this variable template.
#endif

/// Const member function selector for overload_cast
///  - regular: static_cast<Return (Class::*)(Arg) const>(&Class::func)
///  - sweet:   overload_cast<Arg>(&Class::func, const_)
static constexpr auto const_ = std::true_type{};

#if !defined(PYBIND11_CPP14) // no overload_cast: providing something that static_assert-fails:
template <typename... Args> struct overload_cast {
    static_assert(detail::deferred_t<std::false_type, Args...>::value,
                  "pybind11::overload_cast<...> requires compiling in C++14 mode");
};
#endif // overload_cast

PYBIND11_NAMESPACE_BEGIN(detail)

// Adaptor for converting arbitrary container arguments into a vector; implicitly convertible from
// any standard container (or C-style array) supporting std::begin/std::end, any singleton
// arithmetic type (if T is arithmetic), or explicitly constructible from an iterator pair.
template <typename T>
class any_container {
    std::vector<T> v;
public:
    any_container() = default;

    // Can construct from a pair of iterators
    template <typename It, typename = enable_if_t<is_input_iterator<It>::value>>
    any_container(It first, It last) : v(first, last) { }

    // Implicit conversion constructor from any arbitrary container type with values convertible to T
    template <typename Container, typename = enable_if_t<std::is_convertible<decltype(*std::begin(std::declval<const Container &>())), T>::value>>
    any_container(const Container &c) : any_container(std::begin(c), std::end(c)) { }

    // initializer_list's aren't deducible, so don't get matched by the above template; we need this
    // to explicitly allow implicit conversion from one:
    template <typename TIn, typename = enable_if_t<std::is_convertible<TIn, T>::value>>
    any_container(const std::initializer_list<TIn> &c) : any_container(c.begin(), c.end()) { }

    // Avoid copying if given an rvalue vector of the correct type.
    any_container(std::vector<T> &&v) : v(std::move(v)) { }

    // Moves the vector out of an rvalue any_container
    operator std::vector<T> &&() && { return std::move(v); }

    // Dereferencing obtains a reference to the underlying vector
    std::vector<T> &operator*() { return v; }
    const std::vector<T> &operator*() const { return v; }

    // -> lets you call methods on the underlying vector
    std::vector<T> *operator->() { return &v; }
    const std::vector<T> *operator->() const { return &v; }
};

PYBIND11_NAMESPACE_END(detail)



PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "detail/common.h" ============


//========= begin of #include "buffer_info.h" ============

/*
    pybind11/buffer_info.h: Python buffer object interface

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "detail/common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// Information record describing a Python buffer object
struct buffer_info {
    void *ptr = nullptr;          // Pointer to the underlying storage
    ssize_t itemsize = 0;         // Size of individual items in bytes
    ssize_t size = 0;             // Total number of entries
    std::string format;           // For homogeneous buffers, this should be set to format_descriptor<T>::format()
    ssize_t ndim = 0;             // Number of dimensions
    std::vector<ssize_t> shape;   // Shape of the tensor (1 entry per dimension)
    std::vector<ssize_t> strides; // Number of bytes between adjacent entries (for each per dimension)
    bool readonly = false;        // flag to indicate if the underlying storage may be written to

    buffer_info() { }

    buffer_info(void *ptr, ssize_t itemsize, const std::string &format, ssize_t ndim,
                detail::any_container<ssize_t> shape_in, detail::any_container<ssize_t> strides_in, bool readonly=false)
    : ptr(ptr), itemsize(itemsize), size(1), format(format), ndim(ndim),
      shape(std::move(shape_in)), strides(std::move(strides_in)), readonly(readonly) {
        if (ndim != (ssize_t) shape.size() || ndim != (ssize_t) strides.size())
            pybind11_fail("buffer_info: ndim doesn't match shape and/or strides length");
        for (size_t i = 0; i < (size_t) ndim; ++i)
            size *= shape[i];
    }

    template <typename T>
    buffer_info(T *ptr, detail::any_container<ssize_t> shape_in, detail::any_container<ssize_t> strides_in, bool readonly=false)
    : buffer_info(private_ctr_tag(), ptr, sizeof(T), format_descriptor<T>::format(), static_cast<ssize_t>(shape_in->size()), std::move(shape_in), std::move(strides_in), readonly) { }

    buffer_info(void *ptr, ssize_t itemsize, const std::string &format, ssize_t size, bool readonly=false)
    : buffer_info(ptr, itemsize, format, 1, {size}, {itemsize}, readonly) { }

    template <typename T>
    buffer_info(T *ptr, ssize_t size, bool readonly=false)
    : buffer_info(ptr, sizeof(T), format_descriptor<T>::format(), size, readonly) { }

    template <typename T>
    buffer_info(const T *ptr, ssize_t size, bool readonly=true)
    : buffer_info(const_cast<T*>(ptr), sizeof(T), format_descriptor<T>::format(), size, readonly) { }

    explicit buffer_info(Py_buffer *view, bool ownview = true)
    : buffer_info(view->buf, view->itemsize, view->format, view->ndim,
            {view->shape, view->shape + view->ndim}, {view->strides, view->strides + view->ndim}, view->readonly) {
        this->m_view = view;
        this->ownview = ownview;
    }

    buffer_info(const buffer_info &) = delete;
    buffer_info& operator=(const buffer_info &) = delete;

    buffer_info(buffer_info &&other) {
        (*this) = std::move(other);
    }

    buffer_info& operator=(buffer_info &&rhs) {
        ptr = rhs.ptr;
        itemsize = rhs.itemsize;
        size = rhs.size;
        format = std::move(rhs.format);
        ndim = rhs.ndim;
        shape = std::move(rhs.shape);
        strides = std::move(rhs.strides);
        std::swap(m_view, rhs.m_view);
        std::swap(ownview, rhs.ownview);
        readonly = rhs.readonly;
        return *this;
    }

    ~buffer_info() {
        if (m_view && ownview) { PyBuffer_Release(m_view); delete m_view; }
    }

    Py_buffer *view() const { return m_view; }
    Py_buffer *&view() { return m_view; }
private:
    struct private_ctr_tag { };

    buffer_info(private_ctr_tag, void *ptr, ssize_t itemsize, const std::string &format, ssize_t ndim,
                detail::any_container<ssize_t> &&shape_in, detail::any_container<ssize_t> &&strides_in, bool readonly)
    : buffer_info(ptr, itemsize, format, ndim, std::move(shape_in), std::move(strides_in), readonly) { }

    Py_buffer *m_view = nullptr;
    bool ownview = false;
};

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T, typename SFINAE = void> struct compare_buffer_info {
    static bool compare(const buffer_info& b) {
        return b.format == format_descriptor<T>::format() && b.itemsize == (ssize_t) sizeof(T);
    }
};

template <typename T> struct compare_buffer_info<T, detail::enable_if_t<std::is_integral<T>::value>> {
    static bool compare(const buffer_info& b) {
        return (size_t) b.itemsize == sizeof(T) && (b.format == format_descriptor<T>::value ||
            ((sizeof(T) == sizeof(long)) && b.format == (std::is_unsigned<T>::value ? "L" : "l")) ||
            ((sizeof(T) == sizeof(size_t)) && b.format == (std::is_unsigned<T>::value ? "N" : "n")));
    }
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "buffer_info.h" ============


//========= begin of #include "pytypes.h" ============

/*
    pybind11/pytypes.h: Convenience wrapper classes for basic Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "detail/common.h"
// ans ignore: #include "buffer_info.h"
#include <utility>
#include <type_traits>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/* A few forward declarations */
class handle; class object;
class str; class iterator;
struct arg; struct arg_v;

PYBIND11_NAMESPACE_BEGIN(detail)
class args_proxy;
inline bool isinstance_generic(handle obj, const std::type_info &tp);

// Accessor forward declarations
template <typename Policy> class accessor;
namespace accessor_policies {
    struct obj_attr;
    struct str_attr;
    struct generic_item;
    struct sequence_item;
    struct list_item;
    struct tuple_item;
}
using obj_attr_accessor = accessor<accessor_policies::obj_attr>;
using str_attr_accessor = accessor<accessor_policies::str_attr>;
using item_accessor = accessor<accessor_policies::generic_item>;
using sequence_accessor = accessor<accessor_policies::sequence_item>;
using list_accessor = accessor<accessor_policies::list_item>;
using tuple_accessor = accessor<accessor_policies::tuple_item>;

/// Tag and check to identify a class which implements the Python object API
class pyobject_tag { };
template <typename T> using is_pyobject = std::is_base_of<pyobject_tag, remove_reference_t<T>>;

/** \rst
    A mixin class which adds common functions to `handle`, `object` and various accessors.
    The only requirement for `Derived` is to implement ``PyObject *Derived::ptr() const``.
\endrst */
template <typename Derived>
class object_api : public pyobject_tag {
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

public:
    /** \rst
        Return an iterator equivalent to calling ``iter()`` in Python. The object
        must be a collection which supports the iteration protocol.
    \endrst */
    iterator begin() const;
    /// Return a sentinel which ends iteration.
    iterator end() const;

    /** \rst
        Return an internal functor to invoke the object's sequence protocol. Casting
        the returned ``detail::item_accessor`` instance to a `handle` or `object`
        subclass causes a corresponding call to ``__getitem__``. Assigning a `handle`
        or `object` subclass causes a call to ``__setitem__``.
    \endrst */
    item_accessor operator[](handle key) const;
    /// See above (the only difference is that they key is provided as a string literal)
    item_accessor operator[](const char *key) const;

    /** \rst
        Return an internal functor to access the object's attributes. Casting the
        returned ``detail::obj_attr_accessor`` instance to a `handle` or `object`
        subclass causes a corresponding call to ``getattr``. Assigning a `handle`
        or `object` subclass causes a call to ``setattr``.
    \endrst */
    obj_attr_accessor attr(handle key) const;
    /// See above (the only difference is that they key is provided as a string literal)
    str_attr_accessor attr(const char *key) const;

    /** \rst
        Matches * unpacking in Python, e.g. to unpack arguments out of a ``tuple``
        or ``list`` for a function call. Applying another * to the result yields
        ** unpacking, e.g. to unpack a dict as function keyword arguments.
        See :ref:`calling_python_functions`.
    \endrst */
    args_proxy operator*() const;

    /// Check if the given item is contained within this object, i.e. ``item in obj``.
    template <typename T> bool contains(T &&item) const;

    /** \rst
        Assuming the Python object is a function or implements the ``__call__``
        protocol, ``operator()`` invokes the underlying function, passing an
        arbitrary set of parameters. The result is returned as a `object` and
        may need to be converted back into a Python object using `handle::cast()`.

        When some of the arguments cannot be converted to Python objects, the
        function will throw a `cast_error` exception. When the Python function
        call fails, a `error_already_set` exception is thrown.
    \endrst */
    template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
    object operator()(Args &&...args) const;
    template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
    PYBIND11_DEPRECATED("call(...) was deprecated in favor of operator()(...)")
        object call(Args&&... args) const;

    /// Equivalent to ``obj is other`` in Python.
    bool is(object_api const& other) const { return derived().ptr() == other.derived().ptr(); }
    /// Equivalent to ``obj is None`` in Python.
    bool is_none() const { return derived().ptr() == Py_None; }
    /// Equivalent to obj == other in Python
    bool equal(object_api const &other) const      { return rich_compare(other, Py_EQ); }
    bool not_equal(object_api const &other) const  { return rich_compare(other, Py_NE); }
    bool operator<(object_api const &other) const  { return rich_compare(other, Py_LT); }
    bool operator<=(object_api const &other) const { return rich_compare(other, Py_LE); }
    bool operator>(object_api const &other) const  { return rich_compare(other, Py_GT); }
    bool operator>=(object_api const &other) const { return rich_compare(other, Py_GE); }

    object operator-() const;
    object operator~() const;
    object operator+(object_api const &other) const;
    object operator+=(object_api const &other) const;
    object operator-(object_api const &other) const;
    object operator-=(object_api const &other) const;
    object operator*(object_api const &other) const;
    object operator*=(object_api const &other) const;
    object operator/(object_api const &other) const;
    object operator/=(object_api const &other) const;
    object operator|(object_api const &other) const;
    object operator|=(object_api const &other) const;
    object operator&(object_api const &other) const;
    object operator&=(object_api const &other) const;
    object operator^(object_api const &other) const;
    object operator^=(object_api const &other) const;
    object operator<<(object_api const &other) const;
    object operator<<=(object_api const &other) const;
    object operator>>(object_api const &other) const;
    object operator>>=(object_api const &other) const;

    PYBIND11_DEPRECATED("Use py::str(obj) instead")
    pybind11::str str() const;

    /// Get or set the object's docstring, i.e. ``obj.__doc__``.
    str_attr_accessor doc() const;

    /// Return the object's current reference count
    int ref_count() const { return static_cast<int>(Py_REFCNT(derived().ptr())); }
    /// Return a handle to the Python type object underlying the instance
    handle get_type() const;

private:
    bool rich_compare(object_api const &other, int value) const;
};

PYBIND11_NAMESPACE_END(detail)

/** \rst
    Holds a reference to a Python object (no reference counting)

    The `handle` class is a thin wrapper around an arbitrary Python object (i.e. a
    ``PyObject *`` in Python's C API). It does not perform any automatic reference
    counting and merely provides a basic C++ interface to various Python API functions.

    .. seealso::
        The `object` class inherits from `handle` and adds automatic reference
        counting features.
\endrst */
class handle : public detail::object_api<handle> {
public:
    /// The default constructor creates a handle with a ``nullptr``-valued pointer
    handle() = default;
    /// Creates a ``handle`` from the given raw Python object pointer
    handle(PyObject *ptr) : m_ptr(ptr) { } // Allow implicit conversion from PyObject*

    /// Return the underlying ``PyObject *`` pointer
    PyObject *ptr() const { return m_ptr; }
    PyObject *&ptr() { return m_ptr; }

    /** \rst
        Manually increase the reference count of the Python object. Usually, it is
        preferable to use the `object` class which derives from `handle` and calls
        this function automatically. Returns a reference to itself.
    \endrst */
    const handle& inc_ref() const & { Py_XINCREF(m_ptr); return *this; }

    /** \rst
        Manually decrease the reference count of the Python object. Usually, it is
        preferable to use the `object` class which derives from `handle` and calls
        this function automatically. Returns a reference to itself.
    \endrst */
    const handle& dec_ref() const & { Py_XDECREF(m_ptr); return *this; }

    /** \rst
        Attempt to cast the Python object into the given C++ type. A `cast_error`
        will be throw upon failure.
    \endrst */
    template <typename T> T cast() const;
    /// Return ``true`` when the `handle` wraps a valid Python object
    explicit operator bool() const { return m_ptr != nullptr; }
    /** \rst
        Deprecated: Check that the underlying pointers are the same.
        Equivalent to ``obj1 is obj2`` in Python.
    \endrst */
    PYBIND11_DEPRECATED("Use obj1.is(obj2) instead")
    bool operator==(const handle &h) const { return m_ptr == h.m_ptr; }
    PYBIND11_DEPRECATED("Use !obj1.is(obj2) instead")
    bool operator!=(const handle &h) const { return m_ptr != h.m_ptr; }
    PYBIND11_DEPRECATED("Use handle::operator bool() instead")
    bool check() const { return m_ptr != nullptr; }
protected:
    PyObject *m_ptr = nullptr;
};

/** \rst
    Holds a reference to a Python object (with reference counting)

    Like `handle`, the `object` class is a thin wrapper around an arbitrary Python
    object (i.e. a ``PyObject *`` in Python's C API). In contrast to `handle`, it
    optionally increases the object's reference count upon construction, and it
    *always* decreases the reference count when the `object` instance goes out of
    scope and is destructed. When using `object` instances consistently, it is much
    easier to get reference counting right at the first attempt.
\endrst */
class object : public handle {
public:
    object() = default;
    PYBIND11_DEPRECATED("Use reinterpret_borrow<object>() or reinterpret_steal<object>()")
    object(handle h, bool is_borrowed) : handle(h) { if (is_borrowed) inc_ref(); }
    /// Copy constructor; always increases the reference count
    object(const object &o) : handle(o) { inc_ref(); }
    /// Move constructor; steals the object from ``other`` and preserves its reference count
    object(object &&other) noexcept { m_ptr = other.m_ptr; other.m_ptr = nullptr; }
    /// Destructor; automatically calls `handle::dec_ref()`
    ~object() { dec_ref(); }

    /** \rst
        Resets the internal pointer to ``nullptr`` without decreasing the
        object's reference count. The function returns a raw handle to the original
        Python object.
    \endrst */
    handle release() {
      PyObject *tmp = m_ptr;
      m_ptr = nullptr;
      return handle(tmp);
    }

    object& operator=(const object &other) {
        other.inc_ref();
        dec_ref();
        m_ptr = other.m_ptr;
        return *this;
    }

    object& operator=(object &&other) noexcept {
        if (this != &other) {
            handle temp(m_ptr);
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
            temp.dec_ref();
        }
        return *this;
    }

    // Calling cast() on an object lvalue just copies (via handle::cast)
    template <typename T> T cast() const &;
    // Calling on an object rvalue does a move, if needed and/or possible
    template <typename T> T cast() &&;

protected:
    // Tags for choosing constructors from raw PyObject *
    struct borrowed_t { };
    struct stolen_t { };

    template <typename T> friend T reinterpret_borrow(handle);
    template <typename T> friend T reinterpret_steal(handle);

public:
    // Only accessible from derived classes and the reinterpret_* functions
    object(handle h, borrowed_t) : handle(h) { inc_ref(); }
    object(handle h, stolen_t) : handle(h) { }
};

/** \rst
    Declare that a `handle` or ``PyObject *`` is a certain type and borrow the reference.
    The target type ``T`` must be `object` or one of its derived classes. The function
    doesn't do any conversions or checks. It's up to the user to make sure that the
    target type is correct.

    .. code-block:: cpp

        PyObject *p = PyList_GetItem(obj, index);
        py::object o = reinterpret_borrow<py::object>(p);
        // or
        py::tuple t = reinterpret_borrow<py::tuple>(p); // <-- `p` must be already be a `tuple`
\endrst */
template <typename T> T reinterpret_borrow(handle h) { return {h, object::borrowed_t{}}; }

/** \rst
    Like `reinterpret_borrow`, but steals the reference.

     .. code-block:: cpp

        PyObject *p = PyObject_Str(obj);
        py::str s = reinterpret_steal<py::str>(p); // <-- `p` must be already be a `str`
\endrst */
template <typename T> T reinterpret_steal(handle h) { return {h, object::stolen_t{}}; }

PYBIND11_NAMESPACE_BEGIN(detail)
inline std::string error_string();
PYBIND11_NAMESPACE_END(detail)

/// Fetch and hold an error which was already set in Python.  An instance of this is typically
/// thrown to propagate python-side errors back through C++ which can either be caught manually or
/// else falls back to the function dispatcher (which then raises the captured error back to
/// python).
class error_already_set : public std::runtime_error {
public:
    /// Constructs a new exception from the current Python error indicator, if any.  The current
    /// Python error indicator will be cleared.
    error_already_set() : std::runtime_error(detail::error_string()) {
        PyErr_Fetch(&m_type.ptr(), &m_value.ptr(), &m_trace.ptr());
    }

    error_already_set(const error_already_set &) = default;
    error_already_set(error_already_set &&) = default;

    inline ~error_already_set();

    /// Give the currently-held error back to Python, if any.  If there is currently a Python error
    /// already set it is cleared first.  After this call, the current object no longer stores the
    /// error variables (but the `.what()` string is still available).
    void restore() { PyErr_Restore(m_type.release().ptr(), m_value.release().ptr(), m_trace.release().ptr()); }

    /// If it is impossible to raise the currently-held error, such as in destructor, we can write
    /// it out using Python's unraisable hook (sys.unraisablehook). The error context should be
    /// some object whose repr() helps identify the location of the error. Python already knows the
    /// type and value of the error, so there is no need to repeat that. For example, __func__ could
    /// be helpful. After this call, the current object no longer stores the error variables,
    /// and neither does Python.
    void discard_as_unraisable(object err_context) {
        restore();
        PyErr_WriteUnraisable(err_context.ptr());
    }
    void discard_as_unraisable(const char *err_context) {
        discard_as_unraisable(reinterpret_steal<object>(PYBIND11_FROM_STRING(err_context)));
    }

    // Does nothing; provided for backwards compatibility.
    PYBIND11_DEPRECATED("Use of error_already_set.clear() is deprecated")
    void clear() {}

    /// Check if the currently trapped error type matches the given Python exception class (or a
    /// subclass thereof).  May also be passed a tuple to search for any exception class matches in
    /// the given tuple.
    bool matches(handle exc) const { return PyErr_GivenExceptionMatches(m_type.ptr(), exc.ptr()); }

    const object& type() const { return m_type; }
    const object& value() const { return m_value; }
    const object& trace() const { return m_trace; }

private:
    object m_type, m_value, m_trace;
};

/** \defgroup python_builtins _
    Unless stated otherwise, the following C++ functions behave the same
    as their Python counterparts.
 */

/** \ingroup python_builtins
    \rst
    Return true if ``obj`` is an instance of ``T``. Type ``T`` must be a subclass of
    `object` or a class which was exposed to Python as ``py::class_<T>``.
\endrst */
template <typename T, detail::enable_if_t<std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) { return T::check_(obj); }

template <typename T, detail::enable_if_t<!std::is_base_of<object, T>::value, int> = 0>
bool isinstance(handle obj) { return detail::isinstance_generic(obj, typeid(T)); }

template <> inline bool isinstance<handle>(handle) = delete;
template <> inline bool isinstance<object>(handle obj) { return obj.ptr() != nullptr; }

/// \ingroup python_builtins
/// Return true if ``obj`` is an instance of the ``type``.
inline bool isinstance(handle obj, handle type) {
    const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
    if (result == -1)
        throw error_already_set();
    return result != 0;
}

/// \addtogroup python_builtins
/// @{
inline bool hasattr(handle obj, handle name) {
    return PyObject_HasAttr(obj.ptr(), name.ptr()) == 1;
}

inline bool hasattr(handle obj, const char *name) {
    return PyObject_HasAttrString(obj.ptr(), name) == 1;
}

inline void delattr(handle obj, handle name) {
    if (PyObject_DelAttr(obj.ptr(), name.ptr()) != 0) { throw error_already_set(); }
}

inline void delattr(handle obj, const char *name) {
    if (PyObject_DelAttrString(obj.ptr(), name) != 0) { throw error_already_set(); }
}

inline object getattr(handle obj, handle name) {
    PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, const char *name) {
    PyObject *result = PyObject_GetAttrString(obj.ptr(), name);
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

inline object getattr(handle obj, handle name, handle default_) {
    if (PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr())) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

inline object getattr(handle obj, const char *name, handle default_) {
    if (PyObject *result = PyObject_GetAttrString(obj.ptr(), name)) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

inline void setattr(handle obj, handle name, handle value) {
    if (PyObject_SetAttr(obj.ptr(), name.ptr(), value.ptr()) != 0) { throw error_already_set(); }
}

inline void setattr(handle obj, const char *name, handle value) {
    if (PyObject_SetAttrString(obj.ptr(), name, value.ptr()) != 0) { throw error_already_set(); }
}

inline ssize_t hash(handle obj) {
    auto h = PyObject_Hash(obj.ptr());
    if (h == -1) { throw error_already_set(); }
    return h;
}

/// @} python_builtins

PYBIND11_NAMESPACE_BEGIN(detail)
inline handle get_function(handle value) {
    if (value) {
#if PY_MAJOR_VERSION >= 3
        if (PyInstanceMethod_Check(value.ptr()))
            value = PyInstanceMethod_GET_FUNCTION(value.ptr());
        else
#endif
        if (PyMethod_Check(value.ptr()))
            value = PyMethod_GET_FUNCTION(value.ptr());
    }
    return value;
}

// Helper aliases/functions to support implicit casting of values given to python accessors/methods.
// When given a pyobject, this simply returns the pyobject as-is; for other C++ type, the value goes
// through pybind11::cast(obj) to convert it to an `object`.
template <typename T, enable_if_t<is_pyobject<T>::value, int> = 0>
auto object_or_cast(T &&o) -> decltype(std::forward<T>(o)) { return std::forward<T>(o); }
// The following casting version is implemented in cast.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int> = 0>
object object_or_cast(T &&o);
// Match a PyObject*, which we want to convert directly to handle via its converting constructor
inline handle object_or_cast(PyObject *ptr) { return ptr; }

template <typename Policy>
class accessor : public object_api<accessor<Policy>> {
    using key_type = typename Policy::key_type;

public:
    accessor(handle obj, key_type key) : obj(obj), key(std::move(key)) { }
    accessor(const accessor &) = default;
    accessor(accessor &&) = default;

    // accessor overload required to override default assignment operator (templates are not allowed
    // to replace default compiler-generated assignments).
    void operator=(const accessor &a) && { std::move(*this).operator=(handle(a)); }
    void operator=(const accessor &a) & { operator=(handle(a)); }

    template <typename T> void operator=(T &&value) && {
        Policy::set(obj, key, object_or_cast(std::forward<T>(value)));
    }
    template <typename T> void operator=(T &&value) & {
        get_cache() = reinterpret_borrow<object>(object_or_cast(std::forward<T>(value)));
    }

    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj.attr(...) as bool is deprecated in favor of pybind11::hasattr(obj, ...)")
    explicit operator enable_if_t<std::is_same<T, accessor_policies::str_attr>::value ||
            std::is_same<T, accessor_policies::obj_attr>::value, bool>() const {
        return hasattr(obj, key);
    }
    template <typename T = Policy>
    PYBIND11_DEPRECATED("Use of obj[key] as bool is deprecated in favor of obj.contains(key)")
    explicit operator enable_if_t<std::is_same<T, accessor_policies::generic_item>::value, bool>() const {
        return obj.contains(key);
    }

    operator object() const { return get_cache(); }
    PyObject *ptr() const { return get_cache().ptr(); }
    template <typename T> T cast() const { return get_cache().template cast<T>(); }

private:
    object &get_cache() const {
        if (!cache) { cache = Policy::get(obj, key); }
        return cache;
    }

private:
    handle obj;
    key_type key;
    mutable object cache;
};

PYBIND11_NAMESPACE_BEGIN(accessor_policies)
struct obj_attr {
    using key_type = object;
    static object get(handle obj, handle key) { return getattr(obj, key); }
    static void set(handle obj, handle key, handle val) { setattr(obj, key, val); }
};

struct str_attr {
    using key_type = const char *;
    static object get(handle obj, const char *key) { return getattr(obj, key); }
    static void set(handle obj, const char *key, handle val) { setattr(obj, key, val); }
};

struct generic_item {
    using key_type = object;

    static object get(handle obj, handle key) {
        PyObject *result = PyObject_GetItem(obj.ptr(), key.ptr());
        if (!result) { throw error_already_set(); }
        return reinterpret_steal<object>(result);
    }

    static void set(handle obj, handle key, handle val) {
        if (PyObject_SetItem(obj.ptr(), key.ptr(), val.ptr()) != 0) { throw error_already_set(); }
    }
};

struct sequence_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PySequence_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_steal<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PySequence_SetItem does not steal a reference to 'val'
        if (PySequence_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct list_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PyList_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PyList_SetItem steals a reference to 'val'
        if (PyList_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};

struct tuple_item {
    using key_type = size_t;

    static object get(handle obj, size_t index) {
        PyObject *result = PyTuple_GetItem(obj.ptr(), static_cast<ssize_t>(index));
        if (!result) { throw error_already_set(); }
        return reinterpret_borrow<object>(result);
    }

    static void set(handle obj, size_t index, handle val) {
        // PyTuple_SetItem steals a reference to 'val'
        if (PyTuple_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
            throw error_already_set();
        }
    }
};
PYBIND11_NAMESPACE_END(accessor_policies)

/// STL iterator template used for tuple, list, sequence and dict
template <typename Policy>
class generic_iterator : public Policy {
    using It = generic_iterator;

public:
    using difference_type = ssize_t;
    using iterator_category = typename Policy::iterator_category;
    using value_type = typename Policy::value_type;
    using reference = typename Policy::reference;
    using pointer = typename Policy::pointer;

    generic_iterator() = default;
    generic_iterator(handle seq, ssize_t index) : Policy(seq, index) { }

    reference operator*() const { return Policy::dereference(); }
    reference operator[](difference_type n) const { return *(*this + n); }
    pointer operator->() const { return **this; }

    It &operator++() { Policy::increment(); return *this; }
    It operator++(int) { auto copy = *this; Policy::increment(); return copy; }
    It &operator--() { Policy::decrement(); return *this; }
    It operator--(int) { auto copy = *this; Policy::decrement(); return copy; }
    It &operator+=(difference_type n) { Policy::advance(n); return *this; }
    It &operator-=(difference_type n) { Policy::advance(-n); return *this; }

    friend It operator+(const It &a, difference_type n) { auto copy = a; return copy += n; }
    friend It operator+(difference_type n, const It &b) { return b + n; }
    friend It operator-(const It &a, difference_type n) { auto copy = a; return copy -= n; }
    friend difference_type operator-(const It &a, const It &b) { return a.distance_to(b); }

    friend bool operator==(const It &a, const It &b) { return a.equal(b); }
    friend bool operator!=(const It &a, const It &b) { return !(a == b); }
    friend bool operator< (const It &a, const It &b) { return b - a > 0; }
    friend bool operator> (const It &a, const It &b) { return b < a; }
    friend bool operator>=(const It &a, const It &b) { return !(a < b); }
    friend bool operator<=(const It &a, const It &b) { return !(a > b); }
};

PYBIND11_NAMESPACE_BEGIN(iterator_policies)
/// Quick proxy class needed to implement ``operator->`` for iterators which can't return pointers
template <typename T>
struct arrow_proxy {
    T value;

    arrow_proxy(T &&value) : value(std::move(value)) { }
    T *operator->() const { return &value; }
};

/// Lightweight iterator policy using just a simple pointer: see ``PySequence_Fast_ITEMS``
class sequence_fast_readonly {
protected:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = handle;
    using reference = const handle;
    using pointer = arrow_proxy<const handle>;

    sequence_fast_readonly(handle obj, ssize_t n) : ptr(PySequence_Fast_ITEMS(obj.ptr()) + n) { }

    reference dereference() const { return *ptr; }
    void increment() { ++ptr; }
    void decrement() { --ptr; }
    void advance(ssize_t n) { ptr += n; }
    bool equal(const sequence_fast_readonly &b) const { return ptr == b.ptr; }
    ssize_t distance_to(const sequence_fast_readonly &b) const { return ptr - b.ptr; }

private:
    PyObject **ptr;
};

/// Full read and write access using the sequence protocol: see ``detail::sequence_accessor``
class sequence_slow_readwrite {
protected:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = object;
    using reference = sequence_accessor;
    using pointer = arrow_proxy<const sequence_accessor>;

    sequence_slow_readwrite(handle obj, ssize_t index) : obj(obj), index(index) { }

    reference dereference() const { return {obj, static_cast<size_t>(index)}; }
    void increment() { ++index; }
    void decrement() { --index; }
    void advance(ssize_t n) { index += n; }
    bool equal(const sequence_slow_readwrite &b) const { return index == b.index; }
    ssize_t distance_to(const sequence_slow_readwrite &b) const { return index - b.index; }

private:
    handle obj;
    ssize_t index;
};

/// Python's dictionary protocol permits this to be a forward iterator
class dict_readonly {
protected:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<handle, handle>;
    using reference = const value_type;
    using pointer = arrow_proxy<const value_type>;

    dict_readonly() = default;
    dict_readonly(handle obj, ssize_t pos) : obj(obj), pos(pos) { increment(); }

    reference dereference() const { return {key, value}; }
    void increment() { if (!PyDict_Next(obj.ptr(), &pos, &key, &value)) { pos = -1; } }
    bool equal(const dict_readonly &b) const { return pos == b.pos; }

private:
    handle obj;
    PyObject *key = nullptr, *value = nullptr;
    ssize_t pos = -1;
};
PYBIND11_NAMESPACE_END(iterator_policies)

#if !defined(PYPY_VERSION)
using tuple_iterator = generic_iterator<iterator_policies::sequence_fast_readonly>;
using list_iterator = generic_iterator<iterator_policies::sequence_fast_readonly>;
#else
using tuple_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
using list_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
#endif

using sequence_iterator = generic_iterator<iterator_policies::sequence_slow_readwrite>;
using dict_iterator = generic_iterator<iterator_policies::dict_readonly>;

inline bool PyIterable_Check(PyObject *obj) {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter) {
        Py_DECREF(iter);
        return true;
    } else {
        PyErr_Clear();
        return false;
    }
}

inline bool PyNone_Check(PyObject *o) { return o == Py_None; }
inline bool PyEllipsis_Check(PyObject *o) { return o == Py_Ellipsis; }

inline bool PyUnicode_Check_Permissive(PyObject *o) { return PyUnicode_Check(o) || PYBIND11_BYTES_CHECK(o); }

inline bool PyStaticMethod_Check(PyObject *o) { return o->ob_type == &PyStaticMethod_Type; }

class kwargs_proxy : public handle {
public:
    explicit kwargs_proxy(handle h) : handle(h) { }
};

class args_proxy : public handle {
public:
    explicit args_proxy(handle h) : handle(h) { }
    kwargs_proxy operator*() const { return kwargs_proxy(*this); }
};

/// Python argument categories (using PEP 448 terms)
template <typename T> using is_keyword = std::is_base_of<arg, T>;
template <typename T> using is_s_unpacking = std::is_same<args_proxy, T>; // * unpacking
template <typename T> using is_ds_unpacking = std::is_same<kwargs_proxy, T>; // ** unpacking
template <typename T> using is_positional = satisfies_none_of<T,
    is_keyword, is_s_unpacking, is_ds_unpacking
>;
template <typename T> using is_keyword_or_ds = satisfies_any_of<T, is_keyword, is_ds_unpacking>;

// Call argument collector forward declarations
template <return_value_policy policy = return_value_policy::automatic_reference>
class simple_collector;
template <return_value_policy policy = return_value_policy::automatic_reference>
class unpacking_collector;

PYBIND11_NAMESPACE_END(detail)

// TODO: After the deprecated constructors are removed, this macro can be simplified by
//       inheriting ctors: `using Parent::Parent`. It's not an option right now because
//       the `using` statement triggers the parent deprecation warning even if the ctor
//       isn't even used.
#define PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    public: \
        PYBIND11_DEPRECATED("Use reinterpret_borrow<"#Name">() or reinterpret_steal<"#Name">()") \
        Name(handle h, bool is_borrowed) : Parent(is_borrowed ? Parent(h, borrowed_t{}) : Parent(h, stolen_t{})) { } \
        Name(handle h, borrowed_t) : Parent(h, borrowed_t{}) { } \
        Name(handle h, stolen_t) : Parent(h, stolen_t{}) { } \
        PYBIND11_DEPRECATED("Use py::isinstance<py::python_type>(obj) instead") \
        bool check() const { return m_ptr != nullptr && (bool) CheckFun(m_ptr); } \
        static bool check_(handle h) { return h.ptr() != nullptr && CheckFun(h.ptr()); }

#define PYBIND11_OBJECT_CVT(Name, Parent, CheckFun, ConvertFun) \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */ \
    Name(const object &o) \
    : Parent(check_(o) ? o.inc_ref().ptr() : ConvertFun(o.ptr()), stolen_t{}) \
    { if (!m_ptr) throw error_already_set(); } \
    Name(object &&o) \
    : Parent(check_(o) ? o.release().ptr() : ConvertFun(o.ptr()), stolen_t{}) \
    { if (!m_ptr) throw error_already_set(); } \
    template <typename Policy_> \
    Name(const ::pybind11::detail::accessor<Policy_> &a) : Name(object(a)) { }

#define PYBIND11_OBJECT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT_COMMON(Name, Parent, CheckFun) \
    /* This is deliberately not 'explicit' to allow implicit conversion from object: */ \
    Name(const object &o) : Parent(o) { } \
    Name(object &&o) : Parent(std::move(o)) { }

#define PYBIND11_OBJECT_DEFAULT(Name, Parent, CheckFun) \
    PYBIND11_OBJECT(Name, Parent, CheckFun) \
    Name() : Parent() { }

/// \addtogroup pytypes
/// @{

/** \rst
    Wraps a Python iterator so that it can also be used as a C++ input iterator

    Caveat: copying an iterator does not (and cannot) clone the internal
    state of the Python iterable. This also applies to the post-increment
    operator. This iterator should only be used to retrieve the current
    value using ``operator*()``.
\endrst */
class iterator : public object {
public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = ssize_t;
    using value_type = handle;
    using reference = const handle;
    using pointer = const handle *;

    PYBIND11_OBJECT_DEFAULT(iterator, object, PyIter_Check)

    iterator& operator++() {
        advance();
        return *this;
    }

    iterator operator++(int) {
        auto rv = *this;
        advance();
        return rv;
    }

    reference operator*() const {
        if (m_ptr && !value.ptr()) {
            auto& self = const_cast<iterator &>(*this);
            self.advance();
        }
        return value;
    }

    pointer operator->() const { operator*(); return &value; }

    /** \rst
         The value which marks the end of the iteration. ``it == iterator::sentinel()``
         is equivalent to catching ``StopIteration`` in Python.

         .. code-block:: cpp

             void foo(py::iterator it) {
                 while (it != py::iterator::sentinel()) {
                    // use `*it`
                    ++it;
                 }
             }
    \endrst */
    static iterator sentinel() { return {}; }

    friend bool operator==(const iterator &a, const iterator &b) { return a->ptr() == b->ptr(); }
    friend bool operator!=(const iterator &a, const iterator &b) { return a->ptr() != b->ptr(); }

private:
    void advance() {
        value = reinterpret_steal<object>(PyIter_Next(m_ptr));
        if (PyErr_Occurred()) { throw error_already_set(); }
    }

private:
    object value = {};
};

class iterable : public object {
public:
    PYBIND11_OBJECT_DEFAULT(iterable, object, detail::PyIterable_Check)
};

class bytes;

class str : public object {
public:
    PYBIND11_OBJECT_CVT(str, object, detail::PyUnicode_Check_Permissive, raw_str)

    str(const char *c, size_t n)
        : object(PyUnicode_FromStringAndSize(c, (ssize_t) n), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    // 'explicit' is explicitly omitted from the following constructors to allow implicit conversion to py::str from C++ string-like objects
    str(const char *c = "")
        : object(PyUnicode_FromString(c), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate string object!");
    }

    str(const std::string &s) : str(s.data(), s.size()) { }

    explicit str(const bytes &b);

    /** \rst
        Return a string representation of the object. This is analogous to
        the ``str()`` function in Python.
    \endrst */
    explicit str(handle h) : object(raw_str(h.ptr()), stolen_t{}) { }

    operator std::string() const {
        object temp = *this;
        if (PyUnicode_Check(m_ptr)) {
            temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(m_ptr));
            if (!temp)
                pybind11_fail("Unable to extract string contents! (encoding issue)");
        }
        char *buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
            pybind11_fail("Unable to extract string contents! (invalid type)");
        return std::string(buffer, (size_t) length);
    }

    template <typename... Args>
    str format(Args &&...args) const {
        return attr("format")(std::forward<Args>(args)...);
    }

private:
    /// Return string representation -- always returns a new reference, even if already a str
    static PyObject *raw_str(PyObject *op) {
        PyObject *str_value = PyObject_Str(op);
        if (!str_value) throw error_already_set();
#if PY_MAJOR_VERSION < 3
        PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
        Py_XDECREF(str_value); str_value = unicode;
#endif
        return str_value;
    }
};
/// @} pytypes

inline namespace literals {
/** \rst
    String literal version of `str`
 \endrst */
inline str operator"" _s(const char *s, size_t size) { return {s, size}; }
}

/// \addtogroup pytypes
/// @{
class bytes : public object {
public:
    PYBIND11_OBJECT(bytes, object, PYBIND11_BYTES_CHECK)

    // Allow implicit conversion:
    bytes(const char *c = "")
        : object(PYBIND11_BYTES_FROM_STRING(c), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    bytes(const char *c, size_t n)
        : object(PYBIND11_BYTES_FROM_STRING_AND_SIZE(c, (ssize_t) n), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
    }

    // Allow implicit conversion:
    bytes(const std::string &s) : bytes(s.data(), s.size()) { }

    explicit bytes(const pybind11::str &s);

    operator std::string() const {
        char *buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(m_ptr, &buffer, &length))
            pybind11_fail("Unable to extract bytes contents!");
        return std::string(buffer, (size_t) length);
    }
};
// Note: breathe >= 4.17.0 will fail to build docs if the below two constructors
// are included in the doxygen group; close here and reopen after as a workaround
/// @} pytypes

inline bytes::bytes(const pybind11::str &s) {
    object temp = s;
    if (PyUnicode_Check(s.ptr())) {
        temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(s.ptr()));
        if (!temp)
            pybind11_fail("Unable to extract string contents! (encoding issue)");
    }
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract string contents! (invalid type)");
    auto obj = reinterpret_steal<object>(PYBIND11_BYTES_FROM_STRING_AND_SIZE(buffer, length));
    if (!obj)
        pybind11_fail("Could not allocate bytes object!");
    m_ptr = obj.release().ptr();
}

inline str::str(const bytes& b) {
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract bytes contents!");
    auto obj = reinterpret_steal<object>(PyUnicode_FromStringAndSize(buffer, (ssize_t) length));
    if (!obj)
        pybind11_fail("Could not allocate string object!");
    m_ptr = obj.release().ptr();
}

/// \addtogroup pytypes
/// @{
class none : public object {
public:
    PYBIND11_OBJECT(none, object, detail::PyNone_Check)
    none() : object(Py_None, borrowed_t{}) { }
};

class ellipsis : public object {
public:
    PYBIND11_OBJECT(ellipsis, object, detail::PyEllipsis_Check)
    ellipsis() : object(Py_Ellipsis, borrowed_t{}) { }
};

class bool_ : public object {
public:
    PYBIND11_OBJECT_CVT(bool_, object, PyBool_Check, raw_bool)
    bool_() : object(Py_False, borrowed_t{}) { }
    // Allow implicit conversion from and to `bool`:
    bool_(bool value) : object(value ? Py_True : Py_False, borrowed_t{}) { }
    operator bool() const { return m_ptr && PyLong_AsLong(m_ptr) != 0; }

private:
    /// Return the truth value of an object -- always returns a new reference
    static PyObject *raw_bool(PyObject *op) {
        const auto value = PyObject_IsTrue(op);
        if (value == -1) return nullptr;
        return handle(value ? Py_True : Py_False).inc_ref().ptr();
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)
// Converts a value to the given unsigned type.  If an error occurs, you get back (Unsigned) -1;
// otherwise you get back the unsigned long or unsigned long long value cast to (Unsigned).
// (The distinction is critically important when casting a returned -1 error value to some other
// unsigned type: (A)-1 != (B)-1 when A and B are unsigned types of different sizes).
template <typename Unsigned>
Unsigned as_unsigned(PyObject *o) {
    if (sizeof(Unsigned) <= sizeof(unsigned long)
#if PY_VERSION_HEX < 0x03000000
            || PyInt_Check(o)
#endif
    ) {
        unsigned long v = PyLong_AsUnsignedLong(o);
        return v == (unsigned long) -1 && PyErr_Occurred() ? (Unsigned) -1 : (Unsigned) v;
    }
    else {
        unsigned long long v = PyLong_AsUnsignedLongLong(o);
        return v == (unsigned long long) -1 && PyErr_Occurred() ? (Unsigned) -1 : (Unsigned) v;
    }
}
PYBIND11_NAMESPACE_END(detail)

class int_ : public object {
public:
    PYBIND11_OBJECT_CVT(int_, object, PYBIND11_LONG_CHECK, PyNumber_Long)
    int_() : object(PyLong_FromLong(0), stolen_t{}) { }
    // Allow implicit conversion from C++ integral types:
    template <typename T,
              detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    int_(T value) {
        if (sizeof(T) <= sizeof(long)) {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLong((long) value);
            else
                m_ptr = PyLong_FromUnsignedLong((unsigned long) value);
        } else {
            if (std::is_signed<T>::value)
                m_ptr = PyLong_FromLongLong((long long) value);
            else
                m_ptr = PyLong_FromUnsignedLongLong((unsigned long long) value);
        }
        if (!m_ptr) pybind11_fail("Could not allocate int object!");
    }

    template <typename T,
              detail::enable_if_t<std::is_integral<T>::value, int> = 0>
    operator T() const {
        return std::is_unsigned<T>::value
            ? detail::as_unsigned<T>(m_ptr)
            : sizeof(T) <= sizeof(long)
              ? (T) PyLong_AsLong(m_ptr)
              : (T) PYBIND11_LONG_AS_LONGLONG(m_ptr);
    }
};

class float_ : public object {
public:
    PYBIND11_OBJECT_CVT(float_, object, PyFloat_Check, PyNumber_Float)
    // Allow implicit conversion from float/double:
    float_(float value) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    float_(double value = .0) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate float object!");
    }
    operator float() const { return (float) PyFloat_AsDouble(m_ptr); }
    operator double() const { return (double) PyFloat_AsDouble(m_ptr); }
};

class weakref : public object {
public:
    PYBIND11_OBJECT_DEFAULT(weakref, object, PyWeakref_Check)
    explicit weakref(handle obj, handle callback = {})
        : object(PyWeakref_NewRef(obj.ptr(), callback.ptr()), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate weak reference!");
    }
};

class slice : public object {
public:
    PYBIND11_OBJECT_DEFAULT(slice, object, PySlice_Check)
    slice(ssize_t start_, ssize_t stop_, ssize_t step_) {
        int_ start(start_), stop(stop_), step(step_);
        m_ptr = PySlice_New(start.ptr(), stop.ptr(), step.ptr());
        if (!m_ptr) pybind11_fail("Could not allocate slice object!");
    }
    bool compute(size_t length, size_t *start, size_t *stop, size_t *step,
                 size_t *slicelength) const {
        return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
                                    (ssize_t) length, (ssize_t *) start,
                                    (ssize_t *) stop, (ssize_t *) step,
                                    (ssize_t *) slicelength) == 0;
    }
    bool compute(ssize_t length, ssize_t *start, ssize_t *stop, ssize_t *step,
      ssize_t *slicelength) const {
      return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
          length, start,
          stop, step,
          slicelength) == 0;
    }
};

class capsule : public object {
public:
    PYBIND11_OBJECT_DEFAULT(capsule, object, PyCapsule_CheckExact)
    PYBIND11_DEPRECATED("Use reinterpret_borrow<capsule>() or reinterpret_steal<capsule>()")
    capsule(PyObject *ptr, bool is_borrowed) : object(is_borrowed ? object(ptr, borrowed_t{}) : object(ptr, stolen_t{})) { }

    explicit capsule(const void *value, const char *name = nullptr, void (*destructor)(PyObject *) = nullptr)
        : object(PyCapsule_New(const_cast<void *>(value), name, destructor), stolen_t{}) {
        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");
    }

    PYBIND11_DEPRECATED("Please pass a destructor that takes a void pointer as input")
    capsule(const void *value, void (*destruct)(PyObject *))
        : object(PyCapsule_New(const_cast<void*>(value), nullptr, destruct), stolen_t{}) {
        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");
    }

    capsule(const void *value, void (*destructor)(void *)) {
        m_ptr = PyCapsule_New(const_cast<void *>(value), nullptr, [](PyObject *o) {
            auto destructor = reinterpret_cast<void (*)(void *)>(PyCapsule_GetContext(o));
            void *ptr = PyCapsule_GetPointer(o, nullptr);
            destructor(ptr);
        });

        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");

        if (PyCapsule_SetContext(m_ptr, (void *) destructor) != 0)
            pybind11_fail("Could not set capsule context!");
    }

    capsule(void (*destructor)()) {
        m_ptr = PyCapsule_New(reinterpret_cast<void *>(destructor), nullptr, [](PyObject *o) {
            auto destructor = reinterpret_cast<void (*)()>(PyCapsule_GetPointer(o, nullptr));
            destructor();
        });

        if (!m_ptr)
            pybind11_fail("Could not allocate capsule object!");
    }

    template <typename T> operator T *() const {
        auto name = this->name();
        T * result = static_cast<T *>(PyCapsule_GetPointer(m_ptr, name));
        if (!result) pybind11_fail("Unable to extract capsule contents!");
        return result;
    }

    const char *name() const { return PyCapsule_GetName(m_ptr); }
};

class tuple : public object {
public:
    PYBIND11_OBJECT_CVT(tuple, object, PyTuple_Check, PySequence_Tuple)
    explicit tuple(size_t size = 0) : object(PyTuple_New((ssize_t) size), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate tuple object!");
    }
    size_t size() const { return (size_t) PyTuple_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    detail::tuple_accessor operator[](size_t index) const { return {*this, index}; }
    detail::item_accessor operator[](handle h) const { return object::operator[](h); }
    detail::tuple_iterator begin() const { return {*this, 0}; }
    detail::tuple_iterator end() const { return {*this, PyTuple_GET_SIZE(m_ptr)}; }
};

class dict : public object {
public:
    PYBIND11_OBJECT_CVT(dict, object, PyDict_Check, raw_dict)
    dict() : object(PyDict_New(), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate dict object!");
    }
    template <typename... Args,
              typename = detail::enable_if_t<detail::all_of<detail::is_keyword_or_ds<Args>...>::value>,
              // MSVC workaround: it can't compile an out-of-line definition, so defer the collector
              typename collector = detail::deferred_t<detail::unpacking_collector<>, Args...>>
    explicit dict(Args &&...args) : dict(collector(std::forward<Args>(args)...).kwargs()) { }

    size_t size() const { return (size_t) PyDict_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    detail::dict_iterator begin() const { return {*this, 0}; }
    detail::dict_iterator end() const { return {}; }
    void clear() const { PyDict_Clear(ptr()); }
    template <typename T> bool contains(T &&key) const {
        return PyDict_Contains(m_ptr, detail::object_or_cast(std::forward<T>(key)).ptr()) == 1;
    }

private:
    /// Call the `dict` Python type -- always returns a new reference
    static PyObject *raw_dict(PyObject *op) {
        if (PyDict_Check(op))
            return handle(op).inc_ref().ptr();
        return PyObject_CallFunctionObjArgs((PyObject *) &PyDict_Type, op, nullptr);
    }
};

class sequence : public object {
public:
    PYBIND11_OBJECT_DEFAULT(sequence, object, PySequence_Check)
    size_t size() const {
        ssize_t result = PySequence_Size(m_ptr);
        if (result == -1)
            throw error_already_set();
        return (size_t) result;
    }
    bool empty() const { return size() == 0; }
    detail::sequence_accessor operator[](size_t index) const { return {*this, index}; }
    detail::item_accessor operator[](handle h) const { return object::operator[](h); }
    detail::sequence_iterator begin() const { return {*this, 0}; }
    detail::sequence_iterator end() const { return {*this, PySequence_Size(m_ptr)}; }
};

class list : public object {
public:
    PYBIND11_OBJECT_CVT(list, object, PyList_Check, PySequence_List)
    explicit list(size_t size = 0) : object(PyList_New((ssize_t) size), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate list object!");
    }
    size_t size() const { return (size_t) PyList_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    detail::list_accessor operator[](size_t index) const { return {*this, index}; }
    detail::item_accessor operator[](handle h) const { return object::operator[](h); }
    detail::list_iterator begin() const { return {*this, 0}; }
    detail::list_iterator end() const { return {*this, PyList_GET_SIZE(m_ptr)}; }
    template <typename T> void append(T &&val) const {
        PyList_Append(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr());
    }
    template <typename T> void insert(size_t index, T &&val) const {
        PyList_Insert(m_ptr, static_cast<ssize_t>(index),
            detail::object_or_cast(std::forward<T>(val)).ptr());
    }
};

class args : public tuple { PYBIND11_OBJECT_DEFAULT(args, tuple, PyTuple_Check) };
class kwargs : public dict { PYBIND11_OBJECT_DEFAULT(kwargs, dict, PyDict_Check)  };

class set : public object {
public:
    PYBIND11_OBJECT_CVT(set, object, PySet_Check, PySet_New)
    set() : object(PySet_New(nullptr), stolen_t{}) {
        if (!m_ptr) pybind11_fail("Could not allocate set object!");
    }
    size_t size() const { return (size_t) PySet_Size(m_ptr); }
    bool empty() const { return size() == 0; }
    template <typename T> bool add(T &&val) const {
        return PySet_Add(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr()) == 0;
    }
    void clear() const { PySet_Clear(m_ptr); }
    template <typename T> bool contains(T &&val) const {
        return PySet_Contains(m_ptr, detail::object_or_cast(std::forward<T>(val)).ptr()) == 1;
    }
};

class function : public object {
public:
    PYBIND11_OBJECT_DEFAULT(function, object, PyCallable_Check)
    handle cpp_function() const {
        handle fun = detail::get_function(m_ptr);
        if (fun && PyCFunction_Check(fun.ptr()))
            return fun;
        return handle();
    }
    bool is_cpp_function() const { return (bool) cpp_function(); }
};

class staticmethod : public object {
public:
    PYBIND11_OBJECT_CVT(staticmethod, object, detail::PyStaticMethod_Check, PyStaticMethod_New)
};

class buffer : public object {
public:
    PYBIND11_OBJECT_DEFAULT(buffer, object, PyObject_CheckBuffer)

    buffer_info request(bool writable = false) const {
        int flags = PyBUF_STRIDES | PyBUF_FORMAT;
        if (writable) flags |= PyBUF_WRITABLE;
        Py_buffer *view = new Py_buffer();
        if (PyObject_GetBuffer(m_ptr, view, flags) != 0) {
            delete view;
            throw error_already_set();
        }
        return buffer_info(view);
    }
};

class memoryview : public object {
public:
    PYBIND11_OBJECT_CVT(memoryview, object, PyMemoryView_Check, PyMemoryView_FromObject)

    /** \rst
        Creates ``memoryview`` from ``buffer_info``.

        ``buffer_info`` must be created from ``buffer::request()``. Otherwise
        throws an exception.

        For creating a ``memoryview`` from objects that support buffer protocol,
        use ``memoryview(const object& obj)`` instead of this constructor.
     \endrst */
    explicit memoryview(const buffer_info& info) {
        if (!info.view())
            pybind11_fail("Prohibited to create memoryview without Py_buffer");
        // Note: PyMemoryView_FromBuffer never increments obj reference.
        m_ptr = (info.view()->obj) ?
            PyMemoryView_FromObject(info.view()->obj) :
            PyMemoryView_FromBuffer(info.view());
        if (!m_ptr)
            pybind11_fail("Unable to create memoryview from buffer descriptor");
    }

    /** \rst
        Creates ``memoryview`` from static buffer.

        This method is meant for providing a ``memoryview`` for C/C++ buffer not
        managed by Python. The caller is responsible for managing the lifetime
        of ``ptr`` and ``format``, which MUST outlive the memoryview constructed
        here.

        See also: Python C API documentation for `PyMemoryView_FromBuffer`_.

        .. _PyMemoryView_FromBuffer: https://docs.python.org/c-api/memoryview.html#c.PyMemoryView_FromBuffer

        :param ptr: Pointer to the buffer.
        :param itemsize: Byte size of an element.
        :param format: Pointer to the null-terminated format string. For
            homogeneous Buffers, this should be set to
            ``format_descriptor<T>::value``.
        :param shape: Shape of the tensor (1 entry per dimension).
        :param strides: Number of bytes between adjacent entries (for each
            per dimension).
        :param readonly: Flag to indicate if the underlying storage may be
            written to.
     \endrst */
    static memoryview from_buffer(
        void *ptr, ssize_t itemsize, const char *format,
        detail::any_container<ssize_t> shape,
        detail::any_container<ssize_t> strides, bool readonly = false);

    static memoryview from_buffer(
        const void *ptr, ssize_t itemsize, const char *format,
        detail::any_container<ssize_t> shape,
        detail::any_container<ssize_t> strides) {
        return memoryview::from_buffer(
            const_cast<void*>(ptr), itemsize, format, shape, strides, true);
    }

    template<typename T>
    static memoryview from_buffer(
        T *ptr, detail::any_container<ssize_t> shape,
        detail::any_container<ssize_t> strides, bool readonly = false) {
        return memoryview::from_buffer(
            reinterpret_cast<void*>(ptr), sizeof(T),
            format_descriptor<T>::value, shape, strides, readonly);
    }

    template<typename T>
    static memoryview from_buffer(
        const T *ptr, detail::any_container<ssize_t> shape,
        detail::any_container<ssize_t> strides) {
        return memoryview::from_buffer(
            const_cast<T*>(ptr), shape, strides, true);
    }

#if PY_MAJOR_VERSION >= 3
    /** \rst
        Creates ``memoryview`` from static memory.

        This method is meant for providing a ``memoryview`` for C/C++ buffer not
        managed by Python. The caller is responsible for managing the lifetime
        of ``mem``, which MUST outlive the memoryview constructed here.

        This method is not available in Python 2.

        See also: Python C API documentation for `PyMemoryView_FromBuffer`_.

        .. _PyMemoryView_FromMemory: https://docs.python.org/c-api/memoryview.html#c.PyMemoryView_FromMemory
     \endrst */
    static memoryview from_memory(void *mem, ssize_t size, bool readonly = false) {
        PyObject* ptr = PyMemoryView_FromMemory(
            reinterpret_cast<char*>(mem), size,
            (readonly) ? PyBUF_READ : PyBUF_WRITE);
        if (!ptr)
            pybind11_fail("Could not allocate memoryview object!");
        return memoryview(object(ptr, stolen_t{}));
    }

    static memoryview from_memory(const void *mem, ssize_t size) {
        return memoryview::from_memory(const_cast<void*>(mem), size, true);
    }
#endif
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
inline memoryview memoryview::from_buffer(
    void *ptr, ssize_t itemsize, const char* format,
    detail::any_container<ssize_t> shape,
    detail::any_container<ssize_t> strides, bool readonly) {
    size_t ndim = shape->size();
    if (ndim != strides->size())
        pybind11_fail("memoryview: shape length doesn't match strides length");
    ssize_t size = ndim ? 1 : 0;
    for (size_t i = 0; i < ndim; ++i)
        size *= (*shape)[i];
    Py_buffer view;
    view.buf = ptr;
    view.obj = nullptr;
    view.len = size * itemsize;
    view.readonly = static_cast<int>(readonly);
    view.itemsize = itemsize;
    view.format = const_cast<char*>(format);
    view.ndim = static_cast<int>(ndim);
    view.shape = shape->data();
    view.strides = strides->data();
    view.suboffsets = nullptr;
    view.internal = nullptr;
    PyObject* obj = PyMemoryView_FromBuffer(&view);
    if (!obj)
        throw error_already_set();
    return memoryview(object(obj, stolen_t{}));
}
#endif  // DOXYGEN_SHOULD_SKIP_THIS
/// @} pytypes

/// \addtogroup python_builtins
/// @{
inline size_t len(handle h) {
    ssize_t result = PyObject_Length(h.ptr());
    if (result < 0)
        pybind11_fail("Unable to compute length of object");
    return (size_t) result;
}

inline size_t len_hint(handle h) {
#if PY_VERSION_HEX >= 0x03040000
    ssize_t result = PyObject_LengthHint(h.ptr(), 0);
#else
    ssize_t result = PyObject_Length(h.ptr());
#endif
    if (result < 0) {
        // Sometimes a length can't be determined at all (eg generators)
        // In which case simply return 0
        PyErr_Clear();
        return 0;
    }
    return (size_t) result;
}

inline str repr(handle h) {
    PyObject *str_value = PyObject_Repr(h.ptr());
    if (!str_value) throw error_already_set();
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
    Py_XDECREF(str_value); str_value = unicode;
    if (!str_value) throw error_already_set();
#endif
    return reinterpret_steal<str>(str_value);
}

inline iterator iter(handle obj) {
    PyObject *result = PyObject_GetIter(obj.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<iterator>(result);
}
/// @} python_builtins

PYBIND11_NAMESPACE_BEGIN(detail)
template <typename D> iterator object_api<D>::begin() const { return iter(derived()); }
template <typename D> iterator object_api<D>::end() const { return iterator::sentinel(); }
template <typename D> item_accessor object_api<D>::operator[](handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D> item_accessor object_api<D>::operator[](const char *key) const {
    return {derived(), pybind11::str(key)};
}
template <typename D> obj_attr_accessor object_api<D>::attr(handle key) const {
    return {derived(), reinterpret_borrow<object>(key)};
}
template <typename D> str_attr_accessor object_api<D>::attr(const char *key) const {
    return {derived(), key};
}
template <typename D> args_proxy object_api<D>::operator*() const {
    return args_proxy(derived().ptr());
}
template <typename D> template <typename T> bool object_api<D>::contains(T &&item) const {
    return attr("__contains__")(std::forward<T>(item)).template cast<bool>();
}

template <typename D>
pybind11::str object_api<D>::str() const { return pybind11::str(derived()); }

template <typename D>
str_attr_accessor object_api<D>::doc() const { return attr("__doc__"); }

template <typename D>
handle object_api<D>::get_type() const { return (PyObject *) Py_TYPE(derived().ptr()); }

template <typename D>
bool object_api<D>::rich_compare(object_api const &other, int value) const {
    int rv = PyObject_RichCompareBool(derived().ptr(), other.derived().ptr(), value);
    if (rv == -1)
        throw error_already_set();
    return rv == 1;
}

#define PYBIND11_MATH_OPERATOR_UNARY(op, fn)                                   \
    template <typename D> object object_api<D>::op() const {                   \
        object result = reinterpret_steal<object>(fn(derived().ptr()));        \
        if (!result.ptr())                                                     \
            throw error_already_set();                                         \
        return result;                                                         \
    }

#define PYBIND11_MATH_OPERATOR_BINARY(op, fn)                                  \
    template <typename D>                                                      \
    object object_api<D>::op(object_api const &other) const {                  \
        object result = reinterpret_steal<object>(                             \
            fn(derived().ptr(), other.derived().ptr()));                       \
        if (!result.ptr())                                                     \
            throw error_already_set();                                         \
        return result;                                                         \
    }

PYBIND11_MATH_OPERATOR_UNARY (operator~,   PyNumber_Invert)
PYBIND11_MATH_OPERATOR_UNARY (operator-,   PyNumber_Negative)
PYBIND11_MATH_OPERATOR_BINARY(operator+,   PyNumber_Add)
PYBIND11_MATH_OPERATOR_BINARY(operator+=,  PyNumber_InPlaceAdd)
PYBIND11_MATH_OPERATOR_BINARY(operator-,   PyNumber_Subtract)
PYBIND11_MATH_OPERATOR_BINARY(operator-=,  PyNumber_InPlaceSubtract)
PYBIND11_MATH_OPERATOR_BINARY(operator*,   PyNumber_Multiply)
PYBIND11_MATH_OPERATOR_BINARY(operator*=,  PyNumber_InPlaceMultiply)
PYBIND11_MATH_OPERATOR_BINARY(operator/,   PyNumber_TrueDivide)
PYBIND11_MATH_OPERATOR_BINARY(operator/=,  PyNumber_InPlaceTrueDivide)
PYBIND11_MATH_OPERATOR_BINARY(operator|,   PyNumber_Or)
PYBIND11_MATH_OPERATOR_BINARY(operator|=,  PyNumber_InPlaceOr)
PYBIND11_MATH_OPERATOR_BINARY(operator&,   PyNumber_And)
PYBIND11_MATH_OPERATOR_BINARY(operator&=,  PyNumber_InPlaceAnd)
PYBIND11_MATH_OPERATOR_BINARY(operator^,   PyNumber_Xor)
PYBIND11_MATH_OPERATOR_BINARY(operator^=,  PyNumber_InPlaceXor)
PYBIND11_MATH_OPERATOR_BINARY(operator<<,  PyNumber_Lshift)
PYBIND11_MATH_OPERATOR_BINARY(operator<<=, PyNumber_InPlaceLshift)
PYBIND11_MATH_OPERATOR_BINARY(operator>>,  PyNumber_Rshift)
PYBIND11_MATH_OPERATOR_BINARY(operator>>=, PyNumber_InPlaceRshift)

#undef PYBIND11_MATH_OPERATOR_UNARY
#undef PYBIND11_MATH_OPERATOR_BINARY

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "pytypes.h" ============


//========= begin of #include "detail/typeid.h" ============

/*
    pybind11/detail/typeid.h: Compiler-independent access to type identifiers

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

// ans ignore: #include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
/// Erase all occurrences of a substring
inline void erase_all(std::string &string, const std::string &search) {
    for (size_t pos = 0;;) {
        pos = string.find(search, pos);
        if (pos == std::string::npos) break;
        string.erase(pos, search.length());
    }
}

PYBIND11_NOINLINE inline void clean_type_id(std::string &name) {
#if defined(__GNUG__)
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> res {
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free };
    if (status == 0)
        name = res.get();
#else
    detail::erase_all(name, "class ");
    detail::erase_all(name, "struct ");
    detail::erase_all(name, "enum ");
#endif
    detail::erase_all(name, "pybind11::");
}
PYBIND11_NAMESPACE_END(detail)

/// Return a string representation of a C++ type
template <typename T> static std::string type_id() {
    std::string name(typeid(T).name());
    detail::clean_type_id(name);
    return name;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "detail/typeid.h" ============


//========= begin of #include "detail/descr.h" ============

/*
    pybind11/detail/descr.h: Helper type for concatenating type signatures at compile time

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#if !defined(_MSC_VER)
#  define PYBIND11_DESCR_CONSTEXPR static constexpr
#else
#  define PYBIND11_DESCR_CONSTEXPR const
#endif

/* Concatenate type signatures at compile time */
template <size_t N, typename... Ts>
struct descr {
    char text[N + 1];

    constexpr descr() : text{'\0'} { }
    constexpr descr(char const (&s)[N+1]) : descr(s, make_index_sequence<N>()) { }

    template <size_t... Is>
    constexpr descr(char const (&s)[N+1], index_sequence<Is...>) : text{s[Is]..., '\0'} { }

    template <typename... Chars>
    constexpr descr(char c, Chars... cs) : text{c, static_cast<char>(cs)..., '\0'} { }

    static constexpr std::array<const std::type_info *, sizeof...(Ts) + 1> types() {
        return {{&typeid(Ts)..., nullptr}};
    }
};

template <size_t N1, size_t N2, typename... Ts1, typename... Ts2, size_t... Is1, size_t... Is2>
constexpr descr<N1 + N2, Ts1..., Ts2...> plus_impl(const descr<N1, Ts1...> &a, const descr<N2, Ts2...> &b,
                                                   index_sequence<Is1...>, index_sequence<Is2...>) {
    return {a.text[Is1]..., b.text[Is2]...};
}

template <size_t N1, size_t N2, typename... Ts1, typename... Ts2>
constexpr descr<N1 + N2, Ts1..., Ts2...> operator+(const descr<N1, Ts1...> &a, const descr<N2, Ts2...> &b) {
    return plus_impl(a, b, make_index_sequence<N1>(), make_index_sequence<N2>());
}

template <size_t N>
constexpr descr<N - 1> _(char const(&text)[N]) { return descr<N - 1>(text); }
constexpr descr<0> _(char const(&)[1]) { return {}; }

template <size_t Rem, size_t... Digits> struct int_to_str : int_to_str<Rem/10, Rem%10, Digits...> { };
template <size_t...Digits> struct int_to_str<0, Digits...> {
    static constexpr auto digits = descr<sizeof...(Digits)>(('0' + Digits)...);
};

// Ternary description (like std::conditional)
template <bool B, size_t N1, size_t N2>
constexpr enable_if_t<B, descr<N1 - 1>> _(char const(&text1)[N1], char const(&)[N2]) {
    return _(text1);
}
template <bool B, size_t N1, size_t N2>
constexpr enable_if_t<!B, descr<N2 - 1>> _(char const(&)[N1], char const(&text2)[N2]) {
    return _(text2);
}

template <bool B, typename T1, typename T2>
constexpr enable_if_t<B, T1> _(const T1 &d, const T2 &) { return d; }
template <bool B, typename T1, typename T2>
constexpr enable_if_t<!B, T2> _(const T1 &, const T2 &d) { return d; }

template <size_t Size> auto constexpr _() -> decltype(int_to_str<Size / 10, Size % 10>::digits) {
    return int_to_str<Size / 10, Size % 10>::digits;
}

template <typename Type> constexpr descr<1, Type> _() { return {'%'}; }

constexpr descr<0> concat() { return {}; }

template <size_t N, typename... Ts>
constexpr descr<N, Ts...> concat(const descr<N, Ts...> &descr) { return descr; }

template <size_t N, typename... Ts, typename... Args>
constexpr auto concat(const descr<N, Ts...> &d, const Args &...args)
    -> decltype(std::declval<descr<N + 2, Ts...>>() + concat(args...)) {
    return d + _(", ") + concat(args...);
}

template <size_t N, typename... Ts>
constexpr descr<N + 2, Ts...> type_descr(const descr<N, Ts...> &descr) {
    return _("{") + descr + _("}");
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "detail/descr.h" ============


//========= begin of #include "detail/internals.h" ============

/*
    pybind11/detail/internals.h: Internal data structure and related functions

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "../pytypes.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
// Forward declarations
inline PyTypeObject *make_static_property_type();
inline PyTypeObject *make_default_metaclass();
inline PyObject *make_object_base_type(PyTypeObject *metaclass);

// The old Python Thread Local Storage (TLS) API is deprecated in Python 3.7 in favor of the new
// Thread Specific Storage (TSS) API.
#if PY_VERSION_HEX >= 0x03070000
#    define PYBIND11_TLS_KEY_INIT(var) Py_tss_t *var = nullptr
#    define PYBIND11_TLS_GET_VALUE(key) PyThread_tss_get((key))
#    define PYBIND11_TLS_REPLACE_VALUE(key, value) PyThread_tss_set((key), (value))
#    define PYBIND11_TLS_DELETE_VALUE(key) PyThread_tss_set((key), nullptr)
#    define PYBIND11_TLS_FREE(key) PyThread_tss_free(key)
#else
    // Usually an int but a long on Cygwin64 with Python 3.x
#    define PYBIND11_TLS_KEY_INIT(var) decltype(PyThread_create_key()) var = 0
#    define PYBIND11_TLS_GET_VALUE(key) PyThread_get_key_value((key))
#    if PY_MAJOR_VERSION < 3
#        define PYBIND11_TLS_DELETE_VALUE(key)                               \
             PyThread_delete_key_value(key)
#        define PYBIND11_TLS_REPLACE_VALUE(key, value)                       \
             do {                                                            \
                 PyThread_delete_key_value((key));                           \
                 PyThread_set_key_value((key), (value));                     \
             } while (false)
#    else
#        define PYBIND11_TLS_DELETE_VALUE(key)                               \
             PyThread_set_key_value((key), nullptr)
#        define PYBIND11_TLS_REPLACE_VALUE(key, value)                       \
             PyThread_set_key_value((key), (value))
#    endif
#    define PYBIND11_TLS_FREE(key) (void)key
#endif

// Python loads modules by default with dlopen with the RTLD_LOCAL flag; under libc++ and possibly
// other STLs, this means `typeid(A)` from one module won't equal `typeid(A)` from another module
// even when `A` is the same, non-hidden-visibility type (e.g. from a common include).  Under
// libstdc++, this doesn't happen: equality and the type_index hash are based on the type name,
// which works.  If not under a known-good stl, provide our own name-based hash and equality
// functions that use the type name.
#if defined(__GLIBCXX__)
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) { return lhs == rhs; }
using type_hash = std::hash<std::type_index>;
using type_equal_to = std::equal_to<std::type_index>;
#else
inline bool same_type(const std::type_info &lhs, const std::type_info &rhs) {
    return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
}

struct type_hash {
    size_t operator()(const std::type_index &t) const {
        size_t hash = 5381;
        const char *ptr = t.name();
        while (auto c = static_cast<unsigned char>(*ptr++))
            hash = (hash * 33) ^ c;
        return hash;
    }
};

struct type_equal_to {
    bool operator()(const std::type_index &lhs, const std::type_index &rhs) const {
        return lhs.name() == rhs.name() || std::strcmp(lhs.name(), rhs.name()) == 0;
    }
};
#endif

template <typename value_type>
using type_map = std::unordered_map<std::type_index, value_type, type_hash, type_equal_to>;

struct overload_hash {
    inline size_t operator()(const std::pair<const PyObject *, const char *>& v) const {
        size_t value = std::hash<const void *>()(v.first);
        value ^= std::hash<const void *>()(v.second)  + 0x9e3779b9 + (value<<6) + (value>>2);
        return value;
    }
};

/// Internal data structure used to track registered instances and types.
/// Whenever binary incompatible changes are made to this structure,
/// `PYBIND11_INTERNALS_VERSION` must be incremented.
struct internals {
    type_map<type_info *> registered_types_cpp; // std::type_index -> pybind11's type information
    std::unordered_map<PyTypeObject *, std::vector<type_info *>> registered_types_py; // PyTypeObject* -> base type_info(s)
    std::unordered_multimap<const void *, instance*> registered_instances; // void * -> instance*
    std::unordered_set<std::pair<const PyObject *, const char *>, overload_hash> inactive_overload_cache;
    type_map<std::vector<bool (*)(PyObject *, void *&)>> direct_conversions;
    std::unordered_map<const PyObject *, std::vector<PyObject *>> patients;
    std::forward_list<void (*) (std::exception_ptr)> registered_exception_translators;
    std::unordered_map<std::string, void *> shared_data; // Custom data to be shared across extensions
    std::vector<PyObject *> loader_patient_stack; // Used by `loader_life_support`
    std::forward_list<std::string> static_strings; // Stores the std::strings backing detail::c_str()
    PyTypeObject *static_property_type;
    PyTypeObject *default_metaclass;
    PyObject *instance_base;
#if defined(WITH_THREAD)
    PYBIND11_TLS_KEY_INIT(tstate);
    PyInterpreterState *istate = nullptr;
    ~internals() {
        // This destructor is called *after* Py_Finalize() in finalize_interpreter().
        // That *SHOULD BE* fine. The following details what happens whe PyThread_tss_free is called.
        // PYBIND11_TLS_FREE is PyThread_tss_free on python 3.7+. On older python, it does nothing.
        // PyThread_tss_free calls PyThread_tss_delete and PyMem_RawFree.
        // PyThread_tss_delete just calls TlsFree (on Windows) or pthread_key_delete (on *NIX). Neither
        // of those have anything to do with CPython internals.
        // PyMem_RawFree *requires* that the `tstate` be allocated with the CPython allocator.
        PYBIND11_TLS_FREE(tstate);
    }
#endif
};

/// Additional type information which does not fit into the PyTypeObject.
/// Changes to this struct also require bumping `PYBIND11_INTERNALS_VERSION`.
struct type_info {
    PyTypeObject *type;
    const std::type_info *cpptype;
    size_t type_size, type_align, holder_size_in_ptrs;
    void *(*operator_new)(size_t);
    void (*init_instance)(instance *, const void *);
    void (*dealloc)(value_and_holder &v_h);
    std::vector<PyObject *(*)(PyObject *, PyTypeObject *)> implicit_conversions;
    std::vector<std::pair<const std::type_info *, void *(*)(void *)>> implicit_casts;
    std::vector<bool (*)(PyObject *, void *&)> *direct_conversions;
    buffer_info *(*get_buffer)(PyObject *, void *) = nullptr;
    void *get_buffer_data = nullptr;
    void *(*module_local_load)(PyObject *, const type_info *) = nullptr;
    /* A simple type never occurs as a (direct or indirect) parent
     * of a class that makes use of multiple inheritance */
    bool simple_type : 1;
    /* True if there is no multiple inheritance in this type's inheritance tree */
    bool simple_ancestors : 1;
    /* for base vs derived holder_type checks */
    bool default_holder : 1;
    /* true if this is a type registered with py::module_local */
    bool module_local : 1;
};

/// Tracks the `internals` and `type_info` ABI version independent of the main library version
#define PYBIND11_INTERNALS_VERSION 4

/// On MSVC, debug and release builds are not ABI-compatible!
#if defined(_MSC_VER) && defined(_DEBUG)
#   define PYBIND11_BUILD_TYPE "_debug"
#else
#   define PYBIND11_BUILD_TYPE ""
#endif

/// Let's assume that different compilers are ABI-incompatible.
#if defined(_MSC_VER)
#   define PYBIND11_COMPILER_TYPE "_msvc"
#elif defined(__INTEL_COMPILER)
#   define PYBIND11_COMPILER_TYPE "_icc"
#elif defined(__clang__)
#   define PYBIND11_COMPILER_TYPE "_clang"
#elif defined(__PGI)
#   define PYBIND11_COMPILER_TYPE "_pgi"
#elif defined(__MINGW32__)
#   define PYBIND11_COMPILER_TYPE "_mingw"
#elif defined(__CYGWIN__)
#   define PYBIND11_COMPILER_TYPE "_gcc_cygwin"
#elif defined(__GNUC__)
#   define PYBIND11_COMPILER_TYPE "_gcc"
#else
#   define PYBIND11_COMPILER_TYPE "_unknown"
#endif

#if defined(_LIBCPP_VERSION)
#  define PYBIND11_STDLIB "_libcpp"
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#  define PYBIND11_STDLIB "_libstdcpp"
#else
#  define PYBIND11_STDLIB ""
#endif

/// On Linux/OSX, changes in __GXX_ABI_VERSION__ indicate ABI incompatibility.
#if defined(__GXX_ABI_VERSION)
#  define PYBIND11_BUILD_ABI "_cxxabi" PYBIND11_TOSTRING(__GXX_ABI_VERSION)
#else
#  define PYBIND11_BUILD_ABI ""
#endif

#if defined(WITH_THREAD)
#  define PYBIND11_INTERNALS_KIND ""
#else
#  define PYBIND11_INTERNALS_KIND "_without_thread"
#endif

#define PYBIND11_INTERNALS_ID "__pybind11_internals_v" \
    PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) PYBIND11_INTERNALS_KIND PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI PYBIND11_BUILD_TYPE "__"

#define PYBIND11_MODULE_LOCAL_ID "__pybind11_module_local_v" \
    PYBIND11_TOSTRING(PYBIND11_INTERNALS_VERSION) PYBIND11_INTERNALS_KIND PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI PYBIND11_BUILD_TYPE "__"

/// Each module locally stores a pointer to the `internals` data. The data
/// itself is shared among modules with the same `PYBIND11_INTERNALS_ID`.
inline internals **&get_internals_pp() {
    static internals **internals_pp = nullptr;
    return internals_pp;
}

inline void translate_exception(std::exception_ptr p) {
    try {
        if (p) std::rethrow_exception(p);
    } catch (error_already_set &e)           { e.restore();                                    return;
    } catch (const builtin_exception &e)     { e.set_error();                                  return;
    } catch (const std::bad_alloc &e)        { PyErr_SetString(PyExc_MemoryError,   e.what()); return;
    } catch (const std::domain_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
    } catch (const std::invalid_argument &e) { PyErr_SetString(PyExc_ValueError,    e.what()); return;
    } catch (const std::length_error &e)     { PyErr_SetString(PyExc_ValueError,    e.what()); return;
    } catch (const std::out_of_range &e)     { PyErr_SetString(PyExc_IndexError,    e.what()); return;
    } catch (const std::range_error &e)      { PyErr_SetString(PyExc_ValueError,    e.what()); return;
    } catch (const std::overflow_error &e)   { PyErr_SetString(PyExc_OverflowError, e.what()); return;
    } catch (const std::exception &e)        { PyErr_SetString(PyExc_RuntimeError,  e.what()); return;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Caught an unknown exception!");
        return;
    }
}

#if !defined(__GLIBCXX__)
inline void translate_local_exception(std::exception_ptr p) {
    try {
        if (p) std::rethrow_exception(p);
    } catch (error_already_set &e)       { e.restore();   return;
    } catch (const builtin_exception &e) { e.set_error(); return;
    }
}
#endif

/// Return a reference to the current `internals` data
PYBIND11_NOINLINE inline internals &get_internals() {
    auto **&internals_pp = get_internals_pp();
    if (internals_pp && *internals_pp)
        return **internals_pp;

    // Ensure that the GIL is held since we will need to make Python calls.
    // Cannot use py::gil_scoped_acquire here since that constructor calls get_internals.
    struct gil_scoped_acquire_local {
        gil_scoped_acquire_local() : state (PyGILState_Ensure()) {}
        ~gil_scoped_acquire_local() { PyGILState_Release(state); }
        const PyGILState_STATE state;
    } gil;

    constexpr auto *id = PYBIND11_INTERNALS_ID;
    auto builtins = handle(PyEval_GetBuiltins());
    if (builtins.contains(id) && isinstance<capsule>(builtins[id])) {
        internals_pp = static_cast<internals **>(capsule(builtins[id]));

        // We loaded builtins through python's builtins, which means that our `error_already_set`
        // and `builtin_exception` may be different local classes than the ones set up in the
        // initial exception translator, below, so add another for our local exception classes.
        //
        // libstdc++ doesn't require this (types there are identified only by name)
#if !defined(__GLIBCXX__)
        (*internals_pp)->registered_exception_translators.push_front(&translate_local_exception);
#endif
    } else {
        if (!internals_pp) internals_pp = new internals*();
        auto *&internals_ptr = *internals_pp;
        internals_ptr = new internals();
#if defined(WITH_THREAD)

        #if PY_VERSION_HEX < 0x03090000
                PyEval_InitThreads();
        #endif
        PyThreadState *tstate = PyThreadState_Get();
        #if PY_VERSION_HEX >= 0x03070000
            internals_ptr->tstate = PyThread_tss_alloc();
            if (!internals_ptr->tstate || PyThread_tss_create(internals_ptr->tstate))
                pybind11_fail("get_internals: could not successfully initialize the TSS key!");
            PyThread_tss_set(internals_ptr->tstate, tstate);
        #else
            internals_ptr->tstate = PyThread_create_key();
            if (internals_ptr->tstate == -1)
                pybind11_fail("get_internals: could not successfully initialize the TLS key!");
            PyThread_set_key_value(internals_ptr->tstate, tstate);
        #endif
        internals_ptr->istate = tstate->interp;
#endif
        builtins[id] = capsule(internals_pp);
        internals_ptr->registered_exception_translators.push_front(&translate_exception);
        internals_ptr->static_property_type = make_static_property_type();
        internals_ptr->default_metaclass = make_default_metaclass();
        internals_ptr->instance_base = make_object_base_type(internals_ptr->default_metaclass);
    }
    return **internals_pp;
}

/// Works like `internals.registered_types_cpp`, but for module-local registered types:
inline type_map<type_info *> &registered_local_types_cpp() {
    static type_map<type_info *> locals{};
    return locals;
}

/// Constructs a std::string with the given arguments, stores it in `internals`, and returns its
/// `c_str()`.  Such strings objects have a long storage duration -- the internal strings are only
/// cleared when the program exits or after interpreter shutdown (when embedding), and so are
/// suitable for c-style strings needed by Python internals (such as PyTypeObject's tp_name).
template <typename... Args>
const char *c_str(Args &&...args) {
    auto &strings = get_internals().static_strings;
    strings.emplace_front(std::forward<Args>(args)...);
    return strings.front().c_str();
}

PYBIND11_NAMESPACE_END(detail)

/// Returns a named pointer that is shared among all extension modules (using the same
/// pybind11 version) running in the current interpreter. Names starting with underscores
/// are reserved for internal usage. Returns `nullptr` if no matching entry was found.
inline PYBIND11_NOINLINE void *get_shared_data(const std::string &name) {
    auto &internals = detail::get_internals();
    auto it = internals.shared_data.find(name);
    return it != internals.shared_data.end() ? it->second : nullptr;
}

/// Set the shared data that can be later recovered by `get_shared_data()`.
inline PYBIND11_NOINLINE void *set_shared_data(const std::string &name, void *data) {
    detail::get_internals().shared_data[name] = data;
    return data;
}

/// Returns a typed reference to a shared data entry (by using `get_shared_data()`) if
/// such entry exists. Otherwise, a new object of default-constructible type `T` is
/// added to the shared data under the given name and a reference to it is returned.
template<typename T>
T &get_or_create_shared_data(const std::string &name) {
    auto &internals = detail::get_internals();
    auto it = internals.shared_data.find(name);
    T *ptr = (T *) (it != internals.shared_data.end() ? it->second : nullptr);
    if (!ptr) {
        ptr = new T();
        internals.shared_data[name] = ptr;
    }
    return *ptr;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "detail/internals.h" ============


//========= begin of #include "cast.h" ============

/*
    pybind11/cast.h: Partial template specializations to cast between
    C++ and Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pytypes.h"
// ans ignore: #include "detail/typeid.h"
// ans ignore: #include "detail/descr.h"
// ans ignore: #include "detail/internals.h"
#include <array>
#include <limits>
#include <tuple>
#include <type_traits>

#if defined(PYBIND11_CPP17)
#  if defined(__has_include)
#    if __has_include(<string_view>)
#      define PYBIND11_HAS_STRING_VIEW
#    endif
#  elif defined(_MSC_VER)
#    define PYBIND11_HAS_STRING_VIEW
#  endif
#endif
#ifdef PYBIND11_HAS_STRING_VIEW
#include <string_view>
#endif

#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
#  define PYBIND11_HAS_U8STRING
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// A life support system for temporary objects created by `type_caster::load()`.
/// Adding a patient will keep it alive up until the enclosing function returns.
class loader_life_support {
public:
    /// A new patient frame is created when a function is entered
    loader_life_support() {
        get_internals().loader_patient_stack.push_back(nullptr);
    }

    /// ... and destroyed after it returns
    ~loader_life_support() {
        auto &stack = get_internals().loader_patient_stack;
        if (stack.empty())
            pybind11_fail("loader_life_support: internal error");

        auto ptr = stack.back();
        stack.pop_back();
        Py_CLEAR(ptr);

        // A heuristic to reduce the stack's capacity (e.g. after long recursive calls)
        if (stack.capacity() > 16 && stack.size() != 0 && stack.capacity() / stack.size() > 2)
            stack.shrink_to_fit();
    }

    /// This can only be used inside a pybind11-bound function, either by `argument_loader`
    /// at argument preparation time or by `py::cast()` at execution time.
    PYBIND11_NOINLINE static void add_patient(handle h) {
        auto &stack = get_internals().loader_patient_stack;
        if (stack.empty())
            throw cast_error("When called outside a bound function, py::cast() cannot "
                             "do Python -> C++ conversions which require the creation "
                             "of temporary values");

        auto &list_ptr = stack.back();
        if (list_ptr == nullptr) {
            list_ptr = PyList_New(1);
            if (!list_ptr)
                pybind11_fail("loader_life_support: error allocating list");
            PyList_SET_ITEM(list_ptr, 0, h.inc_ref().ptr());
        } else {
            auto result = PyList_Append(list_ptr, h.ptr());
            if (result == -1)
                pybind11_fail("loader_life_support: error adding patient");
        }
    }
};

// Gets the cache entry for the given type, creating it if necessary.  The return value is the pair
// returned by emplace, i.e. an iterator for the entry and a bool set to `true` if the entry was
// just created.
inline std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type);

// Populates a just-created cache entry.
PYBIND11_NOINLINE inline void all_type_info_populate(PyTypeObject *t, std::vector<type_info *> &bases) {
    std::vector<PyTypeObject *> check;
    for (handle parent : reinterpret_borrow<tuple>(t->tp_bases))
        check.push_back((PyTypeObject *) parent.ptr());

    auto const &type_dict = get_internals().registered_types_py;
    for (size_t i = 0; i < check.size(); i++) {
        auto type = check[i];
        // Ignore Python2 old-style class super type:
        if (!PyType_Check((PyObject *) type)) continue;

        // Check `type` in the current set of registered python types:
        auto it = type_dict.find(type);
        if (it != type_dict.end()) {
            // We found a cache entry for it, so it's either pybind-registered or has pre-computed
            // pybind bases, but we have to make sure we haven't already seen the type(s) before: we
            // want to follow Python/virtual C++ rules that there should only be one instance of a
            // common base.
            for (auto *tinfo : it->second) {
                // NB: Could use a second set here, rather than doing a linear search, but since
                // having a large number of immediate pybind11-registered types seems fairly
                // unlikely, that probably isn't worthwhile.
                bool found = false;
                for (auto *known : bases) {
                    if (known == tinfo) { found = true; break; }
                }
                if (!found) bases.push_back(tinfo);
            }
        }
        else if (type->tp_bases) {
            // It's some python type, so keep follow its bases classes to look for one or more
            // registered types
            if (i + 1 == check.size()) {
                // When we're at the end, we can pop off the current element to avoid growing
                // `check` when adding just one base (which is typical--i.e. when there is no
                // multiple inheritance)
                check.pop_back();
                i--;
            }
            for (handle parent : reinterpret_borrow<tuple>(type->tp_bases))
                check.push_back((PyTypeObject *) parent.ptr());
        }
    }
}

/**
 * Extracts vector of type_info pointers of pybind-registered roots of the given Python type.  Will
 * be just 1 pybind type for the Python type of a pybind-registered class, or for any Python-side
 * derived class that uses single inheritance.  Will contain as many types as required for a Python
 * class that uses multiple inheritance to inherit (directly or indirectly) from multiple
 * pybind-registered classes.  Will be empty if neither the type nor any base classes are
 * pybind-registered.
 *
 * The value is cached for the lifetime of the Python type.
 */
inline const std::vector<detail::type_info *> &all_type_info(PyTypeObject *type) {
    auto ins = all_type_info_get_cache(type);
    if (ins.second)
        // New cache entry: populate it
        all_type_info_populate(type, ins.first->second);

    return ins.first->second;
}

/**
 * Gets a single pybind11 type info for a python type.  Returns nullptr if neither the type nor any
 * ancestors are pybind11-registered.  Throws an exception if there are multiple bases--use
 * `all_type_info` instead if you want to support multiple bases.
 */
PYBIND11_NOINLINE inline detail::type_info* get_type_info(PyTypeObject *type) {
    auto &bases = all_type_info(type);
    if (bases.size() == 0)
        return nullptr;
    if (bases.size() > 1)
        pybind11_fail("pybind11::detail::get_type_info: type has multiple pybind11-registered bases");
    return bases.front();
}

inline detail::type_info *get_local_type_info(const std::type_index &tp) {
    auto &locals = registered_local_types_cpp();
    auto it = locals.find(tp);
    if (it != locals.end())
        return it->second;
    return nullptr;
}

inline detail::type_info *get_global_type_info(const std::type_index &tp) {
    auto &types = get_internals().registered_types_cpp;
    auto it = types.find(tp);
    if (it != types.end())
        return it->second;
    return nullptr;
}

/// Return the type info for a given C++ type; on lookup failure can either throw or return nullptr.
PYBIND11_NOINLINE inline detail::type_info *get_type_info(const std::type_index &tp,
                                                          bool throw_if_missing = false) {
    if (auto ltype = get_local_type_info(tp))
        return ltype;
    if (auto gtype = get_global_type_info(tp))
        return gtype;

    if (throw_if_missing) {
        std::string tname = tp.name();
        detail::clean_type_id(tname);
        pybind11_fail("pybind11::detail::get_type_info: unable to find type info for \"" + tname + "\"");
    }
    return nullptr;
}

PYBIND11_NOINLINE inline handle get_type_handle(const std::type_info &tp, bool throw_if_missing) {
    detail::type_info *type_info = get_type_info(tp, throw_if_missing);
    return handle(type_info ? ((PyObject *) type_info->type) : nullptr);
}

struct value_and_holder {
    instance *inst = nullptr;
    size_t index = 0u;
    const detail::type_info *type = nullptr;
    void **vh = nullptr;

    // Main constructor for a found value/holder:
    value_and_holder(instance *i, const detail::type_info *type, size_t vpos, size_t index) :
        inst{i}, index{index}, type{type},
        vh{inst->simple_layout ? inst->simple_value_holder : &inst->nonsimple.values_and_holders[vpos]}
    {}

    // Default constructor (used to signal a value-and-holder not found by get_value_and_holder())
    value_and_holder() {}

    // Used for past-the-end iterator
    value_and_holder(size_t index) : index{index} {}

    template <typename V = void> V *&value_ptr() const {
        return reinterpret_cast<V *&>(vh[0]);
    }
    // True if this `value_and_holder` has a non-null value pointer
    explicit operator bool() const { return value_ptr(); }

    template <typename H> H &holder() const {
        return reinterpret_cast<H &>(vh[1]);
    }
    bool holder_constructed() const {
        return inst->simple_layout
            ? inst->simple_holder_constructed
            : inst->nonsimple.status[index] & instance::status_holder_constructed;
    }
    void set_holder_constructed(bool v = true) {
        if (inst->simple_layout)
            inst->simple_holder_constructed = v;
        else if (v)
            inst->nonsimple.status[index] |= instance::status_holder_constructed;
        else
            inst->nonsimple.status[index] &= (uint8_t) ~instance::status_holder_constructed;
    }
    bool instance_registered() const {
        return inst->simple_layout
            ? inst->simple_instance_registered
            : inst->nonsimple.status[index] & instance::status_instance_registered;
    }
    void set_instance_registered(bool v = true) {
        if (inst->simple_layout)
            inst->simple_instance_registered = v;
        else if (v)
            inst->nonsimple.status[index] |= instance::status_instance_registered;
        else
            inst->nonsimple.status[index] &= (uint8_t) ~instance::status_instance_registered;
    }
};

// Container for accessing and iterating over an instance's values/holders
struct values_and_holders {
private:
    instance *inst;
    using type_vec = std::vector<detail::type_info *>;
    const type_vec &tinfo;

public:
    values_and_holders(instance *inst) : inst{inst}, tinfo(all_type_info(Py_TYPE(inst))) {}

    struct iterator {
    private:
        instance *inst = nullptr;
        const type_vec *types = nullptr;
        value_and_holder curr;
        friend struct values_and_holders;
        iterator(instance *inst, const type_vec *tinfo)
            : inst{inst}, types{tinfo},
            curr(inst /* instance */,
                 types->empty() ? nullptr : (*types)[0] /* type info */,
                 0, /* vpos: (non-simple types only): the first vptr comes first */
                 0 /* index */)
        {}
        // Past-the-end iterator:
        iterator(size_t end) : curr(end) {}
    public:
        bool operator==(const iterator &other) const { return curr.index == other.curr.index; }
        bool operator!=(const iterator &other) const { return curr.index != other.curr.index; }
        iterator &operator++() {
            if (!inst->simple_layout)
                curr.vh += 1 + (*types)[curr.index]->holder_size_in_ptrs;
            ++curr.index;
            curr.type = curr.index < types->size() ? (*types)[curr.index] : nullptr;
            return *this;
        }
        value_and_holder &operator*() { return curr; }
        value_and_holder *operator->() { return &curr; }
    };

    iterator begin() { return iterator(inst, &tinfo); }
    iterator end() { return iterator(tinfo.size()); }

    iterator find(const type_info *find_type) {
        auto it = begin(), endit = end();
        while (it != endit && it->type != find_type) ++it;
        return it;
    }

    size_t size() { return tinfo.size(); }
};

/**
 * Extracts C++ value and holder pointer references from an instance (which may contain multiple
 * values/holders for python-side multiple inheritance) that match the given type.  Throws an error
 * if the given type (or ValueType, if omitted) is not a pybind11 base of the given instance.  If
 * `find_type` is omitted (or explicitly specified as nullptr) the first value/holder are returned,
 * regardless of type (and the resulting .type will be nullptr).
 *
 * The returned object should be short-lived: in particular, it must not outlive the called-upon
 * instance.
 */
PYBIND11_NOINLINE inline value_and_holder instance::get_value_and_holder(const type_info *find_type /*= nullptr default in common.h*/, bool throw_if_missing /*= true in common.h*/) {
    // Optimize common case:
    if (!find_type || Py_TYPE(this) == find_type->type)
        return value_and_holder(this, find_type, 0, 0);

    detail::values_and_holders vhs(this);
    auto it = vhs.find(find_type);
    if (it != vhs.end())
        return *it;

    if (!throw_if_missing)
        return value_and_holder();

#if defined(NDEBUG)
    pybind11_fail("pybind11::detail::instance::get_value_and_holder: "
            "type is not a pybind11 base of the given instance "
            "(compile in debug mode for type details)");
#else
    pybind11_fail("pybind11::detail::instance::get_value_and_holder: `" +
            std::string(find_type->type->tp_name) + "' is not a pybind11 base of the given `" +
            std::string(Py_TYPE(this)->tp_name) + "' instance");
#endif
}

PYBIND11_NOINLINE inline void instance::allocate_layout() {
    auto &tinfo = all_type_info(Py_TYPE(this));

    const size_t n_types = tinfo.size();

    if (n_types == 0)
        pybind11_fail("instance allocation failed: new instance has no pybind11-registered base types");

    simple_layout =
        n_types == 1 && tinfo.front()->holder_size_in_ptrs <= instance_simple_holder_in_ptrs();

    // Simple path: no python-side multiple inheritance, and a small-enough holder
    if (simple_layout) {
        simple_value_holder[0] = nullptr;
        simple_holder_constructed = false;
        simple_instance_registered = false;
    }
    else { // multiple base types or a too-large holder
        // Allocate space to hold: [v1*][h1][v2*][h2]...[bb...] where [vN*] is a value pointer,
        // [hN] is the (uninitialized) holder instance for value N, and [bb...] is a set of bool
        // values that tracks whether each associated holder has been initialized.  Each [block] is
        // padded, if necessary, to an integer multiple of sizeof(void *).
        size_t space = 0;
        for (auto t : tinfo) {
            space += 1; // value pointer
            space += t->holder_size_in_ptrs; // holder instance
        }
        size_t flags_at = space;
        space += size_in_ptrs(n_types); // status bytes (holder_constructed and instance_registered)

        // Allocate space for flags, values, and holders, and initialize it to 0 (flags and values,
        // in particular, need to be 0).  Use Python's memory allocation functions: in Python 3.6
        // they default to using pymalloc, which is designed to be efficient for small allocations
        // like the one we're doing here; in earlier versions (and for larger allocations) they are
        // just wrappers around malloc.
#if PY_VERSION_HEX >= 0x03050000
        nonsimple.values_and_holders = (void **) PyMem_Calloc(space, sizeof(void *));
        if (!nonsimple.values_and_holders) throw std::bad_alloc();
#else
        nonsimple.values_and_holders = (void **) PyMem_New(void *, space);
        if (!nonsimple.values_and_holders) throw std::bad_alloc();
        std::memset(nonsimple.values_and_holders, 0, space * sizeof(void *));
#endif
        nonsimple.status = reinterpret_cast<uint8_t *>(&nonsimple.values_and_holders[flags_at]);
    }
    owned = true;
}

PYBIND11_NOINLINE inline void instance::deallocate_layout() {
    if (!simple_layout)
        PyMem_Free(nonsimple.values_and_holders);
}

PYBIND11_NOINLINE inline bool isinstance_generic(handle obj, const std::type_info &tp) {
    handle type = detail::get_type_handle(tp, false);
    if (!type)
        return false;
    return isinstance(obj, type);
}

PYBIND11_NOINLINE inline std::string error_string() {
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown internal error occurred");
        return "Unknown internal error occurred";
    }

    error_scope scope; // Preserve error state

    std::string errorString;
    if (scope.type) {
        errorString += handle(scope.type).attr("__name__").cast<std::string>();
        errorString += ": ";
    }
    if (scope.value)
        errorString += (std::string) str(scope.value);

    PyErr_NormalizeException(&scope.type, &scope.value, &scope.trace);

#if PY_MAJOR_VERSION >= 3
    if (scope.trace != nullptr)
        PyException_SetTraceback(scope.value, scope.trace);
#endif

#if !defined(PYPY_VERSION)
    if (scope.trace) {
        PyTracebackObject *trace = (PyTracebackObject *) scope.trace;

        /* Get the deepest trace possible */
        while (trace->tb_next)
            trace = trace->tb_next;

        PyFrameObject *frame = trace->tb_frame;
        errorString += "\n\nAt:\n";
        while (frame) {
            int lineno = PyFrame_GetLineNumber(frame);
            errorString +=
                "  " + handle(frame->f_code->co_filename).cast<std::string>() +
                "(" + std::to_string(lineno) + "): " +
                handle(frame->f_code->co_name).cast<std::string>() + "\n";
            frame = frame->f_back;
        }
    }
#endif

    return errorString;
}

PYBIND11_NOINLINE inline handle get_object_handle(const void *ptr, const detail::type_info *type ) {
    auto &instances = get_internals().registered_instances;
    auto range = instances.equal_range(ptr);
    for (auto it = range.first; it != range.second; ++it) {
        for (const auto &vh : values_and_holders(it->second)) {
            if (vh.type == type)
                return handle((PyObject *) it->second);
        }
    }
    return handle();
}

inline PyThreadState *get_thread_state_unchecked() {
#if defined(PYPY_VERSION)
    return PyThreadState_GET();
#elif PY_VERSION_HEX < 0x03000000
    return _PyThreadState_Current;
#elif PY_VERSION_HEX < 0x03050000
    return (PyThreadState*) _Py_atomic_load_relaxed(&_PyThreadState_Current);
#elif PY_VERSION_HEX < 0x03050200
    return (PyThreadState*) _PyThreadState_Current.value;
#else
    return _PyThreadState_UncheckedGet();
#endif
}

// Forward declarations
inline void keep_alive_impl(handle nurse, handle patient);
inline PyObject *make_new_instance(PyTypeObject *type);

class type_caster_generic {
public:
    PYBIND11_NOINLINE type_caster_generic(const std::type_info &type_info)
        : typeinfo(get_type_info(type_info)), cpptype(&type_info) { }

    type_caster_generic(const type_info *typeinfo)
        : typeinfo(typeinfo), cpptype(typeinfo ? typeinfo->cpptype : nullptr) { }

    bool load(handle src, bool convert) {
        return load_impl<type_caster_generic>(src, convert);
    }

    PYBIND11_NOINLINE static handle cast(const void *_src, return_value_policy policy, handle parent,
                                         const detail::type_info *tinfo,
                                         void *(*copy_constructor)(const void *),
                                         void *(*move_constructor)(const void *),
                                         const void *existing_holder = nullptr) {
        if (!tinfo) // no type info: error will be set already
            return handle();

        void *src = const_cast<void *>(_src);
        if (src == nullptr)
            return none().release();

        auto it_instances = get_internals().registered_instances.equal_range(src);
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
                if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype))
                    return handle((PyObject *) it_i->second).inc_ref();
            }
        }

        auto inst = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto wrapper = reinterpret_cast<instance *>(inst.ptr());
        wrapper->owned = false;
        void *&valueptr = values_and_holders(wrapper).begin()->value_ptr();

        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::take_ownership:
                valueptr = src;
                wrapper->owned = true;
                break;

            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
                valueptr = src;
                wrapper->owned = false;
                break;

            case return_value_policy::copy:
                if (copy_constructor)
                    valueptr = copy_constructor(src);
                else {
#if defined(NDEBUG)
                    throw cast_error("return_value_policy = copy, but type is "
                                     "non-copyable! (compile in debug mode for details)");
#else
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = copy, but type " +
                                     type_name + " is non-copyable!");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::move:
                if (move_constructor)
                    valueptr = move_constructor(src);
                else if (copy_constructor)
                    valueptr = copy_constructor(src);
                else {
#if defined(NDEBUG)
                    throw cast_error("return_value_policy = move, but type is neither "
                                     "movable nor copyable! "
                                     "(compile in debug mode for details)");
#else
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = move, but type " +
                                     type_name + " is neither movable nor copyable!");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::reference_internal:
                valueptr = src;
                wrapper->owned = false;
                keep_alive_impl(inst, parent);
                break;

            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        }

        tinfo->init_instance(wrapper, existing_holder);

        return inst.release();
    }

    // Base methods for generic caster; there are overridden in copyable_holder_caster
    void load_value(value_and_holder &&v_h) {
        auto *&vptr = v_h.value_ptr();
        // Lazy allocation for unallocated values:
        if (vptr == nullptr) {
            auto *type = v_h.type ? v_h.type : typeinfo;
            if (type->operator_new) {
                vptr = type->operator_new(type->type_size);
            } else {
                #if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
                    if (type->type_align > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
                        vptr = ::operator new(type->type_size,
                                              std::align_val_t(type->type_align));
                    else
                #endif
                vptr = ::operator new(type->type_size);
            }
        }
        value = vptr;
    }
    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            type_caster_generic sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                value = cast.second(sub_caster.value);
                return true;
            }
        }
        return false;
    }
    bool try_direct_conversions(handle src) {
        for (auto &converter : *typeinfo->direct_conversions) {
            if (converter(src.ptr(), value))
                return true;
        }
        return false;
    }
    void check_holder_compat() {}

    PYBIND11_NOINLINE static void *local_load(PyObject *src, const type_info *ti) {
        auto caster = type_caster_generic(ti);
        if (caster.load(src, false))
            return caster.value;
        return nullptr;
    }

    /// Try to load with foreign typeinfo, if available. Used when there is no
    /// native typeinfo, or when the native one wasn't able to produce a value.
    PYBIND11_NOINLINE bool try_load_foreign_module_local(handle src) {
        constexpr auto *local_key = PYBIND11_MODULE_LOCAL_ID;
        const auto pytype = src.get_type();
        if (!hasattr(pytype, local_key))
            return false;

        type_info *foreign_typeinfo = reinterpret_borrow<capsule>(getattr(pytype, local_key));
        // Only consider this foreign loader if actually foreign and is a loader of the correct cpp type
        if (foreign_typeinfo->module_local_load == &local_load
            || (cpptype && !same_type(*cpptype, *foreign_typeinfo->cpptype)))
            return false;

        if (auto result = foreign_typeinfo->module_local_load(src.ptr(), foreign_typeinfo)) {
            value = result;
            return true;
        }
        return false;
    }

    // Implementation of `load`; this takes the type of `this` so that it can dispatch the relevant
    // bits of code between here and copyable_holder_caster where the two classes need different
    // logic (without having to resort to virtual inheritance).
    template <typename ThisT>
    PYBIND11_NOINLINE bool load_impl(handle src, bool convert) {
        if (!src) return false;
        if (!typeinfo) return try_load_foreign_module_local(src);
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) return false;
            value = nullptr;
            return true;
        }

        auto &this_ = static_cast<ThisT &>(*this);
        this_.check_holder_compat();

        PyTypeObject *srctype = Py_TYPE(src.ptr());

        // Case 1: If src is an exact type match for the target type then we can reinterpret_cast
        // the instance's value pointer to the target type:
        if (srctype == typeinfo->type) {
            this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
            return true;
        }
        // Case 2: We have a derived class
        else if (PyType_IsSubtype(srctype, typeinfo->type)) {
            auto &bases = all_type_info(srctype);
            bool no_cpp_mi = typeinfo->simple_type;

            // Case 2a: the python type is a Python-inherited derived class that inherits from just
            // one simple (no MI) pybind11 class, or is an exact match, so the C++ instance is of
            // the right type and we can use reinterpret_cast.
            // (This is essentially the same as case 2b, but because not using multiple inheritance
            // is extremely common, we handle it specially to avoid the loop iterator and type
            // pointer lookup overhead)
            if (bases.size() == 1 && (no_cpp_mi || bases.front()->type == typeinfo->type)) {
                this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder());
                return true;
            }
            // Case 2b: the python type inherits from multiple C++ bases.  Check the bases to see if
            // we can find an exact match (or, for a simple C++ type, an inherited match); if so, we
            // can safely reinterpret_cast to the relevant pointer.
            else if (bases.size() > 1) {
                for (auto base : bases) {
                    if (no_cpp_mi ? PyType_IsSubtype(base->type, typeinfo->type) : base->type == typeinfo->type) {
                        this_.load_value(reinterpret_cast<instance *>(src.ptr())->get_value_and_holder(base));
                        return true;
                    }
                }
            }

            // Case 2c: C++ multiple inheritance is involved and we couldn't find an exact type match
            // in the registered bases, above, so try implicit casting (needed for proper C++ casting
            // when MI is involved).
            if (this_.try_implicit_casts(src, convert))
                return true;
        }

        // Perform an implicit conversion
        if (convert) {
            for (auto &converter : typeinfo->implicit_conversions) {
                auto temp = reinterpret_steal<object>(converter(src.ptr(), typeinfo->type));
                if (load_impl<ThisT>(temp, false)) {
                    loader_life_support::add_patient(temp);
                    return true;
                }
            }
            if (this_.try_direct_conversions(src))
                return true;
        }

        // Failed to match local typeinfo. Try again with global.
        if (typeinfo->module_local) {
            if (auto gtype = get_global_type_info(*typeinfo->cpptype)) {
                typeinfo = gtype;
                return load(src, false);
            }
        }

        // Global typeinfo has precedence over foreign module_local
        return try_load_foreign_module_local(src);
    }


    // Called to do type lookup and wrap the pointer and type in a pair when a dynamic_cast
    // isn't needed or can't be used.  If the type is unknown, sets the error and returns a pair
    // with .second = nullptr.  (p.first = nullptr is not an error: it becomes None).
    PYBIND11_NOINLINE static std::pair<const void *, const type_info *> src_and_type(
            const void *src, const std::type_info &cast_type, const std::type_info *rtti_type = nullptr) {
        if (auto *tpi = get_type_info(cast_type))
            return {src, const_cast<const type_info *>(tpi)};

        // Not found, set error:
        std::string tname = rtti_type ? rtti_type->name() : cast_type.name();
        detail::clean_type_id(tname);
        std::string msg = "Unregistered type : " + tname;
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        return {nullptr, nullptr};
    }

    const type_info *typeinfo = nullptr;
    const std::type_info *cpptype = nullptr;
    void *value = nullptr;
};

/**
 * Determine suitable casting operator for pointer-or-lvalue-casting type casters.  The type caster
 * needs to provide `operator T*()` and `operator T&()` operators.
 *
 * If the type supports moving the value away via an `operator T&&() &&` method, it should use
 * `movable_cast_op_type` instead.
 */
template <typename T>
using cast_op_type =
    conditional_t<std::is_pointer<remove_reference_t<T>>::value,
        typename std::add_pointer<intrinsic_t<T>>::type,
        typename std::add_lvalue_reference<intrinsic_t<T>>::type>;

/**
 * Determine suitable casting operator for a type caster with a movable value.  Such a type caster
 * needs to provide `operator T*()`, `operator T&()`, and `operator T&&() &&`.  The latter will be
 * called in appropriate contexts where the value can be moved rather than copied.
 *
 * These operator are automatically provided when using the PYBIND11_TYPE_CASTER macro.
 */
template <typename T>
using movable_cast_op_type =
    conditional_t<std::is_pointer<typename std::remove_reference<T>::type>::value,
        typename std::add_pointer<intrinsic_t<T>>::type,
    conditional_t<std::is_rvalue_reference<T>::value,
        typename std::add_rvalue_reference<intrinsic_t<T>>::type,
        typename std::add_lvalue_reference<intrinsic_t<T>>::type>>;

// std::is_copy_constructible isn't quite enough: it lets std::vector<T> (and similar) through when
// T is non-copyable, but code containing such a copy constructor fails to actually compile.
template <typename T, typename SFINAE = void> struct is_copy_constructible : std::is_copy_constructible<T> {};

// Specialization for types that appear to be copy constructible but also look like stl containers
// (we specifically check for: has `value_type` and `reference` with `reference = value_type&`): if
// so, copy constructability depends on whether the value_type is copy constructible.
template <typename Container> struct is_copy_constructible<Container, enable_if_t<all_of<
        std::is_copy_constructible<Container>,
        std::is_same<typename Container::value_type &, typename Container::reference>,
        // Avoid infinite recursion
        negation<std::is_same<Container, typename Container::value_type>>
    >::value>> : is_copy_constructible<typename Container::value_type> {};

// Likewise for std::pair
// (after C++17 it is mandatory that the copy constructor not exist when the two types aren't themselves
// copy constructible, but this can not be relied upon when T1 or T2 are themselves containers).
template <typename T1, typename T2> struct is_copy_constructible<std::pair<T1, T2>>
    : all_of<is_copy_constructible<T1>, is_copy_constructible<T2>> {};

// The same problems arise with std::is_copy_assignable, so we use the same workaround.
template <typename T, typename SFINAE = void> struct is_copy_assignable : std::is_copy_assignable<T> {};
template <typename Container> struct is_copy_assignable<Container, enable_if_t<all_of<
        std::is_copy_assignable<Container>,
        std::is_same<typename Container::value_type &, typename Container::reference>
    >::value>> : is_copy_assignable<typename Container::value_type> {};
template <typename T1, typename T2> struct is_copy_assignable<std::pair<T1, T2>>
    : all_of<is_copy_assignable<T1>, is_copy_assignable<T2>> {};

PYBIND11_NAMESPACE_END(detail)

// polymorphic_type_hook<itype>::get(src, tinfo) determines whether the object pointed
// to by `src` actually is an instance of some class derived from `itype`.
// If so, it sets `tinfo` to point to the std::type_info representing that derived
// type, and returns a pointer to the start of the most-derived object of that type
// (in which `src` is a subobject; this will be the same address as `src` in most
// single inheritance cases). If not, or if `src` is nullptr, it simply returns `src`
// and leaves `tinfo` at its default value of nullptr.
//
// The default polymorphic_type_hook just returns src. A specialization for polymorphic
// types determines the runtime type of the passed object and adjusts the this-pointer
// appropriately via dynamic_cast<void*>. This is what enables a C++ Animal* to appear
// to Python as a Dog (if Dog inherits from Animal, Animal is polymorphic, Dog is
// registered with pybind11, and this Animal is in fact a Dog).
//
// You may specialize polymorphic_type_hook yourself for types that want to appear
// polymorphic to Python but do not use C++ RTTI. (This is a not uncommon pattern
// in performance-sensitive applications, used most notably in LLVM.)
//
// polymorphic_type_hook_base allows users to specialize polymorphic_type_hook with
// std::enable_if. User provided specializations will always have higher priority than
// the default implementation and specialization provided in polymorphic_type_hook_base.
template <typename itype, typename SFINAE = void>
struct polymorphic_type_hook_base
{
    static const void *get(const itype *src, const std::type_info*&) { return src; }
};
template <typename itype>
struct polymorphic_type_hook_base<itype, detail::enable_if_t<std::is_polymorphic<itype>::value>>
{
    static const void *get(const itype *src, const std::type_info*& type) {
        type = src ? &typeid(*src) : nullptr;
        return dynamic_cast<const void*>(src);
    }
};
template <typename itype, typename SFINAE = void>
struct polymorphic_type_hook : public polymorphic_type_hook_base<itype> {};

PYBIND11_NAMESPACE_BEGIN(detail)

/// Generic type caster for objects stored on the heap
template <typename type> class type_caster_base : public type_caster_generic {
    using itype = intrinsic_t<type>;

public:
    static constexpr auto name = _<type>();

    type_caster_base() : type_caster_base(typeid(type)) { }
    explicit type_caster_base(const std::type_info &info) : type_caster_generic(info) { }

    static handle cast(const itype &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
    }

    static handle cast(itype &&src, return_value_policy, handle parent) {
        return cast(&src, return_value_policy::move, parent);
    }

    // Returns a (pointer, type_info) pair taking care of necessary type lookup for a
    // polymorphic type (using RTTI by default, but can be overridden by specializing
    // polymorphic_type_hook). If the instance isn't derived, returns the base version.
    static std::pair<const void *, const type_info *> src_and_type(const itype *src) {
        auto &cast_type = typeid(itype);
        const std::type_info *instance_type = nullptr;
        const void *vsrc = polymorphic_type_hook<itype>::get(src, instance_type);
        if (instance_type && !same_type(cast_type, *instance_type)) {
            // This is a base pointer to a derived type. If the derived type is registered
            // with pybind11, we want to make the full derived object available.
            // In the typical case where itype is polymorphic, we get the correct
            // derived pointer (which may be != base pointer) by a dynamic_cast to
            // most derived type. If itype is not polymorphic, we won't get here
            // except via a user-provided specialization of polymorphic_type_hook,
            // and the user has promised that no this-pointer adjustment is
            // required in that case, so it's OK to use static_cast.
            if (const auto *tpi = get_type_info(*instance_type))
                return {vsrc, tpi};
        }
        // Otherwise we have either a nullptr, an `itype` pointer, or an unknown derived pointer, so
        // don't do a cast
        return type_caster_generic::src_and_type(src, cast_type, instance_type);
    }

    static handle cast(const itype *src, return_value_policy policy, handle parent) {
        auto st = src_and_type(src);
        return type_caster_generic::cast(
            st.first, policy, parent, st.second,
            make_copy_constructor(src), make_move_constructor(src));
    }

    static handle cast_holder(const itype *src, const void *holder) {
        auto st = src_and_type(src);
        return type_caster_generic::cast(
            st.first, return_value_policy::take_ownership, {}, st.second,
            nullptr, nullptr, holder);
    }

    template <typename T> using cast_op_type = detail::cast_op_type<T>;

    operator itype*() { return (type *) value; }
    operator itype&() { if (!value) throw reference_cast_error(); return *((itype *) value); }

protected:
    using Constructor = void *(*)(const void *);

    /* Only enabled when the types are {copy,move}-constructible *and* when the type
       does not have a private operator new implementation. */
    template <typename T, typename = enable_if_t<is_copy_constructible<T>::value>>
    static auto make_copy_constructor(const T *x) -> decltype(new T(*x), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(*reinterpret_cast<const T *>(arg));
        };
    }

    template <typename T, typename = enable_if_t<std::is_move_constructible<T>::value>>
    static auto make_move_constructor(const T *x) -> decltype(new T(std::move(*const_cast<T *>(x))), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(std::move(*const_cast<T *>(reinterpret_cast<const T *>(arg))));
        };
    }

    static Constructor make_copy_constructor(...) { return nullptr; }
    static Constructor make_move_constructor(...) { return nullptr; }
};

template <typename type, typename SFINAE = void> class type_caster : public type_caster_base<type> { };
template <typename type> using make_caster = type_caster<intrinsic_t<type>>;

// Shortcut for calling a caster's `cast_op_type` cast operator for casting a type_caster to a T
template <typename T> typename make_caster<T>::template cast_op_type<T> cast_op(make_caster<T> &caster) {
    return caster.operator typename make_caster<T>::template cast_op_type<T>();
}
template <typename T> typename make_caster<T>::template cast_op_type<typename std::add_rvalue_reference<T>::type>
cast_op(make_caster<T> &&caster) {
    return std::move(caster).operator
        typename make_caster<T>::template cast_op_type<typename std::add_rvalue_reference<T>::type>();
}

template <typename type> class type_caster<std::reference_wrapper<type>> {
private:
    using caster_t = make_caster<type>;
    caster_t subcaster;
    using subcaster_cast_op_type = typename caster_t::template cast_op_type<type>;
    static_assert(std::is_same<typename std::remove_const<type>::type &, subcaster_cast_op_type>::value,
            "std::reference_wrapper<T> caster requires T to have a caster with an `T &` operator");
public:
    bool load(handle src, bool convert) { return subcaster.load(src, convert); }
    static constexpr auto name = caster_t::name;
    static handle cast(const std::reference_wrapper<type> &src, return_value_policy policy, handle parent) {
        // It is definitely wrong to take ownership of this pointer, so mask that rvp
        if (policy == return_value_policy::take_ownership || policy == return_value_policy::automatic)
            policy = return_value_policy::automatic_reference;
        return caster_t::cast(&src.get(), policy, parent);
    }
    template <typename T> using cast_op_type = std::reference_wrapper<type>;
    operator std::reference_wrapper<type>() { return subcaster.operator subcaster_cast_op_type&(); }
};

#define PYBIND11_TYPE_CASTER(type, py_name) \
    protected: \
        type value; \
    public: \
        static constexpr auto name = py_name; \
        template <typename T_, enable_if_t<std::is_same<type, remove_cv_t<T_>>::value, int> = 0> \
        static handle cast(T_ *src, return_value_policy policy, handle parent) { \
            if (!src) return none().release(); \
            if (policy == return_value_policy::take_ownership) { \
                auto h = cast(std::move(*src), policy, parent); delete src; return h; \
            } else { \
                return cast(*src, policy, parent); \
            } \
        } \
        operator type*() { return &value; } \
        operator type&() { return value; } \
        operator type&&() && { return std::move(value); } \
        template <typename T_> using cast_op_type = pybind11::detail::movable_cast_op_type<T_>


template <typename CharT> using is_std_char_type = any_of<
    std::is_same<CharT, char>, /* std::string */
#if defined(PYBIND11_HAS_U8STRING)
    std::is_same<CharT, char8_t>, /* std::u8string */
#endif
    std::is_same<CharT, char16_t>, /* std::u16string */
    std::is_same<CharT, char32_t>, /* std::u32string */
    std::is_same<CharT, wchar_t> /* std::wstring */
>;

template <typename T>
struct type_caster<T, enable_if_t<std::is_arithmetic<T>::value && !is_std_char_type<T>::value>> {
    using _py_type_0 = conditional_t<sizeof(T) <= sizeof(long), long, long long>;
    using _py_type_1 = conditional_t<std::is_signed<T>::value, _py_type_0, typename std::make_unsigned<_py_type_0>::type>;
    using py_type = conditional_t<std::is_floating_point<T>::value, double, _py_type_1>;
public:

    bool load(handle src, bool convert) {
        py_type py_value;

        if (!src)
            return false;

        if (std::is_floating_point<T>::value) {
            if (convert || PyFloat_Check(src.ptr()))
                py_value = (py_type) PyFloat_AsDouble(src.ptr());
            else
                return false;
        } else if (PyFloat_Check(src.ptr())) {
            return false;
        } else if (std::is_unsigned<py_type>::value) {
            py_value = as_unsigned<py_type>(src.ptr());
        } else { // signed integer:
            py_value = sizeof(T) <= sizeof(long)
                ? (py_type) PyLong_AsLong(src.ptr())
                : (py_type) PYBIND11_LONG_AS_LONGLONG(src.ptr());
        }

        bool py_err = py_value == (py_type) -1 && PyErr_Occurred();

        // Protect std::numeric_limits::min/max with parentheses
        if (py_err || (std::is_integral<T>::value && sizeof(py_type) != sizeof(T) &&
                       (py_value < (py_type) (std::numeric_limits<T>::min)() ||
                        py_value > (py_type) (std::numeric_limits<T>::max)()))) {
            bool type_error = py_err && PyErr_ExceptionMatches(
#if PY_VERSION_HEX < 0x03000000 && !defined(PYPY_VERSION)
                PyExc_SystemError
#else
                PyExc_TypeError
#endif
            );
            PyErr_Clear();
            if (type_error && convert && PyNumber_Check(src.ptr())) {
                auto tmp = reinterpret_steal<object>(std::is_floating_point<T>::value
                                                     ? PyNumber_Float(src.ptr())
                                                     : PyNumber_Long(src.ptr()));
                PyErr_Clear();
                return load(tmp, false);
            }
            return false;
        }

        value = (T) py_value;
        return true;
    }

    template<typename U = T>
    static typename std::enable_if<std::is_floating_point<U>::value, handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PyFloat_FromDouble((double) src);
    }

    template<typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_signed<U>::value && (sizeof(U) <= sizeof(long)), handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PYBIND11_LONG_FROM_SIGNED((long) src);
    }

    template<typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_unsigned<U>::value && (sizeof(U) <= sizeof(unsigned long)), handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PYBIND11_LONG_FROM_UNSIGNED((unsigned long) src);
    }

    template<typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_signed<U>::value && (sizeof(U) > sizeof(long)), handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PyLong_FromLongLong((long long) src);
    }

    template<typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_unsigned<U>::value && (sizeof(U) > sizeof(unsigned long)), handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PyLong_FromUnsignedLongLong((unsigned long long) src);
    }

    PYBIND11_TYPE_CASTER(T, _<std::is_integral<T>::value>("int", "float"));
};

template<typename T> struct void_caster {
public:
    bool load(handle src, bool) {
        if (src && src.is_none())
            return true;
        return false;
    }
    static handle cast(T, return_value_policy /* policy */, handle /* parent */) {
        return none().inc_ref();
    }
    PYBIND11_TYPE_CASTER(T, _("None"));
};

template <> class type_caster<void_type> : public void_caster<void_type> {};

template <> class type_caster<void> : public type_caster<void_type> {
public:
    using type_caster<void_type>::cast;

    bool load(handle h, bool) {
        if (!h) {
            return false;
        } else if (h.is_none()) {
            value = nullptr;
            return true;
        }

        /* Check if this is a capsule */
        if (isinstance<capsule>(h)) {
            value = reinterpret_borrow<capsule>(h);
            return true;
        }

        /* Check if this is a C++ type */
        auto &bases = all_type_info((PyTypeObject *) h.get_type().ptr());
        if (bases.size() == 1) { // Only allowing loading from a single-value type
            value = values_and_holders(reinterpret_cast<instance *>(h.ptr())).begin()->value_ptr();
            return true;
        }

        /* Fail */
        return false;
    }

    static handle cast(const void *ptr, return_value_policy /* policy */, handle /* parent */) {
        if (ptr)
            return capsule(ptr).release();
        else
            return none().inc_ref();
    }

    template <typename T> using cast_op_type = void*&;
    operator void *&() { return value; }
    static constexpr auto name = _("capsule");
private:
    void *value = nullptr;
};

template <> class type_caster<std::nullptr_t> : public void_caster<std::nullptr_t> { };

template <> class type_caster<bool> {
public:
    bool load(handle src, bool convert) {
        if (!src) return false;
        else if (src.ptr() == Py_True) { value = true; return true; }
        else if (src.ptr() == Py_False) { value = false; return true; }
        else if (convert || !strcmp("numpy.bool_", Py_TYPE(src.ptr())->tp_name)) {
            // (allow non-implicit conversion for numpy booleans)

            Py_ssize_t res = -1;
            if (src.is_none()) {
                res = 0;  // None is implicitly converted to False
            }
            #if defined(PYPY_VERSION)
            // On PyPy, check that "__bool__" (or "__nonzero__" on Python 2.7) attr exists
            else if (hasattr(src, PYBIND11_BOOL_ATTR)) {
                res = PyObject_IsTrue(src.ptr());
            }
            #else
            // Alternate approach for CPython: this does the same as the above, but optimized
            // using the CPython API so as to avoid an unneeded attribute lookup.
            else if (auto tp_as_number = src.ptr()->ob_type->tp_as_number) {
                if (PYBIND11_NB_BOOL(tp_as_number)) {
                    res = (*PYBIND11_NB_BOOL(tp_as_number))(src.ptr());
                }
            }
            #endif
            if (res == 0 || res == 1) {
                value = (bool) res;
                return true;
            } else {
                PyErr_Clear();
            }
        }
        return false;
    }
    static handle cast(bool src, return_value_policy /* policy */, handle /* parent */) {
        return handle(src ? Py_True : Py_False).inc_ref();
    }
    PYBIND11_TYPE_CASTER(bool, _("bool"));
};

// Helper class for UTF-{8,16,32} C++ stl strings:
template <typename StringType, bool IsView = false> struct string_caster {
    using CharT = typename StringType::value_type;

    // Simplify life by being able to assume standard char sizes (the standard only guarantees
    // minimums, but Python requires exact sizes)
    static_assert(!std::is_same<CharT, char>::value || sizeof(CharT) == 1, "Unsupported char size != 1");
#if defined(PYBIND11_HAS_U8STRING)
    static_assert(!std::is_same<CharT, char8_t>::value || sizeof(CharT) == 1, "Unsupported char8_t size != 1");
#endif
    static_assert(!std::is_same<CharT, char16_t>::value || sizeof(CharT) == 2, "Unsupported char16_t size != 2");
    static_assert(!std::is_same<CharT, char32_t>::value || sizeof(CharT) == 4, "Unsupported char32_t size != 4");
    // wchar_t can be either 16 bits (Windows) or 32 (everywhere else)
    static_assert(!std::is_same<CharT, wchar_t>::value || sizeof(CharT) == 2 || sizeof(CharT) == 4,
            "Unsupported wchar_t size != 2/4");
    static constexpr size_t UTF_N = 8 * sizeof(CharT);

    bool load(handle src, bool) {
#if PY_MAJOR_VERSION < 3
        object temp;
#endif
        handle load_src = src;
        if (!src) {
            return false;
        } else if (!PyUnicode_Check(load_src.ptr())) {
#if PY_MAJOR_VERSION >= 3
            return load_bytes(load_src);
#else
            if (std::is_same<CharT, char>::value) {
                return load_bytes(load_src);
            }

            // The below is a guaranteed failure in Python 3 when PyUnicode_Check returns false
            if (!PYBIND11_BYTES_CHECK(load_src.ptr()))
                return false;

            temp = reinterpret_steal<object>(PyUnicode_FromObject(load_src.ptr()));
            if (!temp) { PyErr_Clear(); return false; }
            load_src = temp;
#endif
        }

        object utfNbytes = reinterpret_steal<object>(PyUnicode_AsEncodedString(
            load_src.ptr(), UTF_N == 8 ? "utf-8" : UTF_N == 16 ? "utf-16" : "utf-32", nullptr));
        if (!utfNbytes) { PyErr_Clear(); return false; }

        const CharT *buffer = reinterpret_cast<const CharT *>(PYBIND11_BYTES_AS_STRING(utfNbytes.ptr()));
        size_t length = (size_t) PYBIND11_BYTES_SIZE(utfNbytes.ptr()) / sizeof(CharT);
        if (UTF_N > 8) { buffer++; length--; } // Skip BOM for UTF-16/32
        value = StringType(buffer, length);

        // If we're loading a string_view we need to keep the encoded Python object alive:
        if (IsView)
            loader_life_support::add_patient(utfNbytes);

        return true;
    }

    static handle cast(const StringType &src, return_value_policy /* policy */, handle /* parent */) {
        const char *buffer = reinterpret_cast<const char *>(src.data());
        ssize_t nbytes = ssize_t(src.size() * sizeof(CharT));
        handle s = decode_utfN(buffer, nbytes);
        if (!s) throw error_already_set();
        return s;
    }

    PYBIND11_TYPE_CASTER(StringType, _(PYBIND11_STRING_NAME));

private:
    static handle decode_utfN(const char *buffer, ssize_t nbytes) {
#if !defined(PYPY_VERSION)
        return
            UTF_N == 8  ? PyUnicode_DecodeUTF8(buffer, nbytes, nullptr) :
            UTF_N == 16 ? PyUnicode_DecodeUTF16(buffer, nbytes, nullptr, nullptr) :
                          PyUnicode_DecodeUTF32(buffer, nbytes, nullptr, nullptr);
#else
        // PyPy seems to have multiple problems related to PyUnicode_UTF*: the UTF8 version
        // sometimes segfaults for unknown reasons, while the UTF16 and 32 versions require a
        // non-const char * arguments, which is also a nuisance, so bypass the whole thing by just
        // passing the encoding as a string value, which works properly:
        return PyUnicode_Decode(buffer, nbytes, UTF_N == 8 ? "utf-8" : UTF_N == 16 ? "utf-16" : "utf-32", nullptr);
#endif
    }

    // When loading into a std::string or char*, accept a bytes object as-is (i.e.
    // without any encoding/decoding attempt).  For other C++ char sizes this is a no-op.
    // which supports loading a unicode from a str, doesn't take this path.
    template <typename C = CharT>
    bool load_bytes(enable_if_t<std::is_same<C, char>::value, handle> src) {
        if (PYBIND11_BYTES_CHECK(src.ptr())) {
            // We were passed a Python 3 raw bytes; accept it into a std::string or char*
            // without any encoding attempt.
            const char *bytes = PYBIND11_BYTES_AS_STRING(src.ptr());
            if (bytes) {
                value = StringType(bytes, (size_t) PYBIND11_BYTES_SIZE(src.ptr()));
                return true;
            }
        }

        return false;
    }

    template <typename C = CharT>
    bool load_bytes(enable_if_t<!std::is_same<C, char>::value, handle>) { return false; }
};

template <typename CharT, class Traits, class Allocator>
struct type_caster<std::basic_string<CharT, Traits, Allocator>, enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<std::basic_string<CharT, Traits, Allocator>> {};

#ifdef PYBIND11_HAS_STRING_VIEW
template <typename CharT, class Traits>
struct type_caster<std::basic_string_view<CharT, Traits>, enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<std::basic_string_view<CharT, Traits>, true> {};
#endif

// Type caster for C-style strings.  We basically use a std::string type caster, but also add the
// ability to use None as a nullptr char* (which the string caster doesn't allow).
template <typename CharT> struct type_caster<CharT, enable_if_t<is_std_char_type<CharT>::value>> {
    using StringType = std::basic_string<CharT>;
    using StringCaster = type_caster<StringType>;
    StringCaster str_caster;
    bool none = false;
    CharT one_char = 0;
public:
    bool load(handle src, bool convert) {
        if (!src) return false;
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) return false;
            none = true;
            return true;
        }
        return str_caster.load(src, convert);
    }

    static handle cast(const CharT *src, return_value_policy policy, handle parent) {
        if (src == nullptr) return pybind11::none().inc_ref();
        return StringCaster::cast(StringType(src), policy, parent);
    }

    static handle cast(CharT src, return_value_policy policy, handle parent) {
        if (std::is_same<char, CharT>::value) {
            handle s = PyUnicode_DecodeLatin1((const char *) &src, 1, nullptr);
            if (!s) throw error_already_set();
            return s;
        }
        return StringCaster::cast(StringType(1, src), policy, parent);
    }

    operator CharT*() { return none ? nullptr : const_cast<CharT *>(static_cast<StringType &>(str_caster).c_str()); }
    operator CharT&() {
        if (none)
            throw value_error("Cannot convert None to a character");

        auto &value = static_cast<StringType &>(str_caster);
        size_t str_len = value.size();
        if (str_len == 0)
            throw value_error("Cannot convert empty string to a character");

        // If we're in UTF-8 mode, we have two possible failures: one for a unicode character that
        // is too high, and one for multiple unicode characters (caught later), so we need to figure
        // out how long the first encoded character is in bytes to distinguish between these two
        // errors.  We also allow want to allow unicode characters U+0080 through U+00FF, as those
        // can fit into a single char value.
        if (StringCaster::UTF_N == 8 && str_len > 1 && str_len <= 4) {
            unsigned char v0 = static_cast<unsigned char>(value[0]);
            size_t char0_bytes = !(v0 & 0x80) ? 1 : // low bits only: 0-127
                (v0 & 0xE0) == 0xC0 ? 2 : // 0b110xxxxx - start of 2-byte sequence
                (v0 & 0xF0) == 0xE0 ? 3 : // 0b1110xxxx - start of 3-byte sequence
                4; // 0b11110xxx - start of 4-byte sequence

            if (char0_bytes == str_len) {
                // If we have a 128-255 value, we can decode it into a single char:
                if (char0_bytes == 2 && (v0 & 0xFC) == 0xC0) { // 0x110000xx 0x10xxxxxx
                    one_char = static_cast<CharT>(((v0 & 3) << 6) + (static_cast<unsigned char>(value[1]) & 0x3F));
                    return one_char;
                }
                // Otherwise we have a single character, but it's > U+00FF
                throw value_error("Character code point not in range(0x100)");
            }
        }

        // UTF-16 is much easier: we can only have a surrogate pair for values above U+FFFF, thus a
        // surrogate pair with total length 2 instantly indicates a range error (but not a "your
        // string was too long" error).
        else if (StringCaster::UTF_N == 16 && str_len == 2) {
            one_char = static_cast<CharT>(value[0]);
            if (one_char >= 0xD800 && one_char < 0xE000)
                throw value_error("Character code point not in range(0x10000)");
        }

        if (str_len != 1)
            throw value_error("Expected a character, but multi-character string found");

        one_char = value[0];
        return one_char;
    }

    static constexpr auto name = _(PYBIND11_STRING_NAME);
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
};

// Base implementation for std::tuple and std::pair
template <template<typename...> class Tuple, typename... Ts> class tuple_caster {
    using type = Tuple<Ts...>;
    static constexpr auto size = sizeof...(Ts);
    using indices = make_index_sequence<size>;
public:

    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src))
            return false;
        const auto seq = reinterpret_borrow<sequence>(src);
        if (seq.size() != size)
            return false;
        return load_impl(seq, convert, indices{});
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        return cast_impl(std::forward<T>(src), policy, parent, indices{});
    }

    // copied from the PYBIND11_TYPE_CASTER macro
    template <typename T>
    static handle cast(T *src, return_value_policy policy, handle parent) {
        if (!src) return none().release();
        if (policy == return_value_policy::take_ownership) {
            auto h = cast(std::move(*src), policy, parent); delete src; return h;
        } else {
            return cast(*src, policy, parent);
        }
    }

    static constexpr auto name = _("Tuple[") + concat(make_caster<Ts>::name...) + _("]");

    template <typename T> using cast_op_type = type;

    operator type() & { return implicit_cast(indices{}); }
    operator type() && { return std::move(*this).implicit_cast(indices{}); }

protected:
    template <size_t... Is>
    type implicit_cast(index_sequence<Is...>) & { return type(cast_op<Ts>(std::get<Is>(subcasters))...); }
    template <size_t... Is>
    type implicit_cast(index_sequence<Is...>) && { return type(cast_op<Ts>(std::move(std::get<Is>(subcasters)))...); }

    static constexpr bool load_impl(const sequence &, bool, index_sequence<>) { return true; }

    template <size_t... Is>
    bool load_impl(const sequence &seq, bool convert, index_sequence<Is...>) {
#ifdef __cpp_fold_expressions
        if ((... || !std::get<Is>(subcasters).load(seq[Is], convert)))
            return false;
#else
        for (bool r : {std::get<Is>(subcasters).load(seq[Is], convert)...})
            if (!r)
                return false;
#endif
        return true;
    }

    /* Implementation: Convert a C++ tuple into a Python tuple */
    template <typename T, size_t... Is>
    static handle cast_impl(T &&src, return_value_policy policy, handle parent, index_sequence<Is...>) {
        std::array<object, size> entries{{
            reinterpret_steal<object>(make_caster<Ts>::cast(std::get<Is>(std::forward<T>(src)), policy, parent))...
        }};
        for (const auto &entry: entries)
            if (!entry)
                return handle();
        tuple result(size);
        int counter = 0;
        for (auto & entry: entries)
            PyTuple_SET_ITEM(result.ptr(), counter++, entry.release().ptr());
        return result.release();
    }

    Tuple<make_caster<Ts>...> subcasters;
};

template <typename T1, typename T2> class type_caster<std::pair<T1, T2>>
    : public tuple_caster<std::pair, T1, T2> {};

template <typename... Ts> class type_caster<std::tuple<Ts...>>
    : public tuple_caster<std::tuple, Ts...> {};

/// Helper class which abstracts away certain actions. Users can provide specializations for
/// custom holders, but it's only necessary if the type has a non-standard interface.
template <typename T>
struct holder_helper {
    static auto get(const T &p) -> decltype(p.get()) { return p.get(); }
};

/// Type caster for holder types like std::shared_ptr, etc.
template <typename type, typename holder_type>
struct copyable_holder_caster : public type_caster_base<type> {
public:
    using base = type_caster_base<type>;
    static_assert(std::is_base_of<base, type_caster<type>>::value,
            "Holder classes are only supported for custom types");
    using base::base;
    using base::cast;
    using base::typeinfo;
    using base::value;

    bool load(handle src, bool convert) {
        return base::template load_impl<copyable_holder_caster<type, holder_type>>(src, convert);
    }

    explicit operator type*() { return this->value; }
    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
    // see issue #2180
    explicit operator type&() { return *(static_cast<type *>(this->value)); }
    explicit operator holder_type*() { return std::addressof(holder); }

    // Workaround for Intel compiler bug
    // see pybind11 issue 94
    #if defined(__ICC) || defined(__INTEL_COMPILER)
    operator holder_type&() { return holder; }
    #else
    explicit operator holder_type&() { return holder; }
    #endif

    static handle cast(const holder_type &src, return_value_policy, handle) {
        const auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, &src);
    }

protected:
    friend class type_caster_generic;
    void check_holder_compat() {
        if (typeinfo->default_holder)
            throw cast_error("Unable to load a custom holder type from a default-holder instance");
    }

    bool load_value(value_and_holder &&v_h) {
        if (v_h.holder_constructed()) {
            value = v_h.value_ptr();
            holder = v_h.template holder<holder_type>();
            return true;
        } else {
            throw cast_error("Unable to cast from non-held to held instance (T& to Holder<T>) "
#if defined(NDEBUG)
                             "(compile in debug mode for type information)");
#else
                             "of type '" + type_id<holder_type>() + "''");
#endif
        }
    }

    template <typename T = holder_type, detail::enable_if_t<!std::is_constructible<T, const T &, type*>::value, int> = 0>
    bool try_implicit_casts(handle, bool) { return false; }

    template <typename T = holder_type, detail::enable_if_t<std::is_constructible<T, const T &, type*>::value, int> = 0>
    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            copyable_holder_caster sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                value = cast.second(sub_caster.value);
                holder = holder_type(sub_caster.holder, (type *) value);
                return true;
            }
        }
        return false;
    }

    static bool try_direct_conversions(handle) { return false; }


    holder_type holder;
};

/// Specialize for the common std::shared_ptr, so users don't need to
template <typename T>
class type_caster<std::shared_ptr<T>> : public copyable_holder_caster<T, std::shared_ptr<T>> { };

template <typename type, typename holder_type>
struct move_only_holder_caster {
    static_assert(std::is_base_of<type_caster_base<type>, type_caster<type>>::value,
            "Holder classes are only supported for custom types");

    static handle cast(holder_type &&src, return_value_policy, handle) {
        auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, std::addressof(src));
    }
    static constexpr auto name = type_caster_base<type>::name;
};

template <typename type, typename deleter>
class type_caster<std::unique_ptr<type, deleter>>
    : public move_only_holder_caster<type, std::unique_ptr<type, deleter>> { };

template <typename type, typename holder_type>
using type_caster_holder = conditional_t<is_copy_constructible<holder_type>::value,
                                         copyable_holder_caster<type, holder_type>,
                                         move_only_holder_caster<type, holder_type>>;

template <typename T, bool Value = false> struct always_construct_holder { static constexpr bool value = Value; };

/// Create a specialization for custom holder types (silently ignores std::shared_ptr)
#define PYBIND11_DECLARE_HOLDER_TYPE(type, holder_type, ...) \
    namespace pybind11 { namespace detail { \
    template <typename type> \
    struct always_construct_holder<holder_type> : always_construct_holder<void, ##__VA_ARGS__>  { }; \
    template <typename type> \
    class type_caster<holder_type, enable_if_t<!is_shared_ptr<holder_type>::value>> \
        : public type_caster_holder<type, holder_type> { }; \
    }}

// PYBIND11_DECLARE_HOLDER_TYPE holder types:
template <typename base, typename holder> struct is_holder_type :
    std::is_base_of<detail::type_caster_holder<base, holder>, detail::type_caster<holder>> {};
// Specialization for always-supported unique_ptr holders:
template <typename base, typename deleter> struct is_holder_type<base, std::unique_ptr<base, deleter>> :
    std::true_type {};

template <typename T> struct handle_type_name { static constexpr auto name = _<T>(); };
template <> struct handle_type_name<bytes> { static constexpr auto name = _(PYBIND11_BYTES_NAME); };
template <> struct handle_type_name<int_> { static constexpr auto name = _("int"); };
template <> struct handle_type_name<iterable> { static constexpr auto name = _("Iterable"); };
template <> struct handle_type_name<iterator> { static constexpr auto name = _("Iterator"); };
template <> struct handle_type_name<none> { static constexpr auto name = _("None"); };
template <> struct handle_type_name<args> { static constexpr auto name = _("*args"); };
template <> struct handle_type_name<kwargs> { static constexpr auto name = _("**kwargs"); };

template <typename type>
struct pyobject_caster {
    template <typename T = type, enable_if_t<std::is_same<T, handle>::value, int> = 0>
    bool load(handle src, bool /* convert */) { value = src; return static_cast<bool>(value); }

    template <typename T = type, enable_if_t<std::is_base_of<object, T>::value, int> = 0>
    bool load(handle src, bool /* convert */) {
        if (!isinstance<type>(src))
            return false;
        value = reinterpret_borrow<type>(src);
        return true;
    }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name);
};

template <typename T>
class type_caster<T, enable_if_t<is_pyobject<T>::value>> : public pyobject_caster<T> { };

// Our conditions for enabling moving are quite restrictive:
// At compile time:
// - T needs to be a non-const, non-pointer, non-reference type
// - type_caster<T>::operator T&() must exist
// - the type must be move constructible (obviously)
// At run-time:
// - if the type is non-copy-constructible, the object must be the sole owner of the type (i.e. it
//   must have ref_count() == 1)h
// If any of the above are not satisfied, we fall back to copying.
template <typename T> using move_is_plain_type = satisfies_none_of<T,
    std::is_void, std::is_pointer, std::is_reference, std::is_const
>;
template <typename T, typename SFINAE = void> struct move_always : std::false_type {};
template <typename T> struct move_always<T, enable_if_t<all_of<
    move_is_plain_type<T>,
    negation<is_copy_constructible<T>>,
    std::is_move_constructible<T>,
    std::is_same<decltype(std::declval<make_caster<T>>().operator T&()), T&>
>::value>> : std::true_type {};
template <typename T, typename SFINAE = void> struct move_if_unreferenced : std::false_type {};
template <typename T> struct move_if_unreferenced<T, enable_if_t<all_of<
    move_is_plain_type<T>,
    negation<move_always<T>>,
    std::is_move_constructible<T>,
    std::is_same<decltype(std::declval<make_caster<T>>().operator T&()), T&>
>::value>> : std::true_type {};
template <typename T> using move_never = none_of<move_always<T>, move_if_unreferenced<T>>;

// Detect whether returning a `type` from a cast on type's type_caster is going to result in a
// reference or pointer to a local variable of the type_caster.  Basically, only
// non-reference/pointer `type`s and reference/pointers from a type_caster_generic are safe;
// everything else returns a reference/pointer to a local variable.
template <typename type> using cast_is_temporary_value_reference = bool_constant<
    (std::is_reference<type>::value || std::is_pointer<type>::value) &&
    !std::is_base_of<type_caster_generic, make_caster<type>>::value &&
    !std::is_same<intrinsic_t<type>, void>::value
>;

// When a value returned from a C++ function is being cast back to Python, we almost always want to
// force `policy = move`, regardless of the return value policy the function/method was declared
// with.
template <typename Return, typename SFINAE = void> struct return_value_policy_override {
    static return_value_policy policy(return_value_policy p) { return p; }
};

template <typename Return> struct return_value_policy_override<Return,
        detail::enable_if_t<std::is_base_of<type_caster_generic, make_caster<Return>>::value, void>> {
    static return_value_policy policy(return_value_policy p) {
        return !std::is_lvalue_reference<Return>::value &&
               !std::is_pointer<Return>::value
                   ? return_value_policy::move : p;
    }
};

// Basic python -> C++ casting; throws if casting fails
template <typename T, typename SFINAE> type_caster<T, SFINAE> &load_type(type_caster<T, SFINAE> &conv, const handle &handle) {
    if (!conv.load(handle, true)) {
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ type (compile in debug mode for details)");
#else
        throw cast_error("Unable to cast Python instance of type " +
            (std::string) str(handle.get_type()) + " to C++ type '" + type_id<T>() + "'");
#endif
    }
    return conv;
}
// Wrapper around the above that also constructs and returns a type_caster
template <typename T> make_caster<T> load_type(const handle &handle) {
    make_caster<T> conv;
    load_type(conv, handle);
    return conv;
}

PYBIND11_NAMESPACE_END(detail)

// pytype -> C++ type
template <typename T, detail::enable_if_t<!detail::is_pyobject<T>::value, int> = 0>
T cast(const handle &handle) {
    using namespace detail;
    static_assert(!cast_is_temporary_value_reference<T>::value,
            "Unable to cast type to reference: value is local to type caster");
    return cast_op<T>(load_type<T>(handle));
}

// pytype -> pytype (calls converting constructor)
template <typename T, detail::enable_if_t<detail::is_pyobject<T>::value, int> = 0>
T cast(const handle &handle) { return T(reinterpret_borrow<object>(handle)); }

// C++ type -> py::object
template <typename T, detail::enable_if_t<!detail::is_pyobject<T>::value, int> = 0>
object cast(T &&value, return_value_policy policy = return_value_policy::automatic_reference,
            handle parent = handle()) {
    using no_ref_T = typename std::remove_reference<T>::type;
    if (policy == return_value_policy::automatic)
        policy = std::is_pointer<no_ref_T>::value ? return_value_policy::take_ownership :
                 std::is_lvalue_reference<T>::value ? return_value_policy::copy : return_value_policy::move;
    else if (policy == return_value_policy::automatic_reference)
        policy = std::is_pointer<no_ref_T>::value ? return_value_policy::reference :
                 std::is_lvalue_reference<T>::value ? return_value_policy::copy : return_value_policy::move;
    return reinterpret_steal<object>(detail::make_caster<T>::cast(std::forward<T>(value), policy, parent));
}

template <typename T> T handle::cast() const { return pybind11::cast<T>(*this); }
template <> inline void handle::cast() const { return; }

template <typename T>
detail::enable_if_t<!detail::move_never<T>::value, T> move(object &&obj) {
    if (obj.ref_count() > 1)
#if defined(NDEBUG)
        throw cast_error("Unable to cast Python instance to C++ rvalue: instance has multiple references"
            " (compile in debug mode for details)");
#else
        throw cast_error("Unable to move from Python " + (std::string) str(obj.get_type()) +
                " instance to C++ " + type_id<T>() + " instance: instance has multiple references");
#endif

    // Move into a temporary and return that, because the reference may be a local value of `conv`
    T ret = std::move(detail::load_type<T>(obj).operator T&());
    return ret;
}

// Calling cast() on an rvalue calls pybind11::cast with the object rvalue, which does:
// - If we have to move (because T has no copy constructor), do it.  This will fail if the moved
//   object has multiple references, but trying to copy will fail to compile.
// - If both movable and copyable, check ref count: if 1, move; otherwise copy
// - Otherwise (not movable), copy.
template <typename T> detail::enable_if_t<detail::move_always<T>::value, T> cast(object &&object) {
    return move<T>(std::move(object));
}
template <typename T> detail::enable_if_t<detail::move_if_unreferenced<T>::value, T> cast(object &&object) {
    if (object.ref_count() > 1)
        return cast<T>(object);
    else
        return move<T>(std::move(object));
}
template <typename T> detail::enable_if_t<detail::move_never<T>::value, T> cast(object &&object) {
    return cast<T>(object);
}

template <typename T> T object::cast() const & { return pybind11::cast<T>(*this); }
template <typename T> T object::cast() && { return pybind11::cast<T>(std::move(*this)); }
template <> inline void object::cast() const & { return; }
template <> inline void object::cast() && { return; }

PYBIND11_NAMESPACE_BEGIN(detail)

// Declared in pytypes.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int>>
object object_or_cast(T &&o) { return pybind11::cast(std::forward<T>(o)); }

struct overload_unused {}; // Placeholder type for the unneeded (and dead code) static variable in the OVERLOAD_INT macro
template <typename ret_type> using overload_caster_t = conditional_t<
    cast_is_temporary_value_reference<ret_type>::value, make_caster<ret_type>, overload_unused>;

// Trampoline use: for reference/pointer types to value-converted values, we do a value cast, then
// store the result in the given variable.  For other types, this is a no-op.
template <typename T> enable_if_t<cast_is_temporary_value_reference<T>::value, T> cast_ref(object &&o, make_caster<T> &caster) {
    return cast_op<T>(load_type(caster, o));
}
template <typename T> enable_if_t<!cast_is_temporary_value_reference<T>::value, T> cast_ref(object &&, overload_unused &) {
    pybind11_fail("Internal error: cast_ref fallback invoked"); }

// Trampoline use: Having a pybind11::cast with an invalid reference type is going to static_assert, even
// though if it's in dead code, so we provide a "trampoline" to pybind11::cast that only does anything in
// cases where pybind11::cast is valid.
template <typename T> enable_if_t<!cast_is_temporary_value_reference<T>::value, T> cast_safe(object &&o) {
    return pybind11::cast<T>(std::move(o)); }
template <typename T> enable_if_t<cast_is_temporary_value_reference<T>::value, T> cast_safe(object &&) {
    pybind11_fail("Internal error: cast_safe fallback invoked"); }
template <> inline void cast_safe<void>(object &&) {}

PYBIND11_NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference>
tuple make_tuple() { return tuple(0); }

template <return_value_policy policy = return_value_policy::automatic_reference,
          typename... Args> tuple make_tuple(Args&&... args_) {
    constexpr size_t size = sizeof...(Args);
    std::array<object, size> args {
        { reinterpret_steal<object>(detail::make_caster<Args>::cast(
            std::forward<Args>(args_), policy, nullptr))... }
    };
    for (size_t i = 0; i < args.size(); i++) {
        if (!args[i]) {
#if defined(NDEBUG)
            throw cast_error("make_tuple(): unable to convert arguments to Python object (compile in debug mode for details)");
#else
            std::array<std::string, size> argtypes { {type_id<Args>()...} };
            throw cast_error("make_tuple(): unable to convert argument of type '" +
                argtypes[i] + "' to Python object");
#endif
        }
    }
    tuple result(size);
    int counter = 0;
    for (auto &arg_value : args)
        PyTuple_SET_ITEM(result.ptr(), counter++, arg_value.release().ptr());
    return result;
}

/// \ingroup annotations
/// Annotation for arguments
struct arg {
    /// Constructs an argument with the name of the argument; if null or omitted, this is a positional argument.
    constexpr explicit arg(const char *name = nullptr) : name(name), flag_noconvert(false), flag_none(true) { }
    /// Assign a value to this argument
    template <typename T> arg_v operator=(T &&value) const;
    /// Indicate that the type should not be converted in the type caster
    arg &noconvert(bool flag = true) { flag_noconvert = flag; return *this; }
    /// Indicates that the argument should/shouldn't allow None (e.g. for nullable pointer args)
    arg &none(bool flag = true) { flag_none = flag; return *this; }

    const char *name; ///< If non-null, this is a named kwargs argument
    bool flag_noconvert : 1; ///< If set, do not allow conversion (requires a supporting type caster!)
    bool flag_none : 1; ///< If set (the default), allow None to be passed to this argument
};

/// \ingroup annotations
/// Annotation for arguments with values
struct arg_v : arg {
private:
    template <typename T>
    arg_v(arg &&base, T &&x, const char *descr = nullptr)
        : arg(base),
          value(reinterpret_steal<object>(
              detail::make_caster<T>::cast(x, return_value_policy::automatic, {})
          )),
          descr(descr)
#if !defined(NDEBUG)
        , type(type_id<T>())
#endif
    { }

public:
    /// Direct construction with name, default, and description
    template <typename T>
    arg_v(const char *name, T &&x, const char *descr = nullptr)
        : arg_v(arg(name), std::forward<T>(x), descr) { }

    /// Called internally when invoking `py::arg("a") = value`
    template <typename T>
    arg_v(const arg &base, T &&x, const char *descr = nullptr)
        : arg_v(arg(base), std::forward<T>(x), descr) { }

    /// Same as `arg::noconvert()`, but returns *this as arg_v&, not arg&
    arg_v &noconvert(bool flag = true) { arg::noconvert(flag); return *this; }

    /// Same as `arg::nonone()`, but returns *this as arg_v&, not arg&
    arg_v &none(bool flag = true) { arg::none(flag); return *this; }

    /// The default value
    object value;
    /// The (optional) description of the default value
    const char *descr;
#if !defined(NDEBUG)
    /// The C++ type name of the default value (only available when compiled in debug mode)
    std::string type;
#endif
};

/// \ingroup annotations
/// Annotation indicating that all following arguments are keyword-only; the is the equivalent of an
/// unnamed '*' argument (in Python 3)
struct kwonly {};

template <typename T>
arg_v arg::operator=(T &&value) const { return {std::move(*this), std::forward<T>(value)}; }

/// Alias for backward compatibility -- to be removed in version 2.0
template <typename /*unused*/> using arg_t = arg_v;

inline namespace literals {
/** \rst
    String literal version of `arg`
 \endrst */
constexpr arg operator"" _a(const char *name, size_t) { return arg(name); }
}

PYBIND11_NAMESPACE_BEGIN(detail)

// forward declaration (definition in attr.h)
struct function_record;

/// Internal data associated with a single function call
struct function_call {
    function_call(const function_record &f, handle p); // Implementation in attr.h

    /// The function data:
    const function_record &func;

    /// Arguments passed to the function:
    std::vector<handle> args;

    /// The `convert` value the arguments should be loaded with
    std::vector<bool> args_convert;

    /// Extra references for the optional `py::args` and/or `py::kwargs` arguments (which, if
    /// present, are also in `args` but without a reference).
    object args_ref, kwargs_ref;

    /// The parent, if any
    handle parent;

    /// If this is a call to an initializer, this argument contains `self`
    handle init_self;
};


/// Helper class which loads arguments for C++ functions called from Python
template <typename... Args>
class argument_loader {
    using indices = make_index_sequence<sizeof...(Args)>;

    template <typename Arg> using argument_is_args   = std::is_same<intrinsic_t<Arg>, args>;
    template <typename Arg> using argument_is_kwargs = std::is_same<intrinsic_t<Arg>, kwargs>;
    // Get args/kwargs argument positions relative to the end of the argument list:
    static constexpr auto args_pos = constexpr_first<argument_is_args, Args...>() - (int) sizeof...(Args),
                        kwargs_pos = constexpr_first<argument_is_kwargs, Args...>() - (int) sizeof...(Args);

    static constexpr bool args_kwargs_are_last = kwargs_pos >= - 1 && args_pos >= kwargs_pos - 1;

    static_assert(args_kwargs_are_last, "py::args/py::kwargs are only permitted as the last argument(s) of a function");

public:
    static constexpr bool has_kwargs = kwargs_pos < 0;
    static constexpr bool has_args = args_pos < 0;

    static constexpr auto arg_names = concat(type_descr(make_caster<Args>::name)...);

    bool load_args(function_call &call) {
        return load_impl_sequence(call, indices{});
    }

    template <typename Return, typename Guard, typename Func>
    enable_if_t<!std::is_void<Return>::value, Return> call(Func &&f) && {
        return std::move(*this).template call_impl<Return>(std::forward<Func>(f), indices{}, Guard{});
    }

    template <typename Return, typename Guard, typename Func>
    enable_if_t<std::is_void<Return>::value, void_type> call(Func &&f) && {
        std::move(*this).template call_impl<Return>(std::forward<Func>(f), indices{}, Guard{});
        return void_type();
    }

private:

    static bool load_impl_sequence(function_call &, index_sequence<>) { return true; }

    template <size_t... Is>
    bool load_impl_sequence(function_call &call, index_sequence<Is...>) {
#ifdef __cpp_fold_expressions
        if ((... || !std::get<Is>(argcasters).load(call.args[Is], call.args_convert[Is])))
            return false;
#else
        for (bool r : {std::get<Is>(argcasters).load(call.args[Is], call.args_convert[Is])...})
            if (!r)
                return false;
#endif
        return true;
    }

    template <typename Return, typename Func, size_t... Is, typename Guard>
    Return call_impl(Func &&f, index_sequence<Is...>, Guard &&) && {
        return std::forward<Func>(f)(cast_op<Args>(std::move(std::get<Is>(argcasters)))...);
    }

    std::tuple<make_caster<Args>...> argcasters;
};

/// Helper class which collects only positional arguments for a Python function call.
/// A fancier version below can collect any argument, but this one is optimal for simple calls.
template <return_value_policy policy>
class simple_collector {
public:
    template <typename... Ts>
    explicit simple_collector(Ts &&...values)
        : m_args(pybind11::make_tuple<policy>(std::forward<Ts>(values)...)) { }

    const tuple &args() const & { return m_args; }
    dict kwargs() const { return {}; }

    tuple args() && { return std::move(m_args); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        PyObject *result = PyObject_CallObject(ptr, m_args.ptr());
        if (!result)
            throw error_already_set();
        return reinterpret_steal<object>(result);
    }

private:
    tuple m_args;
};

/// Helper class which collects positional, keyword, * and ** arguments for a Python function call
template <return_value_policy policy>
class unpacking_collector {
public:
    template <typename... Ts>
    explicit unpacking_collector(Ts &&...values) {
        // Tuples aren't (easily) resizable so a list is needed for collection,
        // but the actual function call strictly requires a tuple.
        auto args_list = list();
        int _[] = { 0, (process(args_list, std::forward<Ts>(values)), 0)... };
        ignore_unused(_);

        m_args = std::move(args_list);
    }

    const tuple &args() const & { return m_args; }
    const dict &kwargs() const & { return m_kwargs; }

    tuple args() && { return std::move(m_args); }
    dict kwargs() && { return std::move(m_kwargs); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        PyObject *result = PyObject_Call(ptr, m_args.ptr(), m_kwargs.ptr());
        if (!result)
            throw error_already_set();
        return reinterpret_steal<object>(result);
    }

private:
    template <typename T>
    void process(list &args_list, T &&x) {
        auto o = reinterpret_steal<object>(detail::make_caster<T>::cast(std::forward<T>(x), policy, {}));
        if (!o) {
#if defined(NDEBUG)
            argument_cast_error();
#else
            argument_cast_error(std::to_string(args_list.size()), type_id<T>());
#endif
        }
        args_list.append(o);
    }

    void process(list &args_list, detail::args_proxy ap) {
        for (const auto &a : ap)
            args_list.append(a);
    }

    void process(list &/*args_list*/, arg_v a) {
        if (!a.name)
#if defined(NDEBUG)
            nameless_argument_error();
#else
            nameless_argument_error(a.type);
#endif

        if (m_kwargs.contains(a.name)) {
#if defined(NDEBUG)
            multiple_values_error();
#else
            multiple_values_error(a.name);
#endif
        }
        if (!a.value) {
#if defined(NDEBUG)
            argument_cast_error();
#else
            argument_cast_error(a.name, a.type);
#endif
        }
        m_kwargs[a.name] = a.value;
    }

    void process(list &/*args_list*/, detail::kwargs_proxy kp) {
        if (!kp)
            return;
        for (const auto &k : reinterpret_borrow<dict>(kp)) {
            if (m_kwargs.contains(k.first)) {
#if defined(NDEBUG)
                multiple_values_error();
#else
                multiple_values_error(str(k.first));
#endif
            }
            m_kwargs[k.first] = k.second;
        }
    }

    [[noreturn]] static void nameless_argument_error() {
        throw type_error("Got kwargs without a name; only named arguments "
                         "may be passed via py::arg() to a python function call. "
                         "(compile in debug mode for details)");
    }
    [[noreturn]] static void nameless_argument_error(std::string type) {
        throw type_error("Got kwargs without a name of type '" + type + "'; only named "
                         "arguments may be passed via py::arg() to a python function call. ");
    }
    [[noreturn]] static void multiple_values_error() {
        throw type_error("Got multiple values for keyword argument "
                         "(compile in debug mode for details)");
    }

    [[noreturn]] static void multiple_values_error(std::string name) {
        throw type_error("Got multiple values for keyword argument '" + name + "'");
    }

    [[noreturn]] static void argument_cast_error() {
        throw cast_error("Unable to convert call argument to Python object "
                         "(compile in debug mode for details)");
    }

    [[noreturn]] static void argument_cast_error(std::string name, std::string type) {
        throw cast_error("Unable to convert call argument '" + name
                         + "' of type '" + type + "' to Python object");
    }

private:
    tuple m_args;
    dict m_kwargs;
};

/// Collect only positional arguments for a Python function call
template <return_value_policy policy, typename... Args,
          typename = enable_if_t<all_of<is_positional<Args>...>::value>>
simple_collector<policy> collect_arguments(Args &&...args) {
    return simple_collector<policy>(std::forward<Args>(args)...);
}

/// Collect all arguments, including keywords and unpacking (only instantiated when needed)
template <return_value_policy policy, typename... Args,
          typename = enable_if_t<!all_of<is_positional<Args>...>::value>>
unpacking_collector<policy> collect_arguments(Args &&...args) {
    // Following argument order rules for generalized unpacking according to PEP 448
    static_assert(
        constexpr_last<is_positional, Args...>() < constexpr_first<is_keyword_or_ds, Args...>()
        && constexpr_last<is_s_unpacking, Args...>() < constexpr_first<is_ds_unpacking, Args...>(),
        "Invalid function call: positional args must precede keywords and ** unpacking; "
        "* unpacking must precede ** unpacking"
    );
    return unpacking_collector<policy>(std::forward<Args>(args)...);
}

template <typename Derived>
template <return_value_policy policy, typename... Args>
object object_api<Derived>::operator()(Args &&...args) const {
    return detail::collect_arguments<policy>(std::forward<Args>(args)...).call(derived().ptr());
}

template <typename Derived>
template <return_value_policy policy, typename... Args>
object object_api<Derived>::call(Args &&...args) const {
    return operator()<policy>(std::forward<Args>(args)...);
}

PYBIND11_NAMESPACE_END(detail)

#define PYBIND11_MAKE_OPAQUE(...) \
    namespace pybind11 { namespace detail { \
        template<> class type_caster<__VA_ARGS__> : public type_caster_base<__VA_ARGS__> { }; \
    }}

/// Lets you pass a type containing a `,` through a macro parameter without needing a separate
/// typedef, e.g.: `PYBIND11_OVERLOAD(PYBIND11_TYPE(ReturnType<A, B>), PYBIND11_TYPE(Parent<C, D>), f, arg)`
#define PYBIND11_TYPE(...) __VA_ARGS__

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "cast.h" ============


//========= begin of #include "attr.h" ============

/*
    pybind11/attr.h: Infrastructure for processing custom
    type and function attributes

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "cast.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// \addtogroup annotations
/// @{

/// Annotation for methods
struct is_method { handle class_; is_method(const handle &c) : class_(c) { } };

/// Annotation for operators
struct is_operator { };

/// Annotation for classes that cannot be subclassed
struct is_final { };

/// Annotation for parent scope
struct scope { handle value; scope(const handle &s) : value(s) { } };

/// Annotation for documentation
struct doc { const char *value; doc(const char *value) : value(value) { } };

/// Annotation for function names
struct name { const char *value; name(const char *value) : value(value) { } };

/// Annotation indicating that a function is an overload associated with a given "sibling"
struct sibling { handle value; sibling(const handle &value) : value(value.ptr()) { } };

/// Annotation indicating that a class derives from another given type
template <typename T> struct base {
    PYBIND11_DEPRECATED("base<T>() was deprecated in favor of specifying 'T' as a template argument to class_")
    base() { }
};

/// Keep patient alive while nurse lives
template <size_t Nurse, size_t Patient> struct keep_alive { };

/// Annotation indicating that a class is involved in a multiple inheritance relationship
struct multiple_inheritance { };

/// Annotation which enables dynamic attributes, i.e. adds `__dict__` to a class
struct dynamic_attr { };

/// Annotation which enables the buffer protocol for a type
struct buffer_protocol { };

/// Annotation which requests that a special metaclass is created for a type
struct metaclass {
    handle value;

    PYBIND11_DEPRECATED("py::metaclass() is no longer required. It's turned on by default now.")
    metaclass() {}

    /// Override pybind11's default metaclass
    explicit metaclass(handle value) : value(value) { }
};

/// Annotation that marks a class as local to the module:
struct module_local { const bool value; constexpr module_local(bool v = true) : value(v) { } };

/// Annotation to mark enums as an arithmetic type
struct arithmetic { };

/** \rst
    A call policy which places one or more guard variables (``Ts...``) around the function call.

    For example, this definition:

    .. code-block:: cpp

        m.def("foo", foo, py::call_guard<T>());

    is equivalent to the following pseudocode:

    .. code-block:: cpp

        m.def("foo", [](args...) {
            T scope_guard;
            return foo(args...); // forwarded arguments
        });
 \endrst */
template <typename... Ts> struct call_guard;

template <> struct call_guard<> { using type = detail::void_type; };

template <typename T>
struct call_guard<T> {
    static_assert(std::is_default_constructible<T>::value,
                  "The guard type must be default constructible");

    using type = T;
};

template <typename T, typename... Ts>
struct call_guard<T, Ts...> {
    struct type {
        T guard{}; // Compose multiple guard types with left-to-right default-constructor order
        typename call_guard<Ts...>::type next{};
    };
};

/// @} annotations

PYBIND11_NAMESPACE_BEGIN(detail)
/* Forward declarations */
enum op_id : int;
enum op_type : int;
struct undefined_t;
template <op_id id, op_type ot, typename L = undefined_t, typename R = undefined_t> struct op_;
inline void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret);

/// Internal data structure which holds metadata about a keyword argument
struct argument_record {
    const char *name;  ///< Argument name
    const char *descr; ///< Human-readable version of the argument value
    handle value;      ///< Associated Python object
    bool convert : 1;  ///< True if the argument is allowed to convert when loading
    bool none : 1;     ///< True if None is allowed when loading

    argument_record(const char *name, const char *descr, handle value, bool convert, bool none)
        : name(name), descr(descr), value(value), convert(convert), none(none) { }
};

/// Internal data structure which holds metadata about a bound function (signature, overloads, etc.)
struct function_record {
    function_record()
        : is_constructor(false), is_new_style_constructor(false), is_stateless(false),
          is_operator(false), is_method(false),
          has_args(false), has_kwargs(false), has_kwonly_args(false) { }

    /// Function name
    char *name = nullptr; /* why no C++ strings? They generate heavier code.. */

    // User-specified documentation string
    char *doc = nullptr;

    /// Human-readable version of the function signature
    char *signature = nullptr;

    /// List of registered keyword arguments
    std::vector<argument_record> args;

    /// Pointer to lambda function which converts arguments and performs the actual call
    handle (*impl) (function_call &) = nullptr;

    /// Storage for the wrapped function pointer and captured data, if any
    void *data[3] = { };

    /// Pointer to custom destructor for 'data' (if needed)
    void (*free_data) (function_record *ptr) = nullptr;

    /// Return value policy associated with this function
    return_value_policy policy = return_value_policy::automatic;

    /// True if name == '__init__'
    bool is_constructor : 1;

    /// True if this is a new-style `__init__` defined in `detail/init.h`
    bool is_new_style_constructor : 1;

    /// True if this is a stateless function pointer
    bool is_stateless : 1;

    /// True if this is an operator (__add__), etc.
    bool is_operator : 1;

    /// True if this is a method
    bool is_method : 1;

    /// True if the function has a '*args' argument
    bool has_args : 1;

    /// True if the function has a '**kwargs' argument
    bool has_kwargs : 1;

    /// True once a 'py::kwonly' is encountered (any following args are keyword-only)
    bool has_kwonly_args : 1;

    /// Number of arguments (including py::args and/or py::kwargs, if present)
    std::uint16_t nargs;

    /// Number of trailing arguments (counted in `nargs`) that are keyword-only
    std::uint16_t nargs_kwonly = 0;

    /// Python method object
    PyMethodDef *def = nullptr;

    /// Python handle to the parent scope (a class or a module)
    handle scope;

    /// Python handle to the sibling function representing an overload chain
    handle sibling;

    /// Pointer to next overload
    function_record *next = nullptr;
};

/// Special data structure which (temporarily) holds metadata about a bound class
struct type_record {
    PYBIND11_NOINLINE type_record()
        : multiple_inheritance(false), dynamic_attr(false), buffer_protocol(false),
          default_holder(true), module_local(false), is_final(false) { }

    /// Handle to the parent scope
    handle scope;

    /// Name of the class
    const char *name = nullptr;

    // Pointer to RTTI type_info data structure
    const std::type_info *type = nullptr;

    /// How large is the underlying C++ type?
    size_t type_size = 0;

    /// What is the alignment of the underlying C++ type?
    size_t type_align = 0;

    /// How large is the type's holder?
    size_t holder_size = 0;

    /// The global operator new can be overridden with a class-specific variant
    void *(*operator_new)(size_t) = nullptr;

    /// Function pointer to class_<..>::init_instance
    void (*init_instance)(instance *, const void *) = nullptr;

    /// Function pointer to class_<..>::dealloc
    void (*dealloc)(detail::value_and_holder &) = nullptr;

    /// List of base classes of the newly created type
    list bases;

    /// Optional docstring
    const char *doc = nullptr;

    /// Custom metaclass (optional)
    handle metaclass;

    /// Multiple inheritance marker
    bool multiple_inheritance : 1;

    /// Does the class manage a __dict__?
    bool dynamic_attr : 1;

    /// Does the class implement the buffer protocol?
    bool buffer_protocol : 1;

    /// Is the default (unique_ptr) holder type used?
    bool default_holder : 1;

    /// Is the class definition local to the module shared object?
    bool module_local : 1;

    /// Is the class inheritable from python classes?
    bool is_final : 1;

    PYBIND11_NOINLINE void add_base(const std::type_info &base, void *(*caster)(void *)) {
        auto base_info = detail::get_type_info(base, false);
        if (!base_info) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name) +
                          "\" referenced unknown base type \"" + tname + "\"");
        }

        if (default_holder != base_info->default_holder) {
            std::string tname(base.name());
            detail::clean_type_id(tname);
            pybind11_fail("generic_type: type \"" + std::string(name) + "\" " +
                    (default_holder ? "does not have" : "has") +
                    " a non-default holder type while its base \"" + tname + "\" " +
                    (base_info->default_holder ? "does not" : "does"));
        }

        bases.append((PyObject *) base_info->type);

        if (base_info->type->tp_dictoffset != 0)
            dynamic_attr = true;

        if (caster)
            base_info->implicit_casts.emplace_back(type, caster);
    }
};

inline function_call::function_call(const function_record &f, handle p) :
        func(f), parent(p) {
    args.reserve(f.nargs);
    args_convert.reserve(f.nargs);
}

/// Tag for a new-style `__init__` defined in `detail/init.h`
struct is_new_style_constructor { };

/**
 * Partial template specializations to process custom attributes provided to
 * cpp_function_ and class_. These are either used to initialize the respective
 * fields in the type_record and function_record data structures or executed at
 * runtime to deal with custom call policies (e.g. keep_alive).
 */
template <typename T, typename SFINAE = void> struct process_attribute;

template <typename T> struct process_attribute_default {
    /// Default implementation: do nothing
    static void init(const T &, function_record *) { }
    static void init(const T &, type_record *) { }
    static void precall(function_call &) { }
    static void postcall(function_call &, handle) { }
};

/// Process an attribute specifying the function's name
template <> struct process_attribute<name> : process_attribute_default<name> {
    static void init(const name &n, function_record *r) { r->name = const_cast<char *>(n.value); }
};

/// Process an attribute specifying the function's docstring
template <> struct process_attribute<doc> : process_attribute_default<doc> {
    static void init(const doc &n, function_record *r) { r->doc = const_cast<char *>(n.value); }
};

/// Process an attribute specifying the function's docstring (provided as a C-style string)
template <> struct process_attribute<const char *> : process_attribute_default<const char *> {
    static void init(const char *d, function_record *r) { r->doc = const_cast<char *>(d); }
    static void init(const char *d, type_record *r) { r->doc = const_cast<char *>(d); }
};
template <> struct process_attribute<char *> : process_attribute<const char *> { };

/// Process an attribute indicating the function's return value policy
template <> struct process_attribute<return_value_policy> : process_attribute_default<return_value_policy> {
    static void init(const return_value_policy &p, function_record *r) { r->policy = p; }
};

/// Process an attribute which indicates that this is an overloaded function associated with a given sibling
template <> struct process_attribute<sibling> : process_attribute_default<sibling> {
    static void init(const sibling &s, function_record *r) { r->sibling = s.value; }
};

/// Process an attribute which indicates that this function is a method
template <> struct process_attribute<is_method> : process_attribute_default<is_method> {
    static void init(const is_method &s, function_record *r) { r->is_method = true; r->scope = s.class_; }
};

/// Process an attribute which indicates the parent scope of a method
template <> struct process_attribute<scope> : process_attribute_default<scope> {
    static void init(const scope &s, function_record *r) { r->scope = s.value; }
};

/// Process an attribute which indicates that this function is an operator
template <> struct process_attribute<is_operator> : process_attribute_default<is_operator> {
    static void init(const is_operator &, function_record *r) { r->is_operator = true; }
};

template <> struct process_attribute<is_new_style_constructor> : process_attribute_default<is_new_style_constructor> {
    static void init(const is_new_style_constructor &, function_record *r) { r->is_new_style_constructor = true; }
};

inline void process_kwonly_arg(const arg &a, function_record *r) {
    if (!a.name || strlen(a.name) == 0)
        pybind11_fail("arg(): cannot specify an unnamed argument after an kwonly() annotation");
    ++r->nargs_kwonly;
}

/// Process a keyword argument attribute (*without* a default value)
template <> struct process_attribute<arg> : process_attribute_default<arg> {
    static void init(const arg &a, function_record *r) {
        if (r->is_method && r->args.empty())
            r->args.emplace_back("self", nullptr, handle(), true /*convert*/, false /*none not allowed*/);
        r->args.emplace_back(a.name, nullptr, handle(), !a.flag_noconvert, a.flag_none);

        if (r->has_kwonly_args) process_kwonly_arg(a, r);
    }
};

/// Process a keyword argument attribute (*with* a default value)
template <> struct process_attribute<arg_v> : process_attribute_default<arg_v> {
    static void init(const arg_v &a, function_record *r) {
        if (r->is_method && r->args.empty())
            r->args.emplace_back("self", nullptr /*descr*/, handle() /*parent*/, true /*convert*/, false /*none not allowed*/);

        if (!a.value) {
#if !defined(NDEBUG)
            std::string descr("'");
            if (a.name) descr += std::string(a.name) + ": ";
            descr += a.type + "'";
            if (r->is_method) {
                if (r->name)
                    descr += " in method '" + (std::string) str(r->scope) + "." + (std::string) r->name + "'";
                else
                    descr += " in method of '" + (std::string) str(r->scope) + "'";
            } else if (r->name) {
                descr += " in function '" + (std::string) r->name + "'";
            }
            pybind11_fail("arg(): could not convert default argument "
                          + descr + " into a Python object (type not registered yet?)");
#else
            pybind11_fail("arg(): could not convert default argument "
                          "into a Python object (type not registered yet?). "
                          "Compile in debug mode for more information.");
#endif
        }
        r->args.emplace_back(a.name, a.descr, a.value.inc_ref(), !a.flag_noconvert, a.flag_none);

        if (r->has_kwonly_args) process_kwonly_arg(a, r);
    }
};

/// Process a keyword-only-arguments-follow pseudo argument
template <> struct process_attribute<kwonly> : process_attribute_default<kwonly> {
    static void init(const kwonly &, function_record *r) {
        r->has_kwonly_args = true;
    }
};

/// Process a parent class attribute.  Single inheritance only (class_ itself already guarantees that)
template <typename T>
struct process_attribute<T, enable_if_t<is_pyobject<T>::value>> : process_attribute_default<handle> {
    static void init(const handle &h, type_record *r) { r->bases.append(h); }
};

/// Process a parent class attribute (deprecated, does not support multiple inheritance)
template <typename T>
struct process_attribute<base<T>> : process_attribute_default<base<T>> {
    static void init(const base<T> &, type_record *r) { r->add_base(typeid(T), nullptr); }
};

/// Process a multiple inheritance attribute
template <>
struct process_attribute<multiple_inheritance> : process_attribute_default<multiple_inheritance> {
    static void init(const multiple_inheritance &, type_record *r) { r->multiple_inheritance = true; }
};

template <>
struct process_attribute<dynamic_attr> : process_attribute_default<dynamic_attr> {
    static void init(const dynamic_attr &, type_record *r) { r->dynamic_attr = true; }
};

template <>
struct process_attribute<is_final> : process_attribute_default<is_final> {
    static void init(const is_final &, type_record *r) { r->is_final = true; }
};

template <>
struct process_attribute<buffer_protocol> : process_attribute_default<buffer_protocol> {
    static void init(const buffer_protocol &, type_record *r) { r->buffer_protocol = true; }
};

template <>
struct process_attribute<metaclass> : process_attribute_default<metaclass> {
    static void init(const metaclass &m, type_record *r) { r->metaclass = m.value; }
};

template <>
struct process_attribute<module_local> : process_attribute_default<module_local> {
    static void init(const module_local &l, type_record *r) { r->module_local = l.value; }
};

/// Process an 'arithmetic' attribute for enums (does nothing here)
template <>
struct process_attribute<arithmetic> : process_attribute_default<arithmetic> {};

template <typename... Ts>
struct process_attribute<call_guard<Ts...>> : process_attribute_default<call_guard<Ts...>> { };

/**
 * Process a keep_alive call policy -- invokes keep_alive_impl during the
 * pre-call handler if both Nurse, Patient != 0 and use the post-call handler
 * otherwise
 */
template <size_t Nurse, size_t Patient> struct process_attribute<keep_alive<Nurse, Patient>> : public process_attribute_default<keep_alive<Nurse, Patient>> {
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N != 0 && P != 0, int> = 0>
    static void precall(function_call &call) { keep_alive_impl(Nurse, Patient, call, handle()); }
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N != 0 && P != 0, int> = 0>
    static void postcall(function_call &, handle) { }
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N == 0 || P == 0, int> = 0>
    static void precall(function_call &) { }
    template <size_t N = Nurse, size_t P = Patient, enable_if_t<N == 0 || P == 0, int> = 0>
    static void postcall(function_call &call, handle ret) { keep_alive_impl(Nurse, Patient, call, ret); }
};

/// Recursively iterate over variadic template arguments
template <typename... Args> struct process_attributes {
    static void init(const Args&... args, function_record *r) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::init(args, r), 0) ... };
        ignore_unused(unused);
    }
    static void init(const Args&... args, type_record *r) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::init(args, r), 0) ... };
        ignore_unused(unused);
    }
    static void precall(function_call &call) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::precall(call), 0) ... };
        ignore_unused(unused);
    }
    static void postcall(function_call &call, handle fn_ret) {
        int unused[] = { 0, (process_attribute<typename std::decay<Args>::type>::postcall(call, fn_ret), 0) ... };
        ignore_unused(unused);
    }
};

template <typename T>
using is_call_guard = is_instantiation<call_guard, T>;

/// Extract the ``type`` from the first `call_guard` in `Extras...` (or `void_type` if none found)
template <typename... Extra>
using extract_guard_t = typename exactly_one_t<is_call_guard, call_guard<>, Extra...>::type;

/// Check the number of named arguments at compile time
template <typename... Extra,
          size_t named = constexpr_sum(std::is_base_of<arg, Extra>::value...),
          size_t self  = constexpr_sum(std::is_same<is_method, Extra>::value...)>
constexpr bool expected_num_args(size_t nargs, bool has_args, bool has_kwargs) {
    return named == 0 || (self + named + has_args + has_kwargs) == nargs;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "attr.h" ============


//========= begin of #include "options.h" ============

/*
    pybind11/options.h: global settings that are configurable at runtime.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "detail/common.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class options {
public:

    // Default RAII constructor, which leaves settings as they currently are.
    options() : previous_state(global_state()) {}

    // Class is non-copyable.
    options(const options&) = delete;
    options& operator=(const options&) = delete;

    // Destructor, which restores settings that were in effect before.
    ~options() {
        global_state() = previous_state;
    }

    // Setter methods (affect the global state):

    options& disable_user_defined_docstrings() & { global_state().show_user_defined_docstrings = false; return *this; }

    options& enable_user_defined_docstrings() & { global_state().show_user_defined_docstrings = true; return *this; }

    options& disable_function_signatures() & { global_state().show_function_signatures = false; return *this; }

    options& enable_function_signatures() & { global_state().show_function_signatures = true; return *this; }

    // Getter methods (return the global state):

    static bool show_user_defined_docstrings() { return global_state().show_user_defined_docstrings; }

    static bool show_function_signatures() { return global_state().show_function_signatures; }

    // This type is not meant to be allocated on the heap.
    void* operator new(size_t) = delete;

private:

    struct state {
        bool show_user_defined_docstrings = true;  //< Include user-supplied texts in docstrings.
        bool show_function_signatures = true;      //< Include auto-generated function signatures in docstrings.
    };

    static state &global_state() {
        static state instance;
        return instance;
    }

    state previous_state;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "options.h" ============


//========= begin of #include "detail/class.h" ============

/*
    pybind11/detail/class.h: Python C API implementation details for py::class_

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "../attr.h"
// ans ignore: #include "../options.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#if PY_VERSION_HEX >= 0x03030000 && !defined(PYPY_VERSION)
#  define PYBIND11_BUILTIN_QUALNAME
#  define PYBIND11_SET_OLDPY_QUALNAME(obj, nameobj)
#else
// In pre-3.3 Python, we still set __qualname__ so that we can produce reliable function type
// signatures; in 3.3+ this macro expands to nothing:
#  define PYBIND11_SET_OLDPY_QUALNAME(obj, nameobj) setattr((PyObject *) obj, "__qualname__", nameobj)
#endif

inline PyTypeObject *type_incref(PyTypeObject *type) {
    Py_INCREF(type);
    return type;
}

#if !defined(PYPY_VERSION)

/// `pybind11_static_property.__get__()`: Always pass the class instead of the instance.
extern "C" inline PyObject *pybind11_static_get(PyObject *self, PyObject * /*ob*/, PyObject *cls) {
    return PyProperty_Type.tp_descr_get(self, cls, cls);
}

/// `pybind11_static_property.__set__()`: Just like the above `__get__()`.
extern "C" inline int pybind11_static_set(PyObject *self, PyObject *obj, PyObject *value) {
    PyObject *cls = PyType_Check(obj) ? obj : (PyObject *) Py_TYPE(obj);
    return PyProperty_Type.tp_descr_set(self, cls, value);
}

/** A `static_property` is the same as a `property` but the `__get__()` and `__set__()`
    methods are modified to always use the object type instead of a concrete instance.
    Return value: New reference. */
inline PyTypeObject *make_static_property_type() {
    constexpr auto *name = "pybind11_static_property";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("make_static_property_type(): error allocating type!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyProperty_Type);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
    type->tp_descr_get = pybind11_static_get;
    type->tp_descr_set = pybind11_static_set;

    if (PyType_Ready(type) < 0)
        pybind11_fail("make_static_property_type(): failure in PyType_Ready()!");

    setattr((PyObject *) type, "__module__", str("pybind11_builtins"));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    return type;
}

#else // PYPY

/** PyPy has some issues with the above C API, so we evaluate Python code instead.
    This function will only be called once so performance isn't really a concern.
    Return value: New reference. */
inline PyTypeObject *make_static_property_type() {
    auto d = dict();
    PyObject *result = PyRun_String(R"(\
        class pybind11_static_property(property):
            def __get__(self, obj, cls):
                return property.__get__(self, cls, cls)

            def __set__(self, obj, value):
                cls = obj if isinstance(obj, type) else type(obj)
                property.__set__(self, cls, value)
        )", Py_file_input, d.ptr(), d.ptr()
    );
    if (result == nullptr)
        throw error_already_set();
    Py_DECREF(result);
    return (PyTypeObject *) d["pybind11_static_property"].cast<object>().release().ptr();
}

#endif // PYPY

/** Types with static properties need to handle `Type.static_prop = x` in a specific way.
    By default, Python replaces the `static_property` itself, but for wrapped C++ types
    we need to call `static_property.__set__()` in order to propagate the new value to
    the underlying C++ data structure. */
extern "C" inline int pybind11_meta_setattro(PyObject* obj, PyObject* name, PyObject* value) {
    // Use `_PyType_Lookup()` instead of `PyObject_GetAttr()` in order to get the raw
    // descriptor (`property`) instead of calling `tp_descr_get` (`property.__get__()`).
    PyObject *descr = _PyType_Lookup((PyTypeObject *) obj, name);

    // The following assignment combinations are possible:
    //   1. `Type.static_prop = value`             --> descr_set: `Type.static_prop.__set__(value)`
    //   2. `Type.static_prop = other_static_prop` --> setattro:  replace existing `static_prop`
    //   3. `Type.regular_attribute = value`       --> setattro:  regular attribute assignment
    const auto static_prop = (PyObject *) get_internals().static_property_type;
    const auto call_descr_set = descr && PyObject_IsInstance(descr, static_prop)
                                && !PyObject_IsInstance(value, static_prop);
    if (call_descr_set) {
        // Call `static_property.__set__()` instead of replacing the `static_property`.
#if !defined(PYPY_VERSION)
        return Py_TYPE(descr)->tp_descr_set(descr, obj, value);
#else
        if (PyObject *result = PyObject_CallMethod(descr, "__set__", "OO", obj, value)) {
            Py_DECREF(result);
            return 0;
        } else {
            return -1;
        }
#endif
    } else {
        // Replace existing attribute.
        return PyType_Type.tp_setattro(obj, name, value);
    }
}

#if PY_MAJOR_VERSION >= 3
/**
 * Python 3's PyInstanceMethod_Type hides itself via its tp_descr_get, which prevents aliasing
 * methods via cls.attr("m2") = cls.attr("m1"): instead the tp_descr_get returns a plain function,
 * when called on a class, or a PyMethod, when called on an instance.  Override that behaviour here
 * to do a special case bypass for PyInstanceMethod_Types.
 */
extern "C" inline PyObject *pybind11_meta_getattro(PyObject *obj, PyObject *name) {
    PyObject *descr = _PyType_Lookup((PyTypeObject *) obj, name);
    if (descr && PyInstanceMethod_Check(descr)) {
        Py_INCREF(descr);
        return descr;
    }
    else {
        return PyType_Type.tp_getattro(obj, name);
    }
}
#endif

/// metaclass `__call__` function that is used to create all pybind11 objects.
extern "C" inline PyObject *pybind11_meta_call(PyObject *type, PyObject *args, PyObject *kwargs) {

    // use the default metaclass call to create/initialize the object
    PyObject *self = PyType_Type.tp_call(type, args, kwargs);
    if (self == nullptr) {
        return nullptr;
    }

    // This must be a pybind11 instance
    auto instance = reinterpret_cast<detail::instance *>(self);

    // Ensure that the base __init__ function(s) were called
    for (const auto &vh : values_and_holders(instance)) {
        if (!vh.holder_constructed()) {
            PyErr_Format(PyExc_TypeError, "%.200s.__init__() must be called when overriding __init__",
                         vh.type->type->tp_name);
            Py_DECREF(self);
            return nullptr;
        }
    }

    return self;
}

/** This metaclass is assigned by default to all pybind11 types and is required in order
    for static properties to function correctly. Users may override this using `py::metaclass`.
    Return value: New reference. */
inline PyTypeObject* make_default_metaclass() {
    constexpr auto *name = "pybind11_type";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
    if (!heap_type)
        pybind11_fail("make_default_metaclass(): error allocating metaclass!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyType_Type);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

    type->tp_call = pybind11_meta_call;

    type->tp_setattro = pybind11_meta_setattro;
#if PY_MAJOR_VERSION >= 3
    type->tp_getattro = pybind11_meta_getattro;
#endif

    if (PyType_Ready(type) < 0)
        pybind11_fail("make_default_metaclass(): failure in PyType_Ready()!");

    setattr((PyObject *) type, "__module__", str("pybind11_builtins"));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    return type;
}

/// For multiple inheritance types we need to recursively register/deregister base pointers for any
/// base classes with pointers that are difference from the instance value pointer so that we can
/// correctly recognize an offset base class pointer. This calls a function with any offset base ptrs.
inline void traverse_offset_bases(void *valueptr, const detail::type_info *tinfo, instance *self,
        bool (*f)(void * /*parentptr*/, instance * /*self*/)) {
    for (handle h : reinterpret_borrow<tuple>(tinfo->type->tp_bases)) {
        if (auto parent_tinfo = get_type_info((PyTypeObject *) h.ptr())) {
            for (auto &c : parent_tinfo->implicit_casts) {
                if (c.first == tinfo->cpptype) {
                    auto *parentptr = c.second(valueptr);
                    if (parentptr != valueptr)
                        f(parentptr, self);
                    traverse_offset_bases(parentptr, parent_tinfo, self, f);
                    break;
                }
            }
        }
    }
}

inline bool register_instance_impl(void *ptr, instance *self) {
    get_internals().registered_instances.emplace(ptr, self);
    return true; // unused, but gives the same signature as the deregister func
}
inline bool deregister_instance_impl(void *ptr, instance *self) {
    auto &registered_instances = get_internals().registered_instances;
    auto range = registered_instances.equal_range(ptr);
    for (auto it = range.first; it != range.second; ++it) {
        if (Py_TYPE(self) == Py_TYPE(it->second)) {
            registered_instances.erase(it);
            return true;
        }
    }
    return false;
}

inline void register_instance(instance *self, void *valptr, const type_info *tinfo) {
    register_instance_impl(valptr, self);
    if (!tinfo->simple_ancestors)
        traverse_offset_bases(valptr, tinfo, self, register_instance_impl);
}

inline bool deregister_instance(instance *self, void *valptr, const type_info *tinfo) {
    bool ret = deregister_instance_impl(valptr, self);
    if (!tinfo->simple_ancestors)
        traverse_offset_bases(valptr, tinfo, self, deregister_instance_impl);
    return ret;
}

/// Instance creation function for all pybind11 types. It allocates the internal instance layout for
/// holding C++ objects and holders.  Allocation is done lazily (the first time the instance is cast
/// to a reference or pointer), and initialization is done by an `__init__` function.
inline PyObject *make_new_instance(PyTypeObject *type) {
#if defined(PYPY_VERSION)
    // PyPy gets tp_basicsize wrong (issue 2482) under multiple inheritance when the first inherited
    // object is a a plain Python type (i.e. not derived from an extension type).  Fix it.
    ssize_t instance_size = static_cast<ssize_t>(sizeof(instance));
    if (type->tp_basicsize < instance_size) {
        type->tp_basicsize = instance_size;
    }
#endif
    PyObject *self = type->tp_alloc(type, 0);
    auto inst = reinterpret_cast<instance *>(self);
    // Allocate the value/holder internals:
    inst->allocate_layout();

    inst->owned = true;

    return self;
}

/// Instance creation function for all pybind11 types. It only allocates space for the
/// C++ object, but doesn't call the constructor -- an `__init__` function must do that.
extern "C" inline PyObject *pybind11_object_new(PyTypeObject *type, PyObject *, PyObject *) {
    return make_new_instance(type);
}

/// An `__init__` function constructs the C++ object. Users should provide at least one
/// of these using `py::init` or directly with `.def(__init__, ...)`. Otherwise, the
/// following default function will be used which simply throws an exception.
extern "C" inline int pybind11_object_init(PyObject *self, PyObject *, PyObject *) {
    PyTypeObject *type = Py_TYPE(self);
    std::string msg;
#if defined(PYPY_VERSION)
    msg += handle((PyObject *) type).attr("__module__").cast<std::string>() + ".";
#endif
    msg += type->tp_name;
    msg += ": No constructor defined!";
    PyErr_SetString(PyExc_TypeError, msg.c_str());
    return -1;
}

inline void add_patient(PyObject *nurse, PyObject *patient) {
    auto &internals = get_internals();
    auto instance = reinterpret_cast<detail::instance *>(nurse);
    instance->has_patients = true;
    Py_INCREF(patient);
    internals.patients[nurse].push_back(patient);
}

inline void clear_patients(PyObject *self) {
    auto instance = reinterpret_cast<detail::instance *>(self);
    auto &internals = get_internals();
    auto pos = internals.patients.find(self);
    assert(pos != internals.patients.end());
    // Clearing the patients can cause more Python code to run, which
    // can invalidate the iterator. Extract the vector of patients
    // from the unordered_map first.
    auto patients = std::move(pos->second);
    internals.patients.erase(pos);
    instance->has_patients = false;
    for (PyObject *&patient : patients)
        Py_CLEAR(patient);
}

/// Clears all internal data from the instance and removes it from registered instances in
/// preparation for deallocation.
inline void clear_instance(PyObject *self) {
    auto instance = reinterpret_cast<detail::instance *>(self);

    // Deallocate any values/holders, if present:
    for (auto &v_h : values_and_holders(instance)) {
        if (v_h) {

            // We have to deregister before we call dealloc because, for virtual MI types, we still
            // need to be able to get the parent pointers.
            if (v_h.instance_registered() && !deregister_instance(instance, v_h.value_ptr(), v_h.type))
                pybind11_fail("pybind11_object_dealloc(): Tried to deallocate unregistered instance!");

            if (instance->owned || v_h.holder_constructed())
                v_h.type->dealloc(v_h);
        }
    }
    // Deallocate the value/holder layout internals:
    instance->deallocate_layout();

    if (instance->weakrefs)
        PyObject_ClearWeakRefs(self);

    PyObject **dict_ptr = _PyObject_GetDictPtr(self);
    if (dict_ptr)
        Py_CLEAR(*dict_ptr);

    if (instance->has_patients)
        clear_patients(self);
}

/// Instance destructor function for all pybind11 types. It calls `type_info.dealloc`
/// to destroy the C++ object itself, while the rest is Python bookkeeping.
extern "C" inline void pybind11_object_dealloc(PyObject *self) {
    clear_instance(self);

    auto type = Py_TYPE(self);
    type->tp_free(self);

#if PY_VERSION_HEX < 0x03080000
    // `type->tp_dealloc != pybind11_object_dealloc` means that we're being called
    // as part of a derived type's dealloc, in which case we're not allowed to decref
    // the type here. For cross-module compatibility, we shouldn't compare directly
    // with `pybind11_object_dealloc`, but with the common one stashed in internals.
    auto pybind11_object_type = (PyTypeObject *) get_internals().instance_base;
    if (type->tp_dealloc == pybind11_object_type->tp_dealloc)
        Py_DECREF(type);
#else
    // This was not needed before Python 3.8 (Python issue 35810)
    // https://github.com/pybind/pybind11/issues/1946
    Py_DECREF(type);
#endif
}

/** Create the type which can be used as a common base for all classes.  This is
    needed in order to satisfy Python's requirements for multiple inheritance.
    Return value: New reference. */
inline PyObject *make_object_base_type(PyTypeObject *metaclass) {
    constexpr auto *name = "pybind11_object";
    auto name_obj = reinterpret_steal<object>(PYBIND11_FROM_STRING(name));

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto heap_type = (PyHeapTypeObject *) metaclass->tp_alloc(metaclass, 0);
    if (!heap_type)
        pybind11_fail("make_object_base_type(): error allocating type!");

    heap_type->ht_name = name_obj.inc_ref().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = name_obj.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = name;
    type->tp_base = type_incref(&PyBaseObject_Type);
    type->tp_basicsize = static_cast<ssize_t>(sizeof(instance));
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

    type->tp_new = pybind11_object_new;
    type->tp_init = pybind11_object_init;
    type->tp_dealloc = pybind11_object_dealloc;

    /* Support weak references (needed for the keep_alive feature) */
    type->tp_weaklistoffset = offsetof(instance, weakrefs);

    if (PyType_Ready(type) < 0)
        pybind11_fail("PyType_Ready failed in make_object_base_type():" + error_string());

    setattr((PyObject *) type, "__module__", str("pybind11_builtins"));
    PYBIND11_SET_OLDPY_QUALNAME(type, name_obj);

    assert(!PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC));
    return (PyObject *) heap_type;
}

/// dynamic_attr: Support for `d = instance.__dict__`.
extern "C" inline PyObject *pybind11_get_dict(PyObject *self, void *) {
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    if (!dict)
        dict = PyDict_New();
    Py_XINCREF(dict);
    return dict;
}

/// dynamic_attr: Support for `instance.__dict__ = dict()`.
extern "C" inline int pybind11_set_dict(PyObject *self, PyObject *new_dict, void *) {
    if (!PyDict_Check(new_dict)) {
        PyErr_Format(PyExc_TypeError, "__dict__ must be set to a dictionary, not a '%.200s'",
                     Py_TYPE(new_dict)->tp_name);
        return -1;
    }
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    Py_INCREF(new_dict);
    Py_CLEAR(dict);
    dict = new_dict;
    return 0;
}

/// dynamic_attr: Allow the garbage collector to traverse the internal instance `__dict__`.
extern "C" inline int pybind11_traverse(PyObject *self, visitproc visit, void *arg) {
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    Py_VISIT(dict);
    return 0;
}

/// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" inline int pybind11_clear(PyObject *self) {
    PyObject *&dict = *_PyObject_GetDictPtr(self);
    Py_CLEAR(dict);
    return 0;
}

/// Give instances of this type a `__dict__` and opt into garbage collection.
inline void enable_dynamic_attributes(PyHeapTypeObject *heap_type) {
    auto type = &heap_type->ht_type;
#if defined(PYPY_VERSION) && (PYPY_VERSION_NUM < 0x06000000)
    pybind11_fail(std::string(type->tp_name) + ": dynamic attributes are "
                                               "currently not supported in "
                                               "conjunction with PyPy!");
#endif
    type->tp_flags |= Py_TPFLAGS_HAVE_GC;
    type->tp_dictoffset = type->tp_basicsize; // place dict at the end
    type->tp_basicsize += (ssize_t)sizeof(PyObject *); // and allocate enough space for it
    type->tp_traverse = pybind11_traverse;
    type->tp_clear = pybind11_clear;

    static PyGetSetDef getset[] = {
        {const_cast<char*>("__dict__"), pybind11_get_dict, pybind11_set_dict, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr}
    };
    type->tp_getset = getset;
}

/// buffer_protocol: Fill in the view as specified by flags.
extern "C" inline int pybind11_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
    // Look for a `get_buffer` implementation in this type's info or any bases (following MRO).
    type_info *tinfo = nullptr;
    for (auto type : reinterpret_borrow<tuple>(Py_TYPE(obj)->tp_mro)) {
        tinfo = get_type_info((PyTypeObject *) type.ptr());
        if (tinfo && tinfo->get_buffer)
            break;
    }
    if (view == nullptr || !tinfo || !tinfo->get_buffer) {
        if (view)
            view->obj = nullptr;
        PyErr_SetString(PyExc_BufferError, "pybind11_getbuffer(): Internal error");
        return -1;
    }
    std::memset(view, 0, sizeof(Py_buffer));
    buffer_info *info = tinfo->get_buffer(obj, tinfo->get_buffer_data);
    view->obj = obj;
    view->ndim = 1;
    view->internal = info;
    view->buf = info->ptr;
    view->itemsize = info->itemsize;
    view->len = view->itemsize;
    for (auto s : info->shape)
        view->len *= s;
    view->readonly = info->readonly;
    if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE && info->readonly) {
        if (view)
            view->obj = nullptr;
        PyErr_SetString(PyExc_BufferError, "Writable buffer requested for readonly storage");
        return -1;
    }
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
        view->format = const_cast<char *>(info->format.c_str());
    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->ndim = (int) info->ndim;
        view->strides = &info->strides[0];
        view->shape = &info->shape[0];
    }
    Py_INCREF(view->obj);
    return 0;
}

/// buffer_protocol: Release the resources of the buffer.
extern "C" inline void pybind11_releasebuffer(PyObject *, Py_buffer *view) {
    delete (buffer_info *) view->internal;
}

/// Give this type a buffer interface.
inline void enable_buffer_protocol(PyHeapTypeObject *heap_type) {
    heap_type->ht_type.tp_as_buffer = &heap_type->as_buffer;
#if PY_MAJOR_VERSION < 3
    heap_type->ht_type.tp_flags |= Py_TPFLAGS_HAVE_NEWBUFFER;
#endif

    heap_type->as_buffer.bf_getbuffer = pybind11_getbuffer;
    heap_type->as_buffer.bf_releasebuffer = pybind11_releasebuffer;
}

/** Create a brand new Python type according to the `type_record` specification.
    Return value: New reference. */
inline PyObject* make_new_python_type(const type_record &rec) {
    auto name = reinterpret_steal<object>(PYBIND11_FROM_STRING(rec.name));

    auto qualname = name;
    if (rec.scope && !PyModule_Check(rec.scope.ptr()) && hasattr(rec.scope, "__qualname__")) {
#if PY_MAJOR_VERSION >= 3
        qualname = reinterpret_steal<object>(
            PyUnicode_FromFormat("%U.%U", rec.scope.attr("__qualname__").ptr(), name.ptr()));
#else
        qualname = str(rec.scope.attr("__qualname__").cast<std::string>() + "." + rec.name);
#endif
    }

    object module;
    if (rec.scope) {
        if (hasattr(rec.scope, "__module__"))
            module = rec.scope.attr("__module__");
        else if (hasattr(rec.scope, "__name__"))
            module = rec.scope.attr("__name__");
    }

    auto full_name = c_str(
#if !defined(PYPY_VERSION)
        module ? str(module).cast<std::string>() + "." + rec.name :
#endif
        rec.name);

    char *tp_doc = nullptr;
    if (rec.doc && options::show_user_defined_docstrings()) {
        /* Allocate memory for docstring (using PyObject_MALLOC, since
           Python will free this later on) */
        size_t size = strlen(rec.doc) + 1;
        tp_doc = (char *) PyObject_MALLOC(size);
        memcpy((void *) tp_doc, rec.doc, size);
    }

    auto &internals = get_internals();
    auto bases = tuple(rec.bases);
    auto base = (bases.size() == 0) ? internals.instance_base
                                    : bases[0].ptr();

    /* Danger zone: from now (and until PyType_Ready), make sure to
       issue no Python C API calls which could potentially invoke the
       garbage collector (the GC will call type_traverse(), which will in
       turn find the newly constructed type in an invalid state) */
    auto metaclass = rec.metaclass.ptr() ? (PyTypeObject *) rec.metaclass.ptr()
                                         : internals.default_metaclass;

    auto heap_type = (PyHeapTypeObject *) metaclass->tp_alloc(metaclass, 0);
    if (!heap_type)
        pybind11_fail(std::string(rec.name) + ": Unable to create type object!");

    heap_type->ht_name = name.release().ptr();
#ifdef PYBIND11_BUILTIN_QUALNAME
    heap_type->ht_qualname = qualname.inc_ref().ptr();
#endif

    auto type = &heap_type->ht_type;
    type->tp_name = full_name;
    type->tp_doc = tp_doc;
    type->tp_base = type_incref((PyTypeObject *)base);
    type->tp_basicsize = static_cast<ssize_t>(sizeof(instance));
    if (bases.size() > 0)
        type->tp_bases = bases.release().ptr();

    /* Don't inherit base __init__ */
    type->tp_init = pybind11_object_init;

    /* Supported protocols */
    type->tp_as_number = &heap_type->as_number;
    type->tp_as_sequence = &heap_type->as_sequence;
    type->tp_as_mapping = &heap_type->as_mapping;
#if PY_VERSION_HEX >= 0x03050000
    type->tp_as_async = &heap_type->as_async;
#endif

    /* Flags */
    type->tp_flags |= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
#if PY_MAJOR_VERSION < 3
    type->tp_flags |= Py_TPFLAGS_CHECKTYPES;
#endif
    if (!rec.is_final)
        type->tp_flags |= Py_TPFLAGS_BASETYPE;

    if (rec.dynamic_attr)
        enable_dynamic_attributes(heap_type);

    if (rec.buffer_protocol)
        enable_buffer_protocol(heap_type);

    if (PyType_Ready(type) < 0)
        pybind11_fail(std::string(rec.name) + ": PyType_Ready failed (" + error_string() + ")!");

    assert(rec.dynamic_attr ? PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC)
                            : !PyType_HasFeature(type, Py_TPFLAGS_HAVE_GC));

    /* Register type with the parent scope */
    if (rec.scope)
        setattr(rec.scope, rec.name, (PyObject *) type);
    else
        Py_INCREF(type); // Keep it alive forever (reference leak)

    if (module) // Needed by pydoc
        setattr((PyObject *) type, "__module__", module);

    PYBIND11_SET_OLDPY_QUALNAME(type, qualname);

    return (PyObject *) type;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "detail/class.h" ============


//========= begin of #include "detail/init.h" ============

/*
    pybind11/detail/init.h: init factory function implementation and support code.

    Copyright (c) 2017 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "class.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <>
class type_caster<value_and_holder> {
public:
    bool load(handle h, bool) {
        value = reinterpret_cast<value_and_holder *>(h.ptr());
        return true;
    }

    template <typename> using cast_op_type = value_and_holder &;
    operator value_and_holder &() { return *value; }
    static constexpr auto name = _<value_and_holder>();

private:
    value_and_holder *value = nullptr;
};

PYBIND11_NAMESPACE_BEGIN(initimpl)

inline void no_nullptr(void *ptr) {
    if (!ptr) throw type_error("pybind11::init(): factory function returned nullptr");
}

// Implementing functions for all forms of py::init<...> and py::init(...)
template <typename Class> using Cpp = typename Class::type;
template <typename Class> using Alias = typename Class::type_alias;
template <typename Class> using Holder = typename Class::holder_type;

template <typename Class> using is_alias_constructible = std::is_constructible<Alias<Class>, Cpp<Class> &&>;

// Takes a Cpp pointer and returns true if it actually is a polymorphic Alias instance.
template <typename Class, enable_if_t<Class::has_alias, int> = 0>
bool is_alias(Cpp<Class> *ptr) {
    return dynamic_cast<Alias<Class> *>(ptr) != nullptr;
}
// Failing fallback version of the above for a no-alias class (always returns false)
template <typename /*Class*/>
constexpr bool is_alias(void *) { return false; }

// Constructs and returns a new object; if the given arguments don't map to a constructor, we fall
// back to brace aggregate initiailization so that for aggregate initialization can be used with
// py::init, e.g.  `py::init<int, int>` to initialize a `struct T { int a; int b; }`.  For
// non-aggregate types, we need to use an ordinary T(...) constructor (invoking as `T{...}` usually
// works, but will not do the expected thing when `T` has an `initializer_list<T>` constructor).
template <typename Class, typename... Args, detail::enable_if_t<std::is_constructible<Class, Args...>::value, int> = 0>
inline Class *construct_or_initialize(Args &&...args) { return new Class(std::forward<Args>(args)...); }
template <typename Class, typename... Args, detail::enable_if_t<!std::is_constructible<Class, Args...>::value, int> = 0>
inline Class *construct_or_initialize(Args &&...args) { return new Class{std::forward<Args>(args)...}; }

// Attempts to constructs an alias using a `Alias(Cpp &&)` constructor.  This allows types with
// an alias to provide only a single Cpp factory function as long as the Alias can be
// constructed from an rvalue reference of the base Cpp type.  This means that Alias classes
// can, when appropriate, simply define a `Alias(Cpp &&)` constructor rather than needing to
// inherit all the base class constructors.
template <typename Class>
void construct_alias_from_cpp(std::true_type /*is_alias_constructible*/,
                              value_and_holder &v_h, Cpp<Class> &&base) {
    v_h.value_ptr() = new Alias<Class>(std::move(base));
}
template <typename Class>
[[noreturn]] void construct_alias_from_cpp(std::false_type /*!is_alias_constructible*/,
                                           value_and_holder &, Cpp<Class> &&) {
    throw type_error("pybind11::init(): unable to convert returned instance to required "
                     "alias class: no `Alias<Class>(Class &&)` constructor available");
}

// Error-generating fallback for factories that don't match one of the below construction
// mechanisms.
template <typename Class>
void construct(...) {
    static_assert(!std::is_same<Class, Class>::value /* always false */,
            "pybind11::init(): init function must return a compatible pointer, "
            "holder, or value");
}

// Pointer return v1: the factory function returns a class pointer for a registered class.
// If we don't need an alias (because this class doesn't have one, or because the final type is
// inherited on the Python side) we can simply take over ownership.  Otherwise we need to try to
// construct an Alias from the returned base instance.
template <typename Class>
void construct(value_and_holder &v_h, Cpp<Class> *ptr, bool need_alias) {
    no_nullptr(ptr);
    if (Class::has_alias && need_alias && !is_alias<Class>(ptr)) {
        // We're going to try to construct an alias by moving the cpp type.  Whether or not
        // that succeeds, we still need to destroy the original cpp pointer (either the
        // moved away leftover, if the alias construction works, or the value itself if we
        // throw an error), but we can't just call `delete ptr`: it might have a special
        // deleter, or might be shared_from_this.  So we construct a holder around it as if
        // it was a normal instance, then steal the holder away into a local variable; thus
        // the holder and destruction happens when we leave the C++ scope, and the holder
        // class gets to handle the destruction however it likes.
        v_h.value_ptr() = ptr;
        v_h.set_instance_registered(true); // To prevent init_instance from registering it
        v_h.type->init_instance(v_h.inst, nullptr); // Set up the holder
        Holder<Class> temp_holder(std::move(v_h.holder<Holder<Class>>())); // Steal the holder
        v_h.type->dealloc(v_h); // Destroys the moved-out holder remains, resets value ptr to null
        v_h.set_instance_registered(false);

        construct_alias_from_cpp<Class>(is_alias_constructible<Class>{}, v_h, std::move(*ptr));
    } else {
        // Otherwise the type isn't inherited, so we don't need an Alias
        v_h.value_ptr() = ptr;
    }
}

// Pointer return v2: a factory that always returns an alias instance ptr.  We simply take over
// ownership of the pointer.
template <typename Class, enable_if_t<Class::has_alias, int> = 0>
void construct(value_and_holder &v_h, Alias<Class> *alias_ptr, bool) {
    no_nullptr(alias_ptr);
    v_h.value_ptr() = static_cast<Cpp<Class> *>(alias_ptr);
}

// Holder return: copy its pointer, and move or copy the returned holder into the new instance's
// holder.  This also handles types like std::shared_ptr<T> and std::unique_ptr<T> where T is a
// derived type (through those holder's implicit conversion from derived class holder constructors).
template <typename Class>
void construct(value_and_holder &v_h, Holder<Class> holder, bool need_alias) {
    auto *ptr = holder_helper<Holder<Class>>::get(holder);
    no_nullptr(ptr);
    // If we need an alias, check that the held pointer is actually an alias instance
    if (Class::has_alias && need_alias && !is_alias<Class>(ptr))
        throw type_error("pybind11::init(): construction failed: returned holder-wrapped instance "
                         "is not an alias instance");

    v_h.value_ptr() = ptr;
    v_h.type->init_instance(v_h.inst, &holder);
}

// return-by-value version 1: returning a cpp class by value.  If the class has an alias and an
// alias is required the alias must have an `Alias(Cpp &&)` constructor so that we can construct
// the alias from the base when needed (i.e. because of Python-side inheritance).  When we don't
// need it, we simply move-construct the cpp value into a new instance.
template <typename Class>
void construct(value_and_holder &v_h, Cpp<Class> &&result, bool need_alias) {
    static_assert(std::is_move_constructible<Cpp<Class>>::value,
        "pybind11::init() return-by-value factory function requires a movable class");
    if (Class::has_alias && need_alias)
        construct_alias_from_cpp<Class>(is_alias_constructible<Class>{}, v_h, std::move(result));
    else
        v_h.value_ptr() = new Cpp<Class>(std::move(result));
}

// return-by-value version 2: returning a value of the alias type itself.  We move-construct an
// Alias instance (even if no the python-side inheritance is involved).  The is intended for
// cases where Alias initialization is always desired.
template <typename Class>
void construct(value_and_holder &v_h, Alias<Class> &&result, bool) {
    static_assert(std::is_move_constructible<Alias<Class>>::value,
        "pybind11::init() return-by-alias-value factory function requires a movable alias class");
    v_h.value_ptr() = new Alias<Class>(std::move(result));
}

// Implementing class for py::init<...>()
template <typename... Args>
struct constructor {
    template <typename Class, typename... Extra, enable_if_t<!Class::has_alias, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            v_h.value_ptr() = construct_or_initialize<Cpp<Class>>(std::forward<Args>(args)...);
        }, is_new_style_constructor(), extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          std::is_constructible<Cpp<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            if (Py_TYPE(v_h.inst) == v_h.type->type)
                v_h.value_ptr() = construct_or_initialize<Cpp<Class>>(std::forward<Args>(args)...);
            else
                v_h.value_ptr() = construct_or_initialize<Alias<Class>>(std::forward<Args>(args)...);
        }, is_new_style_constructor(), extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          !std::is_constructible<Cpp<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            v_h.value_ptr() = construct_or_initialize<Alias<Class>>(std::forward<Args>(args)...);
        }, is_new_style_constructor(), extra...);
    }
};

// Implementing class for py::init_alias<...>()
template <typename... Args> struct alias_constructor {
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && std::is_constructible<Alias<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            v_h.value_ptr() = construct_or_initialize<Alias<Class>>(std::forward<Args>(args)...);
        }, is_new_style_constructor(), extra...);
    }
};

// Implementation class for py::init(Func) and py::init(Func, AliasFunc)
template <typename CFunc, typename AFunc = void_type (*)(),
          typename = function_signature_t<CFunc>, typename = function_signature_t<AFunc>>
struct factory;

// Specialization for py::init(Func)
template <typename Func, typename Return, typename... Args>
struct factory<Func, void_type (*)(), Return(Args...)> {
    remove_reference_t<Func> class_factory;

    factory(Func &&f) : class_factory(std::forward<Func>(f)) { }

    // The given class either has no alias or has no separate alias factory;
    // this always constructs the class itself.  If the class is registered with an alias
    // type and an alias instance is needed (i.e. because the final type is a Python class
    // inheriting from the C++ type) the returned value needs to either already be an alias
    // instance, or the alias needs to be constructible from a `Class &&` argument.
    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) && {
        #if defined(PYBIND11_CPP14)
        cl.def("__init__", [func = std::move(class_factory)]
        #else
        auto &func = class_factory;
        cl.def("__init__", [func]
        #endif
        (value_and_holder &v_h, Args... args) {
            construct<Class>(v_h, func(std::forward<Args>(args)...),
                             Py_TYPE(v_h.inst) != v_h.type->type);
        }, is_new_style_constructor(), extra...);
    }
};

// Specialization for py::init(Func, AliasFunc)
template <typename CFunc, typename AFunc,
          typename CReturn, typename... CArgs, typename AReturn, typename... AArgs>
struct factory<CFunc, AFunc, CReturn(CArgs...), AReturn(AArgs...)> {
    static_assert(sizeof...(CArgs) == sizeof...(AArgs),
                  "pybind11::init(class_factory, alias_factory): class and alias factories "
                  "must have identical argument signatures");
    static_assert(all_of<std::is_same<CArgs, AArgs>...>::value,
                  "pybind11::init(class_factory, alias_factory): class and alias factories "
                  "must have identical argument signatures");

    remove_reference_t<CFunc> class_factory;
    remove_reference_t<AFunc> alias_factory;

    factory(CFunc &&c, AFunc &&a)
        : class_factory(std::forward<CFunc>(c)), alias_factory(std::forward<AFunc>(a)) { }

    // The class factory is called when the `self` type passed to `__init__` is the direct
    // class (i.e. not inherited), the alias factory when `self` is a Python-side subtype.
    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra&... extra) && {
        static_assert(Class::has_alias, "The two-argument version of `py::init()` can "
                                        "only be used if the class has an alias");
        #if defined(PYBIND11_CPP14)
        cl.def("__init__", [class_func = std::move(class_factory), alias_func = std::move(alias_factory)]
        #else
        auto &class_func = class_factory;
        auto &alias_func = alias_factory;
        cl.def("__init__", [class_func, alias_func]
        #endif
        (value_and_holder &v_h, CArgs... args) {
            if (Py_TYPE(v_h.inst) == v_h.type->type)
                // If the instance type equals the registered type we don't have inheritance, so
                // don't need the alias and can construct using the class function:
                construct<Class>(v_h, class_func(std::forward<CArgs>(args)...), false);
            else
                construct<Class>(v_h, alias_func(std::forward<CArgs>(args)...), true);
        }, is_new_style_constructor(), extra...);
    }
};

/// Set just the C++ state. Same as `__init__`.
template <typename Class, typename T>
void setstate(value_and_holder &v_h, T &&result, bool need_alias) {
    construct<Class>(v_h, std::forward<T>(result), need_alias);
}

/// Set both the C++ and Python states
template <typename Class, typename T, typename O,
          enable_if_t<std::is_convertible<O, handle>::value, int> = 0>
void setstate(value_and_holder &v_h, std::pair<T, O> &&result, bool need_alias) {
    construct<Class>(v_h, std::move(result.first), need_alias);
    setattr((PyObject *) v_h.inst, "__dict__", result.second);
}

/// Implementation for py::pickle(GetState, SetState)
template <typename Get, typename Set,
          typename = function_signature_t<Get>, typename = function_signature_t<Set>>
struct pickle_factory;

template <typename Get, typename Set,
          typename RetState, typename Self, typename NewInstance, typename ArgState>
struct pickle_factory<Get, Set, RetState(Self), NewInstance(ArgState)> {
    static_assert(std::is_same<intrinsic_t<RetState>, intrinsic_t<ArgState>>::value,
                  "The type returned by `__getstate__` must be the same "
                  "as the argument accepted by `__setstate__`");

    remove_reference_t<Get> get;
    remove_reference_t<Set> set;

    pickle_factory(Get get, Set set)
        : get(std::forward<Get>(get)), set(std::forward<Set>(set)) { }

    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) && {
        cl.def("__getstate__", std::move(get));

#if defined(PYBIND11_CPP14)
        cl.def("__setstate__", [func = std::move(set)]
#else
        auto &func = set;
        cl.def("__setstate__", [func]
#endif
        (value_and_holder &v_h, ArgState state) {
            setstate<Class>(v_h, func(std::forward<ArgState>(state)),
                            Py_TYPE(v_h.inst) != v_h.type->type);
        }, is_new_style_constructor(), extra...);
    }
};

PYBIND11_NAMESPACE_END(initimpl)
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(pybind11)


//========= end of #include "detail/init.h" ============


//========= begin of #include "pybind11.h" ============

/*
    pybind11/pybind11.h: Main header file of the C++11 python
    binding generator library

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/


// ans ignore: #include "attr.h"
// ans ignore: #include "options.h"
// ans ignore: #include "detail/class.h"
// ans ignore: #include "detail/init.h"

#if defined(__GNUG__) && !defined(__clang__)
#  include <cxxabi.h>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

/// Wraps an arbitrary C++ function/method/lambda function/.. into a callable Python object
class cpp_function : public function {
public:
    cpp_function() { }
    cpp_function(std::nullptr_t) { }

    /// Construct a cpp_function from a vanilla function pointer
    template <typename Return, typename... Args, typename... Extra>
    cpp_function(Return (*f)(Args...), const Extra&... extra) {
        initialize(f, f, extra...);
    }

    /// Construct a cpp_function from a lambda function (possibly with internal state)
    template <typename Func, typename... Extra,
              typename = detail::enable_if_t<detail::is_lambda<Func>::value>>
    cpp_function(Func &&f, const Extra&... extra) {
        initialize(std::forward<Func>(f),
                   (detail::function_signature_t<Func> *) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...), const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*) (Class *, Arg...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (non-const, lvalue ref-qualifier)
    /// A copy of the overload for non-const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...)&, const Extra&... extra) {
        initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*) (Class *, Arg...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const, no ref-qualifier)
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...) const, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, extra...);
    }

    /// Construct a cpp_function from a class method (const, lvalue ref-qualifier)
    /// A copy of the overload for const functions without explicit ref-qualifier
    /// but with an added `&`.
    template <typename Return, typename Class, typename... Arg, typename... Extra>
    cpp_function(Return (Class::*f)(Arg...) const&, const Extra&... extra) {
        initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                   (Return (*)(const Class *, Arg ...)) nullptr, extra...);
    }

    /// Return the function name
    object name() const { return attr("__name__"); }

protected:
    /// Space optimization: don't inline this frequently instantiated fragment
    PYBIND11_NOINLINE detail::function_record *make_function_record() {
        return new detail::function_record();
    }

    /// Special internal constructor for functors, lambda functions, etc.
    template <typename Func, typename Return, typename... Args, typename... Extra>
    void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {
        using namespace detail;
        struct capture { remove_reference_t<Func> f; };

        /* Store the function including any extra state it might have (e.g. a lambda capture object) */
        auto rec = make_function_record();

        /* Store the capture object directly in the function record if there is enough space */
        if (sizeof(capture) <= sizeof(rec->data)) {
            /* Without these pragmas, GCC warns that there might not be
               enough space to use the placement new operator. However, the
               'if' statement above ensures that this is the case. */
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wplacement-new"
#endif
            new ((capture *) &rec->data) capture { std::forward<Func>(f) };
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 6
#  pragma GCC diagnostic pop
#endif
            if (!std::is_trivially_destructible<Func>::value)
                rec->free_data = [](function_record *r) { ((capture *) &r->data)->~capture(); };
        } else {
            rec->data[0] = new capture { std::forward<Func>(f) };
            rec->free_data = [](function_record *r) { delete ((capture *) r->data[0]); };
        }

        /* Type casters for the function arguments and return value */
        using cast_in = argument_loader<Args...>;
        using cast_out = make_caster<
            conditional_t<std::is_void<Return>::value, void_type, Return>
        >;

        static_assert(expected_num_args<Extra...>(sizeof...(Args), cast_in::has_args, cast_in::has_kwargs),
                      "The number of argument annotations does not match the number of function arguments");

        /* Dispatch code which converts function arguments and performs the actual function call */
        rec->impl = [](function_call &call) -> handle {
            cast_in args_converter;

            /* Try to cast the function arguments into the C++ domain */
            if (!args_converter.load_args(call))
                return PYBIND11_TRY_NEXT_OVERLOAD;

            /* Invoke call policy pre-call hook */
            process_attributes<Extra...>::precall(call);

            /* Get a pointer to the capture object */
            auto data = (sizeof(capture) <= sizeof(call.func.data)
                         ? &call.func.data : call.func.data[0]);
            capture *cap = const_cast<capture *>(reinterpret_cast<const capture *>(data));

            /* Override policy for rvalues -- usually to enforce rvp::move on an rvalue */
            return_value_policy policy = return_value_policy_override<Return>::policy(call.func.policy);

            /* Function scope guard -- defaults to the compile-to-nothing `void_type` */
            using Guard = extract_guard_t<Extra...>;

            /* Perform the function call */
            handle result = cast_out::cast(
                std::move(args_converter).template call<Return, Guard>(cap->f), policy, call.parent);

            /* Invoke call policy post-call hook */
            process_attributes<Extra...>::postcall(call, result);

            return result;
        };

        /* Process any user-provided function attributes */
        process_attributes<Extra...>::init(extra..., rec);

        {
            constexpr bool has_kwonly_args = any_of<std::is_same<kwonly, Extra>...>::value,
                           has_args = any_of<std::is_same<args, Args>...>::value,
                           has_arg_annotations = any_of<is_keyword<Extra>...>::value;
            static_assert(has_arg_annotations || !has_kwonly_args, "py::kwonly requires the use of argument annotations");
            static_assert(!(has_args && has_kwonly_args), "py::kwonly cannot be combined with a py::args argument");
        }

        /* Generate a readable signature describing the function's arguments and return value types */
        static constexpr auto signature = _("(") + cast_in::arg_names + _(") -> ") + cast_out::name;
        PYBIND11_DESCR_CONSTEXPR auto types = decltype(signature)::types();

        /* Register the function with Python from generic (non-templated) code */
        initialize_generic(rec, signature.text, types.data(), sizeof...(Args));

        if (cast_in::has_args) rec->has_args = true;
        if (cast_in::has_kwargs) rec->has_kwargs = true;

        /* Stash some additional information used by an important optimization in 'functional.h' */
        using FunctionType = Return (*)(Args...);
        constexpr bool is_function_ptr =
            std::is_convertible<Func, FunctionType>::value &&
            sizeof(capture) == sizeof(void *);
        if (is_function_ptr) {
            rec->is_stateless = true;
            rec->data[1] = const_cast<void *>(reinterpret_cast<const void *>(&typeid(FunctionType)));
        }
    }

    /// Register a function call with Python (generic non-templated code goes here)
    void initialize_generic(detail::function_record *rec, const char *text,
                            const std::type_info *const *types, size_t args) {

        /* Create copies of all referenced C-style strings */
        rec->name = strdup(rec->name ? rec->name : "");
        if (rec->doc) rec->doc = strdup(rec->doc);
        for (auto &a: rec->args) {
            if (a.name)
                a.name = strdup(a.name);
            if (a.descr)
                a.descr = strdup(a.descr);
            else if (a.value)
                a.descr = strdup(repr(a.value).cast<std::string>().c_str());
        }

        rec->is_constructor = !strcmp(rec->name, "__init__") || !strcmp(rec->name, "__setstate__");

#if !defined(NDEBUG) && !defined(PYBIND11_DISABLE_NEW_STYLE_INIT_WARNING)
        if (rec->is_constructor && !rec->is_new_style_constructor) {
            const auto class_name = std::string(((PyTypeObject *) rec->scope.ptr())->tp_name);
            const auto func_name = std::string(rec->name);
            PyErr_WarnEx(
                PyExc_FutureWarning,
                ("pybind11-bound class '" + class_name + "' is using an old-style "
                 "placement-new '" + func_name + "' which has been deprecated. See "
                 "the upgrade guide in pybind11's docs. This message is only visible "
                 "when compiled in debug mode.").c_str(), 0
            );
        }
#endif

        /* Generate a proper function signature */
        std::string signature;
        size_t type_index = 0, arg_index = 0;
        for (auto *pc = text; *pc != '\0'; ++pc) {
            const auto c = *pc;

            if (c == '{') {
                // Write arg name for everything except *args and **kwargs.
                if (*(pc + 1) == '*')
                    continue;

                if (arg_index < rec->args.size() && rec->args[arg_index].name) {
                    signature += rec->args[arg_index].name;
                } else if (arg_index == 0 && rec->is_method) {
                    signature += "self";
                } else {
                    signature += "arg" + std::to_string(arg_index - (rec->is_method ? 1 : 0));
                }
                signature += ": ";
            } else if (c == '}') {
                // Write default value if available.
                if (arg_index < rec->args.size() && rec->args[arg_index].descr) {
                    signature += " = ";
                    signature += rec->args[arg_index].descr;
                }
                arg_index++;
            } else if (c == '%') {
                const std::type_info *t = types[type_index++];
                if (!t)
                    pybind11_fail("Internal error while parsing type signature (1)");
                if (auto tinfo = detail::get_type_info(*t)) {
                    handle th((PyObject *) tinfo->type);
                    signature +=
                        th.attr("__module__").cast<std::string>() + "." +
                        th.attr("__qualname__").cast<std::string>(); // Python 3.3+, but we backport it to earlier versions
                } else if (rec->is_new_style_constructor && arg_index == 0) {
                    // A new-style `__init__` takes `self` as `value_and_holder`.
                    // Rewrite it to the proper class type.
                    signature +=
                        rec->scope.attr("__module__").cast<std::string>() + "." +
                        rec->scope.attr("__qualname__").cast<std::string>();
                } else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    signature += tname;
                }
            } else {
                signature += c;
            }
        }
        if (arg_index != args || types[type_index] != nullptr)
            pybind11_fail("Internal error while parsing type signature (2)");

#if PY_MAJOR_VERSION < 3
        if (strcmp(rec->name, "__next__") == 0) {
            std::free(rec->name);
            rec->name = strdup("next");
        } else if (strcmp(rec->name, "__bool__") == 0) {
            std::free(rec->name);
            rec->name = strdup("__nonzero__");
        }
#endif
        rec->signature = strdup(signature.c_str());
        rec->args.shrink_to_fit();
        rec->nargs = (std::uint16_t) args;

        if (rec->sibling && PYBIND11_INSTANCE_METHOD_CHECK(rec->sibling.ptr()))
            rec->sibling = PYBIND11_INSTANCE_METHOD_GET_FUNCTION(rec->sibling.ptr());

        detail::function_record *chain = nullptr, *chain_start = rec;
        if (rec->sibling) {
            if (PyCFunction_Check(rec->sibling.ptr())) {
                auto rec_capsule = reinterpret_borrow<capsule>(PyCFunction_GET_SELF(rec->sibling.ptr()));
                chain = (detail::function_record *) rec_capsule;
                /* Never append a method to an overload chain of a parent class;
                   instead, hide the parent's overloads in this case */
                if (!chain->scope.is(rec->scope))
                    chain = nullptr;
            }
            // Don't trigger for things like the default __init__, which are wrapper_descriptors that we are intentionally replacing
            else if (!rec->sibling.is_none() && rec->name[0] != '_')
                pybind11_fail("Cannot overload existing non-function object \"" + std::string(rec->name) +
                        "\" with a function of the same name");
        }

        if (!chain) {
            /* No existing overload was found, create a new function object */
            rec->def = new PyMethodDef();
            std::memset(rec->def, 0, sizeof(PyMethodDef));
            rec->def->ml_name = rec->name;
            rec->def->ml_meth = reinterpret_cast<PyCFunction>(reinterpret_cast<void (*) (void)>(*dispatcher));
            rec->def->ml_flags = METH_VARARGS | METH_KEYWORDS;

            capsule rec_capsule(rec, [](void *ptr) {
                destruct((detail::function_record *) ptr);
            });

            object scope_module;
            if (rec->scope) {
                if (hasattr(rec->scope, "__module__")) {
                    scope_module = rec->scope.attr("__module__");
                } else if (hasattr(rec->scope, "__name__")) {
                    scope_module = rec->scope.attr("__name__");
                }
            }

            m_ptr = PyCFunction_NewEx(rec->def, rec_capsule.ptr(), scope_module.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate function object");
        } else {
            /* Append at the end of the overload chain */
            m_ptr = rec->sibling.ptr();
            inc_ref();
            chain_start = chain;
            if (chain->is_method != rec->is_method)
                pybind11_fail("overloading a method with both static and instance methods is not supported; "
                    #if defined(NDEBUG)
                        "compile in debug mode for more details"
                    #else
                        "error while attempting to bind " + std::string(rec->is_method ? "instance" : "static") + " method " +
                        std::string(pybind11::str(rec->scope.attr("__name__"))) + "." + std::string(rec->name) + signature
                    #endif
                );
            while (chain->next)
                chain = chain->next;
            chain->next = rec;
        }

        std::string signatures;
        int index = 0;
        /* Create a nice pydoc rec including all signatures and
           docstrings of the functions in the overload chain */
        if (chain && options::show_function_signatures()) {
            // First a generic signature
            signatures += rec->name;
            signatures += "(*args, **kwargs)\n";
            signatures += "Overloaded function.\n\n";
        }
        // Then specific overload signatures
        bool first_user_def = true;
        for (auto it = chain_start; it != nullptr; it = it->next) {
            if (options::show_function_signatures()) {
                if (index > 0) signatures += "\n";
                if (chain)
                    signatures += std::to_string(++index) + ". ";
                signatures += rec->name;
                signatures += it->signature;
                signatures += "\n";
            }
            if (it->doc && strlen(it->doc) > 0 && options::show_user_defined_docstrings()) {
                // If we're appending another docstring, and aren't printing function signatures, we
                // need to append a newline first:
                if (!options::show_function_signatures()) {
                    if (first_user_def) first_user_def = false;
                    else signatures += "\n";
                }
                if (options::show_function_signatures()) signatures += "\n";
                signatures += it->doc;
                if (options::show_function_signatures()) signatures += "\n";
            }
        }

        /* Install docstring */
        PyCFunctionObject *func = (PyCFunctionObject *) m_ptr;
        if (func->m_ml->ml_doc)
            std::free(const_cast<char *>(func->m_ml->ml_doc));
        func->m_ml->ml_doc = strdup(signatures.c_str());

        if (rec->is_method) {
            m_ptr = PYBIND11_INSTANCE_METHOD_NEW(m_ptr, rec->scope.ptr());
            if (!m_ptr)
                pybind11_fail("cpp_function::cpp_function(): Could not allocate instance method object");
            Py_DECREF(func);
        }
    }

    /// When a cpp_function is GCed, release any memory allocated by pybind11
    static void destruct(detail::function_record *rec) {
        while (rec) {
            detail::function_record *next = rec->next;
            if (rec->free_data)
                rec->free_data(rec);
            std::free((char *) rec->name);
            std::free((char *) rec->doc);
            std::free((char *) rec->signature);
            for (auto &arg: rec->args) {
                std::free(const_cast<char *>(arg.name));
                std::free(const_cast<char *>(arg.descr));
                arg.value.dec_ref();
            }
            if (rec->def) {
                std::free(const_cast<char *>(rec->def->ml_doc));
                delete rec->def;
            }
            delete rec;
            rec = next;
        }
    }

    /// Main dispatch logic for calls to functions bound using pybind11
    static PyObject *dispatcher(PyObject *self, PyObject *args_in, PyObject *kwargs_in) {
        using namespace detail;

        /* Iterator over the list of potentially admissible overloads */
        const function_record *overloads = (function_record *) PyCapsule_GetPointer(self, nullptr),
                              *it = overloads;

        /* Need to know how many arguments + keyword arguments there are to pick the right overload */
        const size_t n_args_in = (size_t) PyTuple_GET_SIZE(args_in);

        handle parent = n_args_in > 0 ? PyTuple_GET_ITEM(args_in, 0) : nullptr,
               result = PYBIND11_TRY_NEXT_OVERLOAD;

        auto self_value_and_holder = value_and_holder();
        if (overloads->is_constructor) {
            const auto tinfo = get_type_info((PyTypeObject *) overloads->scope.ptr());
            const auto pi = reinterpret_cast<instance *>(parent.ptr());
            self_value_and_holder = pi->get_value_and_holder(tinfo, false);

            if (!self_value_and_holder.type || !self_value_and_holder.inst) {
                PyErr_SetString(PyExc_TypeError, "__init__(self, ...) called with invalid `self` argument");
                return nullptr;
            }

            // If this value is already registered it must mean __init__ is invoked multiple times;
            // we really can't support that in C++, so just ignore the second __init__.
            if (self_value_and_holder.instance_registered())
                return none().release().ptr();
        }

        try {
            // We do this in two passes: in the first pass, we load arguments with `convert=false`;
            // in the second, we allow conversion (except for arguments with an explicit
            // py::arg().noconvert()).  This lets us prefer calls without conversion, with
            // conversion as a fallback.
            std::vector<function_call> second_pass;

            // However, if there are no overloads, we can just skip the no-convert pass entirely
            const bool overloaded = it != nullptr && it->next != nullptr;

            for (; it != nullptr; it = it->next) {

                /* For each overload:
                   1. Copy all positional arguments we were given, also checking to make sure that
                      named positional arguments weren't *also* specified via kwarg.
                   2. If we weren't given enough, try to make up the omitted ones by checking
                      whether they were provided by a kwarg matching the `py::arg("name")` name.  If
                      so, use it (and remove it from kwargs; if not, see if the function binding
                      provided a default that we can use.
                   3. Ensure that either all keyword arguments were "consumed", or that the function
                      takes a kwargs argument to accept unconsumed kwargs.
                   4. Any positional arguments still left get put into a tuple (for args), and any
                      leftover kwargs get put into a dict.
                   5. Pack everything into a vector; if we have py::args or py::kwargs, they are an
                      extra tuple or dict at the end of the positional arguments.
                   6. Call the function call dispatcher (function_record::impl)

                   If one of these fail, move on to the next overload and keep trying until we get a
                   result other than PYBIND11_TRY_NEXT_OVERLOAD.
                 */

                const function_record &func = *it;
                size_t num_args = func.nargs;    // Number of positional arguments that we need
                if (func.has_args) --num_args;   // (but don't count py::args
                if (func.has_kwargs) --num_args; //  or py::kwargs)
                size_t pos_args = num_args - func.nargs_kwonly;

                if (!func.has_args && n_args_in > pos_args)
                    continue; // Too many positional arguments for this overload

                if (n_args_in < pos_args && func.args.size() < pos_args)
                    continue; // Not enough positional arguments given, and not enough defaults to fill in the blanks

                function_call call(func, parent);

                size_t args_to_copy = (std::min)(pos_args, n_args_in); // Protect std::min with parentheses
                size_t args_copied = 0;

                // 0. Inject new-style `self` argument
                if (func.is_new_style_constructor) {
                    // The `value` may have been preallocated by an old-style `__init__`
                    // if it was a preceding candidate for overload resolution.
                    if (self_value_and_holder)
                        self_value_and_holder.type->dealloc(self_value_and_holder);

                    call.init_self = PyTuple_GET_ITEM(args_in, 0);
                    call.args.push_back(reinterpret_cast<PyObject *>(&self_value_and_holder));
                    call.args_convert.push_back(false);
                    ++args_copied;
                }

                // 1. Copy any position arguments given.
                bool bad_arg = false;
                for (; args_copied < args_to_copy; ++args_copied) {
                    const argument_record *arg_rec = args_copied < func.args.size() ? &func.args[args_copied] : nullptr;
                    if (kwargs_in && arg_rec && arg_rec->name && PyDict_GetItemString(kwargs_in, arg_rec->name)) {
                        bad_arg = true;
                        break;
                    }

                    handle arg(PyTuple_GET_ITEM(args_in, args_copied));
                    if (arg_rec && !arg_rec->none && arg.is_none()) {
                        bad_arg = true;
                        break;
                    }
                    call.args.push_back(arg);
                    call.args_convert.push_back(arg_rec ? arg_rec->convert : true);
                }
                if (bad_arg)
                    continue; // Maybe it was meant for another overload (issue #688)

                // We'll need to copy this if we steal some kwargs for defaults
                dict kwargs = reinterpret_borrow<dict>(kwargs_in);

                // 2. Check kwargs and, failing that, defaults that may help complete the list
                if (args_copied < num_args) {
                    bool copied_kwargs = false;

                    for (; args_copied < num_args; ++args_copied) {
                        const auto &arg = func.args[args_copied];

                        handle value;
                        if (kwargs_in && arg.name)
                            value = PyDict_GetItemString(kwargs.ptr(), arg.name);

                        if (value) {
                            // Consume a kwargs value
                            if (!copied_kwargs) {
                                kwargs = reinterpret_steal<dict>(PyDict_Copy(kwargs.ptr()));
                                copied_kwargs = true;
                            }
                            PyDict_DelItemString(kwargs.ptr(), arg.name);
                        } else if (arg.value) {
                            value = arg.value;
                        }

                        if (value) {
                            call.args.push_back(value);
                            call.args_convert.push_back(arg.convert);
                        }
                        else
                            break;
                    }

                    if (args_copied < num_args)
                        continue; // Not enough arguments, defaults, or kwargs to fill the positional arguments
                }

                // 3. Check everything was consumed (unless we have a kwargs arg)
                if (kwargs && kwargs.size() > 0 && !func.has_kwargs)
                    continue; // Unconsumed kwargs, but no py::kwargs argument to accept them

                // 4a. If we have a py::args argument, create a new tuple with leftovers
                if (func.has_args) {
                    tuple extra_args;
                    if (args_to_copy == 0) {
                        // We didn't copy out any position arguments from the args_in tuple, so we
                        // can reuse it directly without copying:
                        extra_args = reinterpret_borrow<tuple>(args_in);
                    } else if (args_copied >= n_args_in) {
                        extra_args = tuple(0);
                    } else {
                        size_t args_size = n_args_in - args_copied;
                        extra_args = tuple(args_size);
                        for (size_t i = 0; i < args_size; ++i) {
                            extra_args[i] = PyTuple_GET_ITEM(args_in, args_copied + i);
                        }
                    }
                    call.args.push_back(extra_args);
                    call.args_convert.push_back(false);
                    call.args_ref = std::move(extra_args);
                }

                // 4b. If we have a py::kwargs, pass on any remaining kwargs
                if (func.has_kwargs) {
                    if (!kwargs.ptr())
                        kwargs = dict(); // If we didn't get one, send an empty one
                    call.args.push_back(kwargs);
                    call.args_convert.push_back(false);
                    call.kwargs_ref = std::move(kwargs);
                }

                // 5. Put everything in a vector.  Not technically step 5, we've been building it
                // in `call.args` all along.
                #if !defined(NDEBUG)
                if (call.args.size() != func.nargs || call.args_convert.size() != func.nargs)
                    pybind11_fail("Internal error: function call dispatcher inserted wrong number of arguments!");
                #endif

                std::vector<bool> second_pass_convert;
                if (overloaded) {
                    // We're in the first no-convert pass, so swap out the conversion flags for a
                    // set of all-false flags.  If the call fails, we'll swap the flags back in for
                    // the conversion-allowed call below.
                    second_pass_convert.resize(func.nargs, false);
                    call.args_convert.swap(second_pass_convert);
                }

                // 6. Call the function.
                try {
                    loader_life_support guard{};
                    result = func.impl(call);
                } catch (reference_cast_error &) {
                    result = PYBIND11_TRY_NEXT_OVERLOAD;
                }

                if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD)
                    break;

                if (overloaded) {
                    // The (overloaded) call failed; if the call has at least one argument that
                    // permits conversion (i.e. it hasn't been explicitly specified `.noconvert()`)
                    // then add this call to the list of second pass overloads to try.
                    for (size_t i = func.is_method ? 1 : 0; i < pos_args; i++) {
                        if (second_pass_convert[i]) {
                            // Found one: swap the converting flags back in and store the call for
                            // the second pass.
                            call.args_convert.swap(second_pass_convert);
                            second_pass.push_back(std::move(call));
                            break;
                        }
                    }
                }
            }

            if (overloaded && !second_pass.empty() && result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
                // The no-conversion pass finished without success, try again with conversion allowed
                for (auto &call : second_pass) {
                    try {
                        loader_life_support guard{};
                        result = call.func.impl(call);
                    } catch (reference_cast_error &) {
                        result = PYBIND11_TRY_NEXT_OVERLOAD;
                    }

                    if (result.ptr() != PYBIND11_TRY_NEXT_OVERLOAD) {
                        // The error reporting logic below expects 'it' to be valid, as it would be
                        // if we'd encountered this failure in the first-pass loop.
                        if (!result)
                            it = &call.func;
                        break;
                    }
                }
            }
        } catch (error_already_set &e) {
            e.restore();
            return nullptr;
#if defined(__GNUG__) && !defined(__clang__)
        } catch ( abi::__forced_unwind& ) {
            throw;
#endif
        } catch (...) {
            /* When an exception is caught, give each registered exception
               translator a chance to translate it to a Python exception
               in reverse order of registration.

               A translator may choose to do one of the following:

                - catch the exception and call PyErr_SetString or PyErr_SetObject
                  to set a standard (or custom) Python exception, or
                - do nothing and let the exception fall through to the next translator, or
                - delegate translation to the next translator by throwing a new type of exception. */

            auto last_exception = std::current_exception();
            auto &registered_exception_translators = get_internals().registered_exception_translators;
            for (auto& translator : registered_exception_translators) {
                try {
                    translator(last_exception);
                } catch (...) {
                    last_exception = std::current_exception();
                    continue;
                }
                return nullptr;
            }
            PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
            return nullptr;
        }

        auto append_note_if_missing_header_is_suspected = [](std::string &msg) {
            if (msg.find("std::") != std::string::npos) {
                msg += "\n\n"
                       "Did you forget to `#include <pybind11/stl.h>`? Or <pybind11/complex.h>,\n"
                       "<pybind11/functional.h>, <pybind11/chrono.h>, etc. Some automatic\n"
                       "conversions are optional and require extra headers to be included\n"
                       "when compiling your pybind11 module.";
            }
        };

        if (result.ptr() == PYBIND11_TRY_NEXT_OVERLOAD) {
            if (overloads->is_operator)
                return handle(Py_NotImplemented).inc_ref().ptr();

            std::string msg = std::string(overloads->name) + "(): incompatible " +
                std::string(overloads->is_constructor ? "constructor" : "function") +
                " arguments. The following argument types are supported:\n";

            int ctr = 0;
            for (const function_record *it2 = overloads; it2 != nullptr; it2 = it2->next) {
                msg += "    "+ std::to_string(++ctr) + ". ";

                bool wrote_sig = false;
                if (overloads->is_constructor) {
                    // For a constructor, rewrite `(self: Object, arg0, ...) -> NoneType` as `Object(arg0, ...)`
                    std::string sig = it2->signature;
                    size_t start = sig.find('(') + 7; // skip "(self: "
                    if (start < sig.size()) {
                        // End at the , for the next argument
                        size_t end = sig.find(", "), next = end + 2;
                        size_t ret = sig.rfind(" -> ");
                        // Or the ), if there is no comma:
                        if (end >= sig.size()) next = end = sig.find(')');
                        if (start < end && next < sig.size()) {
                            msg.append(sig, start, end - start);
                            msg += '(';
                            msg.append(sig, next, ret - next);
                            wrote_sig = true;
                        }
                    }
                }
                if (!wrote_sig) msg += it2->signature;

                msg += "\n";
            }
            msg += "\nInvoked with: ";
            auto args_ = reinterpret_borrow<tuple>(args_in);
            bool some_args = false;
            for (size_t ti = overloads->is_constructor ? 1 : 0; ti < args_.size(); ++ti) {
                if (!some_args) some_args = true;
                else msg += ", ";
                try {
                    msg += pybind11::repr(args_[ti]);
                } catch (const error_already_set&) {
                    msg += "<repr raised Error>";
                }
            }
            if (kwargs_in) {
                auto kwargs = reinterpret_borrow<dict>(kwargs_in);
                if (kwargs.size() > 0) {
                    if (some_args) msg += "; ";
                    msg += "kwargs: ";
                    bool first = true;
                    for (auto kwarg : kwargs) {
                        if (first) first = false;
                        else msg += ", ";
                        msg += pybind11::str("{}=").format(kwarg.first);
                        try {
                            msg += pybind11::repr(kwarg.second);
                        } catch (const error_already_set&) {
                            msg += "<repr raised Error>";
                        }
                    }
                }
            }

            append_note_if_missing_header_is_suspected(msg);
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else if (!result) {
            std::string msg = "Unable to convert function return value to a "
                              "Python type! The signature was\n\t";
            msg += it->signature;
            append_note_if_missing_header_is_suspected(msg);
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return nullptr;
        } else {
            if (overloads->is_constructor && !self_value_and_holder.holder_constructed()) {
                auto *pi = reinterpret_cast<instance *>(parent.ptr());
                self_value_and_holder.type->init_instance(pi, nullptr);
            }
            return result.ptr();
        }
    }
};

/// Wrapper for Python extension modules
class module : public object {
public:
    PYBIND11_OBJECT_DEFAULT(module, object, PyModule_Check)

    /// Create a new top-level Python module with the given name and docstring
    explicit module(const char *name, const char *doc = nullptr) {
        if (!options::show_user_defined_docstrings()) doc = nullptr;
#if PY_MAJOR_VERSION >= 3
        PyModuleDef *def = new PyModuleDef();
        std::memset(def, 0, sizeof(PyModuleDef));
        def->m_name = name;
        def->m_doc = doc;
        def->m_size = -1;
        Py_INCREF(def);
        m_ptr = PyModule_Create(def);
#else
        m_ptr = Py_InitModule3(name, nullptr, doc);
#endif
        if (m_ptr == nullptr)
            pybind11_fail("Internal error in module::module()");
        inc_ref();
    }

    /** \rst
        Create Python binding for a new function within the module scope. ``Func``
        can be a plain C++ function, a function pointer, or a lambda function. For
        details on the ``Extra&& ... extra`` argument, see section :ref:`extras`.
    \endrst */
    template <typename Func, typename... Extra>
    module &def(const char *name_, Func &&f, const Extra& ... extra) {
        cpp_function func(std::forward<Func>(f), name(name_), scope(*this),
                          sibling(getattr(*this, name_, none())), extra...);
        // NB: allow overwriting here because cpp_function sets up a chain with the intention of
        // overwriting (and has already checked internally that it isn't overwriting non-functions).
        add_object(name_, func, true /* overwrite */);
        return *this;
    }

    /** \rst
        Create and return a new Python submodule with the given name and docstring.
        This also works recursively, i.e.

        .. code-block:: cpp

            py::module m("example", "pybind11 example plugin");
            py::module m2 = m.def_submodule("sub", "A submodule of 'example'");
            py::module m3 = m2.def_submodule("subsub", "A submodule of 'example.sub'");
    \endrst */
    module def_submodule(const char *name, const char *doc = nullptr) {
        std::string full_name = std::string(PyModule_GetName(m_ptr))
            + std::string(".") + std::string(name);
        auto result = reinterpret_borrow<module>(PyImport_AddModule(full_name.c_str()));
        if (doc && options::show_user_defined_docstrings())
            result.attr("__doc__") = pybind11::str(doc);
        attr(name) = result;
        return result;
    }

    /// Import and return a module or throws `error_already_set`.
    static module import(const char *name) {
        PyObject *obj = PyImport_ImportModule(name);
        if (!obj)
            throw error_already_set();
        return reinterpret_steal<module>(obj);
    }

    /// Reload the module or throws `error_already_set`.
    void reload() {
        PyObject *obj = PyImport_ReloadModule(ptr());
        if (!obj)
            throw error_already_set();
        *this = reinterpret_steal<module>(obj);
    }

    // Adds an object to the module using the given name.  Throws if an object with the given name
    // already exists.
    //
    // overwrite should almost always be false: attempting to overwrite objects that pybind11 has
    // established will, in most cases, break things.
    PYBIND11_NOINLINE void add_object(const char *name, handle obj, bool overwrite = false) {
        if (!overwrite && hasattr(*this, name))
            pybind11_fail("Error during initialization: multiple incompatible definitions with name \"" +
                    std::string(name) + "\"");

        PyModule_AddObject(ptr(), name, obj.inc_ref().ptr() /* steals a reference */);
    }
};

/// \ingroup python_builtins
/// Return a dictionary representing the global variables in the current execution frame,
/// or ``__main__.__dict__`` if there is no frame (usually when the interpreter is embedded).
inline dict globals() {
    PyObject *p = PyEval_GetGlobals();
    return reinterpret_borrow<dict>(p ? p : module::import("__main__").attr("__dict__").ptr());
}

PYBIND11_NAMESPACE_BEGIN(detail)
/// Generic support for creating new Python heap types
class generic_type : public object {
    template <typename...> friend class class_;
public:
    PYBIND11_OBJECT_DEFAULT(generic_type, object, PyType_Check)
protected:
    void initialize(const type_record &rec) {
        if (rec.scope && hasattr(rec.scope, rec.name))
            pybind11_fail("generic_type: cannot initialize type \"" + std::string(rec.name) +
                          "\": an object with that name is already defined");

        if (rec.module_local ? get_local_type_info(*rec.type) : get_global_type_info(*rec.type))
            pybind11_fail("generic_type: type \"" + std::string(rec.name) +
                          "\" is already registered!");

        m_ptr = make_new_python_type(rec);

        /* Register supplemental type information in C++ dict */
        auto *tinfo = new detail::type_info();
        tinfo->type = (PyTypeObject *) m_ptr;
        tinfo->cpptype = rec.type;
        tinfo->type_size = rec.type_size;
        tinfo->type_align = rec.type_align;
        tinfo->operator_new = rec.operator_new;
        tinfo->holder_size_in_ptrs = size_in_ptrs(rec.holder_size);
        tinfo->init_instance = rec.init_instance;
        tinfo->dealloc = rec.dealloc;
        tinfo->simple_type = true;
        tinfo->simple_ancestors = true;
        tinfo->default_holder = rec.default_holder;
        tinfo->module_local = rec.module_local;

        auto &internals = get_internals();
        auto tindex = std::type_index(*rec.type);
        tinfo->direct_conversions = &internals.direct_conversions[tindex];
        if (rec.module_local)
            registered_local_types_cpp()[tindex] = tinfo;
        else
            internals.registered_types_cpp[tindex] = tinfo;
        internals.registered_types_py[(PyTypeObject *) m_ptr] = { tinfo };

        if (rec.bases.size() > 1 || rec.multiple_inheritance) {
            mark_parents_nonsimple(tinfo->type);
            tinfo->simple_ancestors = false;
        }
        else if (rec.bases.size() == 1) {
            auto parent_tinfo = get_type_info((PyTypeObject *) rec.bases[0].ptr());
            tinfo->simple_ancestors = parent_tinfo->simple_ancestors;
        }

        if (rec.module_local) {
            // Stash the local typeinfo and loader so that external modules can access it.
            tinfo->module_local_load = &type_caster_generic::local_load;
            setattr(m_ptr, PYBIND11_MODULE_LOCAL_ID, capsule(tinfo));
        }
    }

    /// Helper function which tags all parents of a type using mult. inheritance
    void mark_parents_nonsimple(PyTypeObject *value) {
        auto t = reinterpret_borrow<tuple>(value->tp_bases);
        for (handle h : t) {
            auto tinfo2 = get_type_info((PyTypeObject *) h.ptr());
            if (tinfo2)
                tinfo2->simple_type = false;
            mark_parents_nonsimple((PyTypeObject *) h.ptr());
        }
    }

    void install_buffer_funcs(
            buffer_info *(*get_buffer)(PyObject *, void *),
            void *get_buffer_data) {
        PyHeapTypeObject *type = (PyHeapTypeObject*) m_ptr;
        auto tinfo = detail::get_type_info(&type->ht_type);

        if (!type->ht_type.tp_as_buffer)
            pybind11_fail(
                "To be able to register buffer protocol support for the type '" +
                std::string(tinfo->type->tp_name) +
                "' the associated class<>(..) invocation must "
                "include the pybind11::buffer_protocol() annotation!");

        tinfo->get_buffer = get_buffer;
        tinfo->get_buffer_data = get_buffer_data;
    }

    // rec_func must be set for either fget or fset.
    void def_property_static_impl(const char *name,
                                  handle fget, handle fset,
                                  detail::function_record *rec_func) {
        const auto is_static = rec_func && !(rec_func->is_method && rec_func->scope);
        const auto has_doc = rec_func && rec_func->doc && pybind11::options::show_user_defined_docstrings();
        auto property = handle((PyObject *) (is_static ? get_internals().static_property_type
                                                       : &PyProperty_Type));
        attr(name) = property(fget.ptr() ? fget : none(),
                              fset.ptr() ? fset : none(),
                              /*deleter*/none(),
                              pybind11::str(has_doc ? rec_func->doc : ""));
    }
};

/// Set the pointer to operator new if it exists. The cast is needed because it can be overloaded.
template <typename T, typename = void_t<decltype(static_cast<void *(*)(size_t)>(T::operator new))>>
void set_operator_new(type_record *r) { r->operator_new = &T::operator new; }

template <typename> void set_operator_new(...) { }

template <typename T, typename SFINAE = void> struct has_operator_delete : std::false_type { };
template <typename T> struct has_operator_delete<T, void_t<decltype(static_cast<void (*)(void *)>(T::operator delete))>>
    : std::true_type { };
template <typename T, typename SFINAE = void> struct has_operator_delete_size : std::false_type { };
template <typename T> struct has_operator_delete_size<T, void_t<decltype(static_cast<void (*)(void *, size_t)>(T::operator delete))>>
    : std::true_type { };
/// Call class-specific delete if it exists or global otherwise. Can also be an overload set.
template <typename T, enable_if_t<has_operator_delete<T>::value, int> = 0>
void call_operator_delete(T *p, size_t, size_t) { T::operator delete(p); }
template <typename T, enable_if_t<!has_operator_delete<T>::value && has_operator_delete_size<T>::value, int> = 0>
void call_operator_delete(T *p, size_t s, size_t) { T::operator delete(p, s); }

inline void call_operator_delete(void *p, size_t s, size_t a) {
    (void)s; (void)a;
    #if defined(__cpp_aligned_new) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
        if (a > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
            #ifdef __cpp_sized_deallocation
                ::operator delete(p, s, std::align_val_t(a));
            #else
                ::operator delete(p, std::align_val_t(a));
            #endif
            return;
        }
    #endif
    #ifdef __cpp_sized_deallocation
        ::operator delete(p, s);
    #else
        ::operator delete(p);
    #endif
}

inline void add_class_method(object& cls, const char *name_, const cpp_function &cf) {
    cls.attr(cf.name()) = cf;
    if (strcmp(name_, "__eq__") == 0 && !cls.attr("__dict__").contains("__hash__")) {
      cls.attr("__hash__") = none();
    }
}

PYBIND11_NAMESPACE_END(detail)

/// Given a pointer to a member function, cast it to its `Derived` version.
/// Forward everything else unchanged.
template <typename /*Derived*/, typename F>
auto method_adaptor(F &&f) -> decltype(std::forward<F>(f)) { return std::forward<F>(f); }

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...)) -> Return (Derived::*)(Args...) {
    static_assert(detail::is_accessible_base_of<Class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

template <typename Derived, typename Return, typename Class, typename... Args>
auto method_adaptor(Return (Class::*pmf)(Args...) const) -> Return (Derived::*)(Args...) const {
    static_assert(detail::is_accessible_base_of<Class, Derived>::value,
        "Cannot bind an inaccessible base class method; use a lambda definition instead");
    return pmf;
}

template <typename type_, typename... options>
class class_ : public detail::generic_type {
    template <typename T> using is_holder = detail::is_holder_type<type_, T>;
    template <typename T> using is_subtype = detail::is_strict_base_of<type_, T>;
    template <typename T> using is_base = detail::is_strict_base_of<T, type_>;
    // struct instead of using here to help MSVC:
    template <typename T> struct is_valid_class_option :
        detail::any_of<is_holder<T>, is_subtype<T>, is_base<T>> {};

public:
    using type = type_;
    using type_alias = detail::exactly_one_t<is_subtype, void, options...>;
    constexpr static bool has_alias = !std::is_void<type_alias>::value;
    using holder_type = detail::exactly_one_t<is_holder, std::unique_ptr<type>, options...>;

    static_assert(detail::all_of<is_valid_class_option<options>...>::value,
            "Unknown/invalid class_ template parameters provided");

    static_assert(!has_alias || std::is_polymorphic<type>::value,
            "Cannot use an alias class with a non-polymorphic type");

    PYBIND11_OBJECT(class_, generic_type, PyType_Check)

    template <typename... Extra>
    class_(handle scope, const char *name, const Extra &... extra) {
        using namespace detail;

        // MI can only be specified via class_ template options, not constructor parameters
        static_assert(
            none_of<is_pyobject<Extra>...>::value || // no base class arguments, or:
            (   constexpr_sum(is_pyobject<Extra>::value...) == 1 && // Exactly one base
                constexpr_sum(is_base<options>::value...)   == 0 && // no template option bases
                none_of<std::is_same<multiple_inheritance, Extra>...>::value), // no multiple_inheritance attr
            "Error: multiple inheritance bases must be specified via class_ template options");

        type_record record;
        record.scope = scope;
        record.name = name;
        record.type = &typeid(type);
        record.type_size = sizeof(conditional_t<has_alias, type_alias, type>);
        record.type_align = alignof(conditional_t<has_alias, type_alias, type>&);
        record.holder_size = sizeof(holder_type);
        record.init_instance = init_instance;
        record.dealloc = dealloc;
        record.default_holder = detail::is_instantiation<std::unique_ptr, holder_type>::value;

        set_operator_new<type>(&record);

        /* Register base classes specified via template arguments to class_, if any */
        PYBIND11_EXPAND_SIDE_EFFECTS(add_base<options>(record));

        /* Process optional arguments, if any */
        process_attributes<Extra...>::init(extra..., &record);

        generic_type::initialize(record);

        if (has_alias) {
            auto &instances = record.module_local ? registered_local_types_cpp() : get_internals().registered_types_cpp;
            instances[std::type_index(typeid(type_alias))] = instances[std::type_index(typeid(type))];
        }
    }

    template <typename Base, detail::enable_if_t<is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &rec) {
        rec.add_base(typeid(Base), [](void *src) -> void * {
            return static_cast<Base *>(reinterpret_cast<type *>(src));
        });
    }

    template <typename Base, detail::enable_if_t<!is_base<Base>::value, int> = 0>
    static void add_base(detail::type_record &) { }

    template <typename Func, typename... Extra>
    class_ &def(const char *name_, Func&& f, const Extra&... extra) {
        cpp_function cf(method_adaptor<type>(std::forward<Func>(f)), name(name_), is_method(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        add_class_method(*this, name_, cf);
        return *this;
    }

    template <typename Func, typename... Extra> class_ &
    def_static(const char *name_, Func &&f, const Extra&... extra) {
        static_assert(!std::is_member_function_pointer<Func>::value,
                "def_static(...) called with a non-static member function pointer");
        cpp_function cf(std::forward<Func>(f), name(name_), scope(*this),
                        sibling(getattr(*this, name_, none())), extra...);
        attr(cf.name()) = staticmethod(cf);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ &def(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute(*this, extra...);
        return *this;
    }

    template <detail::op_id id, detail::op_type ot, typename L, typename R, typename... Extra>
    class_ & def_cast(const detail::op_<id, ot, L, R> &op, const Extra&... extra) {
        op.execute_cast(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::constructor<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(const detail::initimpl::alias_constructor<Args...> &init, const Extra&... extra) {
        init.execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::factory<Args...> &&init, const Extra&... extra) {
        std::move(init).execute(*this, extra...);
        return *this;
    }

    template <typename... Args, typename... Extra>
    class_ &def(detail::initimpl::pickle_factory<Args...> &&pf, const Extra &...extra) {
        std::move(pf).execute(*this, extra...);
        return *this;
    }

    template <typename Func> class_& def_buffer(Func &&func) {
        struct capture { Func func; };
        capture *ptr = new capture { std::forward<Func>(func) };
        install_buffer_funcs([](PyObject *obj, void *ptr) -> buffer_info* {
            detail::make_caster<type> caster;
            if (!caster.load(obj, false))
                return nullptr;
            return new buffer_info(((capture *) ptr)->func(caster));
        }, ptr);
        return *this;
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...)) {
        return def_buffer([func] (type &obj) { return (obj.*func)(); });
    }

    template <typename Return, typename Class, typename... Args>
    class_ &def_buffer(Return (Class::*func)(Args...) const) {
        return def_buffer([func] (const type &obj) { return (obj.*func)(); });
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_readwrite() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this)),
                     fset([pm](type &c, const D &value) { c.*pm = value; }, is_method(*this));
        def_property(name, fget, fset, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename C, typename D, typename... Extra>
    class_ &def_readonly(const char *name, const D C::*pm, const Extra& ...extra) {
        static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_readonly() requires a class member (or base class member)");
        cpp_function fget([pm](const type &c) -> const D &{ return c.*pm; }, is_method(*this));
        def_property_readonly(name, fget, return_value_policy::reference_internal, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readwrite_static(const char *name, D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this)),
                     fset([pm](object, const D &value) { *pm = value; }, scope(*this));
        def_property_static(name, fget, fset, return_value_policy::reference, extra...);
        return *this;
    }

    template <typename D, typename... Extra>
    class_ &def_readonly_static(const char *name, const D *pm, const Extra& ...extra) {
        cpp_function fget([pm](object) -> const D &{ return *pm; }, scope(*this));
        def_property_readonly_static(name, fget, return_value_policy::reference, extra...);
        return *this;
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly(name, cpp_function(method_adaptor<type>(fget)),
                                     return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_readonly_static(const char *name, const Getter &fget, const Extra& ...extra) {
        return def_property_readonly_static(name, cpp_function(fget), return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_readonly_static(const char *name, const cpp_function &fget, const Extra& ...extra) {
        return def_property_static(name, fget, nullptr, extra...);
    }

    /// Uses return_value_policy::reference_internal by default
    template <typename Getter, typename Setter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const Setter &fset, const Extra& ...extra) {
        return def_property(name, fget, cpp_function(method_adaptor<type>(fset)), extra...);
    }
    template <typename Getter, typename... Extra>
    class_ &def_property(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property(name, cpp_function(method_adaptor<type>(fget)), fset,
                            return_value_policy::reference_internal, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, fget, fset, is_method(*this), extra...);
    }

    /// Uses return_value_policy::reference by default
    template <typename Getter, typename... Extra>
    class_ &def_property_static(const char *name, const Getter &fget, const cpp_function &fset, const Extra& ...extra) {
        return def_property_static(name, cpp_function(fget), fset, return_value_policy::reference, extra...);
    }

    /// Uses cpp_function's return_value_policy by default
    template <typename... Extra>
    class_ &def_property_static(const char *name, const cpp_function &fget, const cpp_function &fset, const Extra& ...extra) {
        static_assert( 0 == detail::constexpr_sum(std::is_base_of<arg, Extra>::value...),
                      "Argument annotations are not allowed for properties");
        auto rec_fget = get_function_record(fget), rec_fset = get_function_record(fset);
        auto *rec_active = rec_fget;
        if (rec_fget) {
           char *doc_prev = rec_fget->doc; /* 'extra' field may include a property-specific documentation string */
           detail::process_attributes<Extra...>::init(extra..., rec_fget);
           if (rec_fget->doc && rec_fget->doc != doc_prev) {
              free(doc_prev);
              rec_fget->doc = strdup(rec_fget->doc);
           }
        }
        if (rec_fset) {
            char *doc_prev = rec_fset->doc;
            detail::process_attributes<Extra...>::init(extra..., rec_fset);
            if (rec_fset->doc && rec_fset->doc != doc_prev) {
                free(doc_prev);
                rec_fset->doc = strdup(rec_fset->doc);
            }
            if (! rec_active) rec_active = rec_fset;
        }
        def_property_static_impl(name, fget, fset, rec_active);
        return *this;
    }

private:
    /// Initialize holder object, variant 1: object derives from enable_shared_from_this
    template <typename T>
    static void init_holder(detail::instance *inst, detail::value_and_holder &v_h,
            const holder_type * /* unused */, const std::enable_shared_from_this<T> * /* dummy */) {
        try {
            auto sh = std::dynamic_pointer_cast<typename holder_type::element_type>(
                    v_h.value_ptr<type>()->shared_from_this());
            if (sh) {
                new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(sh));
                v_h.set_holder_constructed();
            }
        } catch (const std::bad_weak_ptr &) {}

        if (!v_h.holder_constructed() && inst->owned) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
            const holder_type *holder_ptr, std::true_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>())) holder_type(*reinterpret_cast<const holder_type *>(holder_ptr));
    }

    static void init_holder_from_existing(const detail::value_and_holder &v_h,
            const holder_type *holder_ptr, std::false_type /*is_copy_constructible*/) {
        new (std::addressof(v_h.holder<holder_type>())) holder_type(std::move(*const_cast<holder_type *>(holder_ptr)));
    }

    /// Initialize holder object, variant 2: try to construct from existing holder object, if possible
    static void init_holder(detail::instance *inst, detail::value_and_holder &v_h,
            const holder_type *holder_ptr, const void * /* dummy -- not enable_shared_from_this<T>) */) {
        if (holder_ptr) {
            init_holder_from_existing(v_h, holder_ptr, std::is_copy_constructible<holder_type>());
            v_h.set_holder_constructed();
        } else if (inst->owned || detail::always_construct_holder<holder_type>::value) {
            new (std::addressof(v_h.holder<holder_type>())) holder_type(v_h.value_ptr<type>());
            v_h.set_holder_constructed();
        }
    }

    /// Performs instance initialization including constructing a holder and registering the known
    /// instance.  Should be called as soon as the `type` value_ptr is set for an instance.  Takes an
    /// optional pointer to an existing holder to use; if not specified and the instance is
    /// `.owned`, a new holder will be constructed to manage the value pointer.
    static void init_instance(detail::instance *inst, const void *holder_ptr) {
        auto v_h = inst->get_value_and_holder(detail::get_type_info(typeid(type)));
        if (!v_h.instance_registered()) {
            register_instance(inst, v_h.value_ptr(), v_h.type);
            v_h.set_instance_registered();
        }
        init_holder(inst, v_h, (const holder_type *) holder_ptr, v_h.value_ptr<type>());
    }

    /// Deallocates an instance; via holder, if constructed; otherwise via operator delete.
    static void dealloc(detail::value_and_holder &v_h) {
        // We could be deallocating because we are cleaning up after a Python exception.
        // If so, the Python error indicator will be set. We need to clear that before
        // running the destructor, in case the destructor code calls more Python.
        // If we don't, the Python API will exit with an exception, and pybind11 will
        // throw error_already_set from the C++ destructor which is forbidden and triggers
        // std::terminate().
        error_scope scope;
        if (v_h.holder_constructed()) {
            v_h.holder<holder_type>().~holder_type();
            v_h.set_holder_constructed(false);
        }
        else {
            detail::call_operator_delete(v_h.value_ptr<type>(),
                v_h.type->type_size,
                v_h.type->type_align
            );
        }
        v_h.value_ptr() = nullptr;
    }

    static detail::function_record *get_function_record(handle h) {
        h = detail::get_function(h);
        return h ? (detail::function_record *) reinterpret_borrow<capsule>(PyCFunction_GET_SELF(h.ptr()))
                 : nullptr;
    }
};

/// Binds an existing constructor taking arguments Args...
template <typename... Args> detail::initimpl::constructor<Args...> init() { return {}; }
/// Like `init<Args...>()`, but the instance is always constructed through the alias class (even
/// when not inheriting on the Python side).
template <typename... Args> detail::initimpl::alias_constructor<Args...> init_alias() { return {}; }

/// Binds a factory function as a constructor
template <typename Func, typename Ret = detail::initimpl::factory<Func>>
Ret init(Func &&f) { return {std::forward<Func>(f)}; }

/// Dual-argument factory function: the first function is called when no alias is needed, the second
/// when an alias is needed (i.e. due to python-side inheritance).  Arguments must be identical.
template <typename CFunc, typename AFunc, typename Ret = detail::initimpl::factory<CFunc, AFunc>>
Ret init(CFunc &&c, AFunc &&a) {
    return {std::forward<CFunc>(c), std::forward<AFunc>(a)};
}

/// Binds pickling functions `__getstate__` and `__setstate__` and ensures that the type
/// returned by `__getstate__` is the same as the argument accepted by `__setstate__`.
template <typename GetState, typename SetState>
detail::initimpl::pickle_factory<GetState, SetState> pickle(GetState &&g, SetState &&s) {
    return {std::forward<GetState>(g), std::forward<SetState>(s)};
}

PYBIND11_NAMESPACE_BEGIN(detail)
struct enum_base {
    enum_base(handle base, handle parent) : m_base(base), m_parent(parent) { }

    PYBIND11_NOINLINE void init(bool is_arithmetic, bool is_convertible) {
        m_base.attr("__entries") = dict();
        auto property = handle((PyObject *) &PyProperty_Type);
        auto static_property = handle((PyObject *) get_internals().static_property_type);

        m_base.attr("__repr__") = cpp_function(
            [](handle arg) -> str {
                handle type = arg.get_type();
                object type_name = type.attr("__name__");
                dict entries = type.attr("__entries");
                for (const auto &kv : entries) {
                    object other = kv.second[int_(0)];
                    if (other.equal(arg))
                        return pybind11::str("{}.{}").format(type_name, kv.first);
                }
                return pybind11::str("{}.???").format(type_name);
            }, name("__repr__"), is_method(m_base)
        );

        m_base.attr("name") = property(cpp_function(
            [](handle arg) -> str {
                dict entries = arg.get_type().attr("__entries");
                for (const auto &kv : entries) {
                    if (handle(kv.second[int_(0)]).equal(arg))
                        return pybind11::str(kv.first);
                }
                return "???";
            }, name("name"), is_method(m_base)
        ));

        m_base.attr("__doc__") = static_property(cpp_function(
            [](handle arg) -> std::string {
                std::string docstring;
                dict entries = arg.attr("__entries");
                if (((PyTypeObject *) arg.ptr())->tp_doc)
                    docstring += std::string(((PyTypeObject *) arg.ptr())->tp_doc) + "\n\n";
                docstring += "Members:";
                for (const auto &kv : entries) {
                    auto key = std::string(pybind11::str(kv.first));
                    auto comment = kv.second[int_(1)];
                    docstring += "\n\n  " + key;
                    if (!comment.is_none())
                        docstring += " : " + (std::string) pybind11::str(comment);
                }
                return docstring;
            }, name("__doc__")
        ), none(), none(), "");

        m_base.attr("__members__") = static_property(cpp_function(
            [](handle arg) -> dict {
                dict entries = arg.attr("__entries"), m;
                for (const auto &kv : entries)
                    m[kv.first] = kv.second[int_(0)];
                return m;
            }, name("__members__")), none(), none(), ""
        );

        #define PYBIND11_ENUM_OP_STRICT(op, expr, strict_behavior)                     \
            m_base.attr(op) = cpp_function(                                            \
                [](object a, object b) {                                               \
                    if (!a.get_type().is(b.get_type()))                                \
                        strict_behavior;                                               \
                    return expr;                                                       \
                },                                                                     \
                name(op), is_method(m_base))

        #define PYBIND11_ENUM_OP_CONV(op, expr)                                        \
            m_base.attr(op) = cpp_function(                                            \
                [](object a_, object b_) {                                             \
                    int_ a(a_), b(b_);                                                 \
                    return expr;                                                       \
                },                                                                     \
                name(op), is_method(m_base))

        #define PYBIND11_ENUM_OP_CONV_LHS(op, expr)                                    \
            m_base.attr(op) = cpp_function(                                            \
                [](object a_, object b) {                                              \
                    int_ a(a_);                                                        \
                    return expr;                                                       \
                },                                                                     \
                name(op), is_method(m_base))

        if (is_convertible) {
            PYBIND11_ENUM_OP_CONV_LHS("__eq__", !b.is_none() &&  a.equal(b));
            PYBIND11_ENUM_OP_CONV_LHS("__ne__",  b.is_none() || !a.equal(b));

            if (is_arithmetic) {
                PYBIND11_ENUM_OP_CONV("__lt__",   a <  b);
                PYBIND11_ENUM_OP_CONV("__gt__",   a >  b);
                PYBIND11_ENUM_OP_CONV("__le__",   a <= b);
                PYBIND11_ENUM_OP_CONV("__ge__",   a >= b);
                PYBIND11_ENUM_OP_CONV("__and__",  a &  b);
                PYBIND11_ENUM_OP_CONV("__rand__", a &  b);
                PYBIND11_ENUM_OP_CONV("__or__",   a |  b);
                PYBIND11_ENUM_OP_CONV("__ror__",  a |  b);
                PYBIND11_ENUM_OP_CONV("__xor__",  a ^  b);
                PYBIND11_ENUM_OP_CONV("__rxor__", a ^  b);
                m_base.attr("__invert__") = cpp_function(
                    [](object arg) { return ~(int_(arg)); }, name("__invert__"), is_method(m_base));
            }
        } else {
            PYBIND11_ENUM_OP_STRICT("__eq__",  int_(a).equal(int_(b)), return false);
            PYBIND11_ENUM_OP_STRICT("__ne__", !int_(a).equal(int_(b)), return true);

            if (is_arithmetic) {
                #define PYBIND11_THROW throw type_error("Expected an enumeration of matching type!");
                PYBIND11_ENUM_OP_STRICT("__lt__", int_(a) <  int_(b), PYBIND11_THROW);
                PYBIND11_ENUM_OP_STRICT("__gt__", int_(a) >  int_(b), PYBIND11_THROW);
                PYBIND11_ENUM_OP_STRICT("__le__", int_(a) <= int_(b), PYBIND11_THROW);
                PYBIND11_ENUM_OP_STRICT("__ge__", int_(a) >= int_(b), PYBIND11_THROW);
                #undef PYBIND11_THROW
            }
        }

        #undef PYBIND11_ENUM_OP_CONV_LHS
        #undef PYBIND11_ENUM_OP_CONV
        #undef PYBIND11_ENUM_OP_STRICT

        m_base.attr("__getstate__") = cpp_function(
            [](object arg) { return int_(arg); }, name("__getstate__"), is_method(m_base));

        m_base.attr("__hash__") = cpp_function(
            [](object arg) { return int_(arg); }, name("__hash__"), is_method(m_base));
    }

    PYBIND11_NOINLINE void value(char const* name_, object value, const char *doc = nullptr) {
        dict entries = m_base.attr("__entries");
        str name(name_);
        if (entries.contains(name)) {
            std::string type_name = (std::string) str(m_base.attr("__name__"));
            throw value_error(type_name + ": element \"" + std::string(name_) + "\" already exists!");
        }

        entries[name] = std::make_pair(value, doc);
        m_base.attr(name) = value;
    }

    PYBIND11_NOINLINE void export_values() {
        dict entries = m_base.attr("__entries");
        for (const auto &kv : entries)
            m_parent.attr(kv.first) = kv.second[int_(0)];
    }

    handle m_base;
    handle m_parent;
};

PYBIND11_NAMESPACE_END(detail)

/// Binds C++ enumerations and enumeration classes to Python
template <typename Type> class enum_ : public class_<Type> {
public:
    using Base = class_<Type>;
    using Base::def;
    using Base::attr;
    using Base::def_property_readonly;
    using Base::def_property_readonly_static;
    using Scalar = typename std::underlying_type<Type>::type;

    template <typename... Extra>
    enum_(const handle &scope, const char *name, const Extra&... extra)
      : class_<Type>(scope, name, extra...), m_base(*this, scope) {
        constexpr bool is_arithmetic = detail::any_of<std::is_same<arithmetic, Extra>...>::value;
        constexpr bool is_convertible = std::is_convertible<Type, Scalar>::value;
        m_base.init(is_arithmetic, is_convertible);

        def(init([](Scalar i) { return static_cast<Type>(i); }));
        def("__int__", [](Type value) { return (Scalar) value; });
        #if PY_MAJOR_VERSION < 3
            def("__long__", [](Type value) { return (Scalar) value; });
        #endif
        #if PY_MAJOR_VERSION > 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 8)
            def("__index__", [](Type value) { return (Scalar) value; });
        #endif

        attr("__setstate__") = cpp_function(
            [](detail::value_and_holder &v_h, Scalar arg) {
                detail::initimpl::setstate<Base>(v_h, static_cast<Type>(arg),
                        Py_TYPE(v_h.inst) != v_h.type->type); },
            detail::is_new_style_constructor(),
            pybind11::name("__setstate__"), is_method(*this));
    }

    /// Export enumeration entries into the parent scope
    enum_& export_values() {
        m_base.export_values();
        return *this;
    }

    /// Add an enumeration entry
    enum_& value(char const* name, Type value, const char *doc = nullptr) {
        m_base.value(name, pybind11::cast(value, return_value_policy::copy), doc);
        return *this;
    }

private:
    detail::enum_base m_base;
};

PYBIND11_NAMESPACE_BEGIN(detail)


inline void keep_alive_impl(handle nurse, handle patient) {
    if (!nurse || !patient)
        pybind11_fail("Could not activate keep_alive!");

    if (patient.is_none() || nurse.is_none())
        return; /* Nothing to keep alive or nothing to be kept alive by */

    auto tinfo = all_type_info(Py_TYPE(nurse.ptr()));
    if (!tinfo.empty()) {
        /* It's a pybind-registered type, so we can store the patient in the
         * internal list. */
        add_patient(nurse.ptr(), patient.ptr());
    }
    else {
        /* Fall back to clever approach based on weak references taken from
         * Boost.Python. This is not used for pybind-registered types because
         * the objects can be destroyed out-of-order in a GC pass. */
        cpp_function disable_lifesupport(
            [patient](handle weakref) { patient.dec_ref(); weakref.dec_ref(); });

        weakref wr(nurse, disable_lifesupport);

        patient.inc_ref(); /* reference patient and leak the weak reference */
        (void) wr.release();
    }
}

PYBIND11_NOINLINE inline void keep_alive_impl(size_t Nurse, size_t Patient, function_call &call, handle ret) {
    auto get_arg = [&](size_t n) {
        if (n == 0)
            return ret;
        else if (n == 1 && call.init_self)
            return call.init_self;
        else if (n <= call.args.size())
            return call.args[n - 1];
        return handle();
    };

    keep_alive_impl(get_arg(Nurse), get_arg(Patient));
}

inline std::pair<decltype(internals::registered_types_py)::iterator, bool> all_type_info_get_cache(PyTypeObject *type) {
    auto res = get_internals().registered_types_py
#ifdef __cpp_lib_unordered_map_try_emplace
        .try_emplace(type);
#else
        .emplace(type, std::vector<detail::type_info *>());
#endif
    if (res.second) {
        // New cache entry created; set up a weak reference to automatically remove it if the type
        // gets destroyed:
        weakref((PyObject *) type, cpp_function([type](handle wr) {
            get_internals().registered_types_py.erase(type);
            wr.dec_ref();
        })).release();
    }

    return res;
}

template <typename Iterator, typename Sentinel, bool KeyIterator, return_value_policy Policy>
struct iterator_state {
    Iterator it;
    Sentinel end;
    bool first_or_done;
};

PYBIND11_NAMESPACE_END(detail)

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename ValueType = decltype(*std::declval<Iterator>()),
          typename... Extra>
iterator make_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, false, Policy> state;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator", pybind11::module_local())
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> ValueType {
                if (!s.first_or_done)
                    ++s.it;
                else
                    s.first_or_done = false;
                if (s.it == s.end) {
                    s.first_or_done = true;
                    throw stop_iteration();
                }
                return *s.it;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return cast(state{first, last, true});
}

/// Makes an python iterator over the keys (`.first`) of a iterator over pairs from a
/// first and past-the-end InputIterator.
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Iterator,
          typename Sentinel,
          typename KeyType = decltype((*std::declval<Iterator>()).first),
          typename... Extra>
iterator make_key_iterator(Iterator first, Sentinel last, Extra &&... extra) {
    typedef detail::iterator_state<Iterator, Sentinel, true, Policy> state;

    if (!detail::get_type_info(typeid(state), false)) {
        class_<state>(handle(), "iterator", pybind11::module_local())
            .def("__iter__", [](state &s) -> state& { return s; })
            .def("__next__", [](state &s) -> KeyType {
                if (!s.first_or_done)
                    ++s.it;
                else
                    s.first_or_done = false;
                if (s.it == s.end) {
                    s.first_or_done = true;
                    throw stop_iteration();
                }
                return (*s.it).first;
            }, std::forward<Extra>(extra)..., Policy);
    }

    return cast(state{first, last, true});
}

/// Makes an iterator over values of an stl container or other container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_iterator(Type &value, Extra&&... extra) {
    return make_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

/// Makes an iterator over the keys (`.first`) of a stl map-like container supporting
/// `std::begin()`/`std::end()`
template <return_value_policy Policy = return_value_policy::reference_internal,
          typename Type, typename... Extra> iterator make_key_iterator(Type &value, Extra&&... extra) {
    return make_key_iterator<Policy>(std::begin(value), std::end(value), extra...);
}

template <typename InputType, typename OutputType> void implicitly_convertible() {
    struct set_flag {
        bool &flag;
        set_flag(bool &flag) : flag(flag) { flag = true; }
        ~set_flag() { flag = false; }
    };
    auto implicit_caster = [](PyObject *obj, PyTypeObject *type) -> PyObject * {
        static bool currently_used = false;
        if (currently_used) // implicit conversions are non-reentrant
            return nullptr;
        set_flag flag_helper(currently_used);
        if (!detail::make_caster<InputType>().load(obj, false))
            return nullptr;
        tuple args(1);
        args[0] = obj;
        PyObject *result = PyObject_Call((PyObject *) type, args.ptr(), nullptr);
        if (result == nullptr)
            PyErr_Clear();
        return result;
    };

    if (auto tinfo = detail::get_type_info(typeid(OutputType)))
        tinfo->implicit_conversions.push_back(implicit_caster);
    else
        pybind11_fail("implicitly_convertible: Unable to find type " + type_id<OutputType>());
}

template <typename ExceptionTranslator>
void register_exception_translator(ExceptionTranslator&& translator) {
    detail::get_internals().registered_exception_translators.push_front(
        std::forward<ExceptionTranslator>(translator));
}

/**
 * Wrapper to generate a new Python exception type.
 *
 * This should only be used with PyErr_SetString for now.
 * It is not (yet) possible to use as a py::base.
 * Template type argument is reserved for future use.
 */
template <typename type>
class exception : public object {
public:
    exception() = default;
    exception(handle scope, const char *name, PyObject *base = PyExc_Exception) {
        std::string full_name = scope.attr("__name__").cast<std::string>() +
                                std::string(".") + name;
        m_ptr = PyErr_NewException(const_cast<char *>(full_name.c_str()), base, NULL);
        if (hasattr(scope, name))
            pybind11_fail("Error during initialization: multiple incompatible "
                          "definitions with name \"" + std::string(name) + "\"");
        scope.attr(name) = *this;
    }

    // Sets the current python exception to this exception object with the given message
    void operator()(const char *message) {
        PyErr_SetString(m_ptr, message);
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)
// Returns a reference to a function-local static exception object used in the simple
// register_exception approach below.  (It would be simpler to have the static local variable
// directly in register_exception, but that makes clang <3.5 segfault - issue #1349).
template <typename CppException>
exception<CppException> &get_exception_object() { static exception<CppException> ex; return ex; }
PYBIND11_NAMESPACE_END(detail)

/**
 * Registers a Python exception in `m` of the given `name` and installs an exception translator to
 * translate the C++ exception to the created Python exception using the exceptions what() method.
 * This is intended for simple exception translations; for more complex translation, register the
 * exception object and translator directly.
 */
template <typename CppException>
exception<CppException> &register_exception(handle scope,
                                            const char *name,
                                            PyObject *base = PyExc_Exception) {
    auto &ex = detail::get_exception_object<CppException>();
    if (!ex) ex = exception<CppException>(scope, name, base);

    register_exception_translator([](std::exception_ptr p) {
        if (!p) return;
        try {
            std::rethrow_exception(p);
        } catch (const CppException &e) {
            detail::get_exception_object<CppException>()(e.what());
        }
    });
    return ex;
}

PYBIND11_NAMESPACE_BEGIN(detail)
PYBIND11_NOINLINE inline void print(tuple args, dict kwargs) {
    auto strings = tuple(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        strings[i] = str(args[i]);
    }
    auto sep = kwargs.contains("sep") ? kwargs["sep"] : cast(" ");
    auto line = sep.attr("join")(strings);

    object file;
    if (kwargs.contains("file")) {
        file = kwargs["file"].cast<object>();
    } else {
        try {
            file = module::import("sys").attr("stdout");
        } catch (const error_already_set &) {
            /* If print() is called from code that is executed as
               part of garbage collection during interpreter shutdown,
               importing 'sys' can fail. Give up rather than crashing the
               interpreter in this case. */
            return;
        }
    }

    auto write = file.attr("write");
    write(line);
    write(kwargs.contains("end") ? kwargs["end"] : cast("\n"));

    if (kwargs.contains("flush") && kwargs["flush"].cast<bool>())
        file.attr("flush")();
}
PYBIND11_NAMESPACE_END(detail)

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
void print(Args &&...args) {
    auto c = detail::collect_arguments<policy>(std::forward<Args>(args)...);
    detail::print(c.args(), c.kwargs());
}

#if defined(WITH_THREAD) && !defined(PYPY_VERSION)

/* The functions below essentially reproduce the PyGILState_* API using a RAII
 * pattern, but there are a few important differences:
 *
 * 1. When acquiring the GIL from an non-main thread during the finalization
 *    phase, the GILState API blindly terminates the calling thread, which
 *    is often not what is wanted. This API does not do this.
 *
 * 2. The gil_scoped_release function can optionally cut the relationship
 *    of a PyThreadState and its associated thread, which allows moving it to
 *    another thread (this is a fairly rare/advanced use case).
 *
 * 3. The reference count of an acquired thread state can be controlled. This
 *    can be handy to prevent cases where callbacks issued from an external
 *    thread would otherwise constantly construct and destroy thread state data
 *    structures.
 *
 * See the Python bindings of NanoGUI (http://github.com/wjakob/nanogui) for an
 * example which uses features 2 and 3 to migrate the Python thread of
 * execution to another thread (to run the event loop on the original thread,
 * in this case).
 */

class gil_scoped_acquire {
public:
    PYBIND11_NOINLINE gil_scoped_acquire() {
        auto const &internals = detail::get_internals();
        tstate = (PyThreadState *) PYBIND11_TLS_GET_VALUE(internals.tstate);

        if (!tstate) {
            /* Check if the GIL was acquired using the PyGILState_* API instead (e.g. if
               calling from a Python thread). Since we use a different key, this ensures
               we don't create a new thread state and deadlock in PyEval_AcquireThread
               below. Note we don't save this state with internals.tstate, since we don't
               create it we would fail to clear it (its reference count should be > 0). */
            tstate = PyGILState_GetThisThreadState();
        }

        if (!tstate) {
            tstate = PyThreadState_New(internals.istate);
            #if !defined(NDEBUG)
                if (!tstate)
                    pybind11_fail("scoped_acquire: could not create thread state!");
            #endif
            tstate->gilstate_counter = 0;
            PYBIND11_TLS_REPLACE_VALUE(internals.tstate, tstate);
        } else {
            release = detail::get_thread_state_unchecked() != tstate;
        }

        if (release) {
            /* Work around an annoying assertion in PyThreadState_Swap */
            #if defined(Py_DEBUG)
                PyInterpreterState *interp = tstate->interp;
                tstate->interp = nullptr;
            #endif
            PyEval_AcquireThread(tstate);
            #if defined(Py_DEBUG)
                tstate->interp = interp;
            #endif
        }

        inc_ref();
    }

    void inc_ref() {
        ++tstate->gilstate_counter;
    }

    PYBIND11_NOINLINE void dec_ref() {
        --tstate->gilstate_counter;
        #if !defined(NDEBUG)
            if (detail::get_thread_state_unchecked() != tstate)
                pybind11_fail("scoped_acquire::dec_ref(): thread state must be current!");
            if (tstate->gilstate_counter < 0)
                pybind11_fail("scoped_acquire::dec_ref(): reference count underflow!");
        #endif
        if (tstate->gilstate_counter == 0) {
            #if !defined(NDEBUG)
                if (!release)
                    pybind11_fail("scoped_acquire::dec_ref(): internal error!");
            #endif
            PyThreadState_Clear(tstate);
            PyThreadState_DeleteCurrent();
            PYBIND11_TLS_DELETE_VALUE(detail::get_internals().tstate);
            release = false;
        }
    }

    PYBIND11_NOINLINE ~gil_scoped_acquire() {
        dec_ref();
        if (release)
           PyEval_SaveThread();
    }
private:
    PyThreadState *tstate = nullptr;
    bool release = true;
};

class gil_scoped_release {
public:
    explicit gil_scoped_release(bool disassoc = false) : disassoc(disassoc) {
        // `get_internals()` must be called here unconditionally in order to initialize
        // `internals.tstate` for subsequent `gil_scoped_acquire` calls. Otherwise, an
        // initialization race could occur as multiple threads try `gil_scoped_acquire`.
        const auto &internals = detail::get_internals();
        tstate = PyEval_SaveThread();
        if (disassoc) {
            auto key = internals.tstate;
            PYBIND11_TLS_DELETE_VALUE(key);
        }
    }
    ~gil_scoped_release() {
        if (!tstate)
            return;
        PyEval_RestoreThread(tstate);
        if (disassoc) {
            auto key = detail::get_internals().tstate;
            PYBIND11_TLS_REPLACE_VALUE(key, tstate);
        }
    }
private:
    PyThreadState *tstate;
    bool disassoc;
};
#elif defined(PYPY_VERSION)
class gil_scoped_acquire {
    PyGILState_STATE state;
public:
    gil_scoped_acquire() { state = PyGILState_Ensure(); }
    ~gil_scoped_acquire() { PyGILState_Release(state); }
};

class gil_scoped_release {
    PyThreadState *state;
public:
    gil_scoped_release() { state = PyEval_SaveThread(); }
    ~gil_scoped_release() { PyEval_RestoreThread(state); }
};
#else
class gil_scoped_acquire { };
class gil_scoped_release { };
#endif

error_already_set::~error_already_set() {
    if (m_type) {
        gil_scoped_acquire gil;
        error_scope scope;
        m_type.release().dec_ref();
        m_value.release().dec_ref();
        m_trace.release().dec_ref();
    }
}

inline function get_type_overload(const void *this_ptr, const detail::type_info *this_type, const char *name)  {
    handle self = detail::get_object_handle(this_ptr, this_type);
    if (!self)
        return function();
    handle type = self.get_type();
    auto key = std::make_pair(type.ptr(), name);

    /* Cache functions that aren't overloaded in Python to avoid
       many costly Python dictionary lookups below */
    auto &cache = detail::get_internals().inactive_overload_cache;
    if (cache.find(key) != cache.end())
        return function();

    function overload = getattr(self, name, function());
    if (overload.is_cpp_function()) {
        cache.insert(key);
        return function();
    }

    /* Don't call dispatch code if invoked from overridden function.
       Unfortunately this doesn't work on PyPy. */
#if !defined(PYPY_VERSION)
    PyFrameObject *frame = PyThreadState_Get()->frame;
    if (frame && (std::string) str(frame->f_code->co_name) == name &&
        frame->f_code->co_argcount > 0) {
        PyFrame_FastToLocals(frame);
        PyObject *self_caller = PyDict_GetItem(
            frame->f_locals, PyTuple_GET_ITEM(frame->f_code->co_varnames, 0));
        if (self_caller == self.ptr())
            return function();
    }
#else
    /* PyPy currently doesn't provide a detailed cpyext emulation of
       frame objects, so we have to emulate this using Python. This
       is going to be slow..*/
    dict d; d["self"] = self; d["name"] = pybind11::str(name);
    PyObject *result = PyRun_String(
        "import inspect\n"
        "frame = inspect.currentframe()\n"
        "if frame is not None:\n"
        "    frame = frame.f_back\n"
        "    if frame is not None and str(frame.f_code.co_name) == name and "
        "frame.f_code.co_argcount > 0:\n"
        "        self_caller = frame.f_locals[frame.f_code.co_varnames[0]]\n"
        "        if self_caller == self:\n"
        "            self = None\n",
        Py_file_input, d.ptr(), d.ptr());
    if (result == nullptr)
        throw error_already_set();
    if (d["self"].is_none())
        return function();
    Py_DECREF(result);
#endif

    return overload;
}

/** \rst
  Try to retrieve a python method by the provided name from the instance pointed to by the this_ptr.

  :this_ptr: The pointer to the object the overload should be retrieved for. This should be the first
                   non-trampoline class encountered in the inheritance chain.
  :name: The name of the overloaded Python method to retrieve.
  :return: The Python method by this name from the object or an empty function wrapper.
 \endrst */
template <class T> function get_overload(const T *this_ptr, const char *name) {
    auto tinfo = detail::get_type_info(typeid(T));
    return tinfo ? get_type_overload(this_ptr, tinfo, name) : function();
}

#define PYBIND11_OVERLOAD_INT(ret_type, cname, name, ...) { \
        pybind11::gil_scoped_acquire gil; \
        pybind11::function overload = pybind11::get_overload(static_cast<const cname *>(this), name); \
        if (overload) { \
            auto o = overload(__VA_ARGS__); \
            if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) { \
                static pybind11::detail::overload_caster_t<ret_type> caster; \
                return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
            } \
            else return pybind11::detail::cast_safe<ret_type>(std::move(o)); \
        } \
    }

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up a method named 'fn'
    from the Python side, deals with the :ref:`gil` and necessary argument conversions to call this method and return
    the appropriate type. See :ref:`overriding_virtuals` for more information. This macro should be used when the method
    name in C is not the same as the method name in Python. For example with `__str__`.

    .. code-block:: cpp

      std::string toString() override {
        PYBIND11_OVERLOAD_NAME(
            std::string, // Return type (ret_type)
            Animal,      // Parent class (cname)
            "__str__",   // Name of method in Python (name)
            toString,    // Name of function in C++ (fn)
        );
      }
\endrst */
#define PYBIND11_OVERLOAD_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__) \
    return cname::fn(__VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERLOAD_NAME`, except that it
    throws if no overload can be found.
\endrst */
#define PYBIND11_OVERLOAD_PURE_NAME(ret_type, cname, name, fn, ...) \
    PYBIND11_OVERLOAD_INT(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), name, __VA_ARGS__) \
    pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(cname) "::" name "\"");

/** \rst
    Macro to populate the virtual method in the trampoline class. This macro tries to look up the method
    from the Python side, deals with the :ref:`gil` and necessary argument conversions to call this method and return
    the appropriate type. This macro should be used if the method name in C and in Python are identical.
    See :ref:`overriding_virtuals` for more information.

    .. code-block:: cpp

      class PyAnimal : public Animal {
      public:
          // Inherit the constructors
          using Animal::Animal;

          // Trampoline (need one for each virtual function)
          std::string go(int n_times) override {
              PYBIND11_OVERLOAD_PURE(
                  std::string, // Return type (ret_type)
                  Animal,      // Parent class (cname)
                  go,          // Name of function in C++ (must match Python name) (fn)
                  n_times      // Argument(s) (...)
              );
          }
      };
\endrst */
#define PYBIND11_OVERLOAD(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

/** \rst
    Macro for pure virtual functions, this function is identical to :c:macro:`PYBIND11_OVERLOAD`, except that it throws
    if no overload can be found.
\endrst */
#define PYBIND11_OVERLOAD_PURE(ret_type, cname, fn, ...) \
    PYBIND11_OVERLOAD_PURE_NAME(PYBIND11_TYPE(ret_type), PYBIND11_TYPE(cname), #fn, fn, __VA_ARGS__)

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#  pragma warning(pop)
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif


//========= end of #include "pybind11.h" ============


//========= begin of #include "chrono.h" ============

/*
    pybind11/chrono.h: Transparent conversion between std::chrono and python's datetime

    Copyright (c) 2016 Trent Houliston <trent@houliston.me> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"
#include <cmath>
#include <ctime>
#include <chrono>
#include <datetime.h>

// Backport the PyDateTime_DELTA functions from Python3.3 if required
#ifndef PyDateTime_DELTA_GET_DAYS
#define PyDateTime_DELTA_GET_DAYS(o)         (((PyDateTime_Delta*)o)->days)
#endif
#ifndef PyDateTime_DELTA_GET_SECONDS
#define PyDateTime_DELTA_GET_SECONDS(o)      (((PyDateTime_Delta*)o)->seconds)
#endif
#ifndef PyDateTime_DELTA_GET_MICROSECONDS
#define PyDateTime_DELTA_GET_MICROSECONDS(o) (((PyDateTime_Delta*)o)->microseconds)
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <typename type> class duration_caster {
public:
    typedef typename type::rep rep;
    typedef typename type::period period;

    typedef std::chrono::duration<uint_fast32_t, std::ratio<86400>> days;

    bool load(handle src, bool) {
        using namespace std::chrono;

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        if (!src) return false;
        // If invoked with datetime.delta object
        if (PyDelta_Check(src.ptr())) {
            value = type(duration_cast<duration<rep, period>>(
                  days(PyDateTime_DELTA_GET_DAYS(src.ptr()))
                + seconds(PyDateTime_DELTA_GET_SECONDS(src.ptr()))
                + microseconds(PyDateTime_DELTA_GET_MICROSECONDS(src.ptr()))));
            return true;
        }
        // If invoked with a float we assume it is seconds and convert
        else if (PyFloat_Check(src.ptr())) {
            value = type(duration_cast<duration<rep, period>>(duration<double>(PyFloat_AsDouble(src.ptr()))));
            return true;
        }
        else return false;
    }

    // If this is a duration just return it back
    static const std::chrono::duration<rep, period>& get_duration(const std::chrono::duration<rep, period> &src) {
        return src;
    }

    // If this is a time_point get the time_since_epoch
    template <typename Clock> static std::chrono::duration<rep, period> get_duration(const std::chrono::time_point<Clock, std::chrono::duration<rep, period>> &src) {
        return src.time_since_epoch();
    }

    static handle cast(const type &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;

        // Use overloaded function to get our duration from our source
        // Works out if it is a duration or time_point and get the duration
        auto d = get_duration(src);

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        using dd_t = duration<int, std::ratio<86400>>;
        using ss_t = duration<int, std::ratio<1>>;
        using us_t = duration<int, std::micro>;

        auto dd = duration_cast<dd_t>(d);
        auto subd = d - dd;
        auto ss = duration_cast<ss_t>(subd);
        auto us = duration_cast<us_t>(subd - ss);
        return PyDelta_FromDSU(dd.count(), ss.count(), us.count());
    }

    PYBIND11_TYPE_CASTER(type, _("datetime.timedelta"));
};

// This is for casting times on the system clock into datetime.datetime instances
template <typename Duration> class type_caster<std::chrono::time_point<std::chrono::system_clock, Duration>> {
public:
    typedef std::chrono::time_point<std::chrono::system_clock, Duration> type;
    bool load(handle src, bool) {
        using namespace std::chrono;

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        if (!src) return false;

        std::tm cal;
        microseconds msecs;

        if (PyDateTime_Check(src.ptr())) {
            cal.tm_sec   = PyDateTime_DATE_GET_SECOND(src.ptr());
            cal.tm_min   = PyDateTime_DATE_GET_MINUTE(src.ptr());
            cal.tm_hour  = PyDateTime_DATE_GET_HOUR(src.ptr());
            cal.tm_mday  = PyDateTime_GET_DAY(src.ptr());
            cal.tm_mon   = PyDateTime_GET_MONTH(src.ptr()) - 1;
            cal.tm_year  = PyDateTime_GET_YEAR(src.ptr()) - 1900;
            cal.tm_isdst = -1;
            msecs        = microseconds(PyDateTime_DATE_GET_MICROSECOND(src.ptr()));
        } else if (PyDate_Check(src.ptr())) {
            cal.tm_sec   = 0;
            cal.tm_min   = 0;
            cal.tm_hour  = 0;
            cal.tm_mday  = PyDateTime_GET_DAY(src.ptr());
            cal.tm_mon   = PyDateTime_GET_MONTH(src.ptr()) - 1;
            cal.tm_year  = PyDateTime_GET_YEAR(src.ptr()) - 1900;
            cal.tm_isdst = -1;
            msecs        = microseconds(0);
        } else if (PyTime_Check(src.ptr())) {
            cal.tm_sec   = PyDateTime_TIME_GET_SECOND(src.ptr());
            cal.tm_min   = PyDateTime_TIME_GET_MINUTE(src.ptr());
            cal.tm_hour  = PyDateTime_TIME_GET_HOUR(src.ptr());
            cal.tm_mday  = 1;   // This date (day, month, year) = (1, 0, 70)
            cal.tm_mon   = 0;   // represents 1-Jan-1970, which is the first
            cal.tm_year  = 70;  // earliest available date for Python's datetime
            cal.tm_isdst = -1;
            msecs        = microseconds(PyDateTime_TIME_GET_MICROSECOND(src.ptr()));
        }
        else return false;

        value = system_clock::from_time_t(std::mktime(&cal)) + msecs;
        return true;
    }

    static handle cast(const std::chrono::time_point<std::chrono::system_clock, Duration> &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        std::time_t tt = system_clock::to_time_t(time_point_cast<system_clock::duration>(src));
        // this function uses static memory so it's best to copy it out asap just in case
        // otherwise other code that is using localtime may break this (not just python code)
        std::tm localtime = *std::localtime(&tt);

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        using us_t = duration<int, std::micro>;

        return PyDateTime_FromDateAndTime(localtime.tm_year + 1900,
                                          localtime.tm_mon + 1,
                                          localtime.tm_mday,
                                          localtime.tm_hour,
                                          localtime.tm_min,
                                          localtime.tm_sec,
                                          (duration_cast<us_t>(src.time_since_epoch() % seconds(1))).count());
    }
    PYBIND11_TYPE_CASTER(type, _("datetime.datetime"));
};

// Other clocks that are not the system clock are not measured as datetime.datetime objects
// since they are not measured on calendar time. So instead we just make them timedeltas
// Or if they have passed us a time as a float we convert that
template <typename Clock, typename Duration> class type_caster<std::chrono::time_point<Clock, Duration>>
: public duration_caster<std::chrono::time_point<Clock, Duration>> {
};

template <typename Rep, typename Period> class type_caster<std::chrono::duration<Rep, Period>>
: public duration_caster<std::chrono::duration<Rep, Period>> {
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "chrono.h" ============


//========= begin of #include "complex.h" ============

/*
    pybind11/complex.h: Complex number support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"
#include <complex>

/// glibc defines I as a macro which breaks things, e.g., boost template names
#ifdef I
#  undef I
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

template <typename T> struct format_descriptor<std::complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr const char c = format_descriptor<T>::c;
    static constexpr const char value[3] = { 'Z', c, '\0' };
    static std::string format() { return std::string(value); }
};

#ifndef PYBIND11_CPP17

template <typename T> constexpr const char format_descriptor<
    std::complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>>::value[3];

#endif

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T> struct is_fmt_numeric<std::complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr bool value = true;
    static constexpr int index = is_fmt_numeric<T>::index + 3;
};

template <typename T> class type_caster<std::complex<T>> {
public:
    bool load(handle src, bool convert) {
        if (!src)
            return false;
        if (!convert && !PyComplex_Check(src.ptr()))
            return false;
        Py_complex result = PyComplex_AsCComplex(src.ptr());
        if (result.real == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        value = std::complex<T>((T) result.real, (T) result.imag);
        return true;
    }

    static handle cast(const std::complex<T> &src, return_value_policy /* policy */, handle /* parent */) {
        return PyComplex_FromDoubles((double) src.real(), (double) src.imag());
    }

    PYBIND11_TYPE_CASTER(std::complex<T>, _("complex"));
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "complex.h" ============


//========= begin of #include "numpy.h" ============

/*
    pybind11/numpy.h: Basic NumPy support, vectorize() wrapper

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"
// ans ignore: #include "complex.h"
#include <numeric>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <functional>
#include <utility>
#include <vector>
#include <typeindex>

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

/* This will be true on all flat address space platforms and allows us to reduce the
   whole npy_intp / ssize_t / Py_intptr_t business down to just ssize_t for all size
   and dimension types (e.g. shape, strides, indexing), instead of inflicting this
   upon the library user. */
static_assert(sizeof(ssize_t) == sizeof(Py_intptr_t), "ssize_t != Py_intptr_t");

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

class array; // Forward declaration

PYBIND11_NAMESPACE_BEGIN(detail)

template <> struct handle_type_name<array> { static constexpr auto name = _("numpy.ndarray"); };

template <typename type, typename SFINAE = void> struct npy_format_descriptor;

struct PyArrayDescr_Proxy {
    PyObject_HEAD
    PyObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    char *subarray;
    PyObject *fields;
    PyObject *names;
};

struct PyArray_Proxy {
    PyObject_HEAD
    char *data;
    int nd;
    ssize_t *dimensions;
    ssize_t *strides;
    PyObject *base;
    PyObject *descr;
    int flags;
};

struct PyVoidScalarObject_Proxy {
    PyObject_VAR_HEAD
    char *obval;
    PyArrayDescr_Proxy *descr;
    int flags;
    PyObject *base;
};

struct numpy_type_info {
    PyObject* dtype_ptr;
    std::string format_str;
};

struct numpy_internals {
    std::unordered_map<std::type_index, numpy_type_info> registered_dtypes;

    numpy_type_info *get_type_info(const std::type_info& tinfo, bool throw_if_missing = true) {
        auto it = registered_dtypes.find(std::type_index(tinfo));
        if (it != registered_dtypes.end())
            return &(it->second);
        if (throw_if_missing)
            pybind11_fail(std::string("NumPy type info missing for ") + tinfo.name());
        return nullptr;
    }

    template<typename T> numpy_type_info *get_type_info(bool throw_if_missing = true) {
        return get_type_info(typeid(typename std::remove_cv<T>::type), throw_if_missing);
    }
};

inline PYBIND11_NOINLINE void load_numpy_internals(numpy_internals* &ptr) {
    ptr = &get_or_create_shared_data<numpy_internals>("_numpy_internals");
}

inline numpy_internals& get_numpy_internals() {
    static numpy_internals* ptr = nullptr;
    if (!ptr)
        load_numpy_internals(ptr);
    return *ptr;
}

template <typename T> struct same_size {
    template <typename U> using as = bool_constant<sizeof(T) == sizeof(U)>;
};

template <typename Concrete> constexpr int platform_lookup() { return -1; }

// Lookup a type according to its size, and return a value corresponding to the NumPy typenum.
template <typename Concrete, typename T, typename... Ts, typename... Ints>
constexpr int platform_lookup(int I, Ints... Is) {
    return sizeof(Concrete) == sizeof(T) ? I : platform_lookup<Concrete, Ts...>(Is...);
}

struct npy_api {
    enum constants {
        NPY_ARRAY_C_CONTIGUOUS_ = 0x0001,
        NPY_ARRAY_F_CONTIGUOUS_ = 0x0002,
        NPY_ARRAY_OWNDATA_ = 0x0004,
        NPY_ARRAY_FORCECAST_ = 0x0010,
        NPY_ARRAY_ENSUREARRAY_ = 0x0040,
        NPY_ARRAY_ALIGNED_ = 0x0100,
        NPY_ARRAY_WRITEABLE_ = 0x0400,
        NPY_BOOL_ = 0,
        NPY_BYTE_, NPY_UBYTE_,
        NPY_SHORT_, NPY_USHORT_,
        NPY_INT_, NPY_UINT_,
        NPY_LONG_, NPY_ULONG_,
        NPY_LONGLONG_, NPY_ULONGLONG_,
        NPY_FLOAT_, NPY_DOUBLE_, NPY_LONGDOUBLE_,
        NPY_CFLOAT_, NPY_CDOUBLE_, NPY_CLONGDOUBLE_,
        NPY_OBJECT_ = 17,
        NPY_STRING_, NPY_UNICODE_, NPY_VOID_,
        // Platform-dependent normalization
        NPY_INT8_ = NPY_BYTE_,
        NPY_UINT8_ = NPY_UBYTE_,
        NPY_INT16_ = NPY_SHORT_,
        NPY_UINT16_ = NPY_USHORT_,
        // `npy_common.h` defines the integer aliases. In order, it checks:
        // NPY_BITSOF_LONG, NPY_BITSOF_LONGLONG, NPY_BITSOF_INT, NPY_BITSOF_SHORT, NPY_BITSOF_CHAR
        // and assigns the alias to the first matching size, so we should check in this order.
        NPY_INT32_ = platform_lookup<std::int32_t, long, int, short>(
            NPY_LONG_, NPY_INT_, NPY_SHORT_),
        NPY_UINT32_ = platform_lookup<std::uint32_t, unsigned long, unsigned int, unsigned short>(
            NPY_ULONG_, NPY_UINT_, NPY_USHORT_),
        NPY_INT64_ = platform_lookup<std::int64_t, long, long long, int>(
            NPY_LONG_, NPY_LONGLONG_, NPY_INT_),
        NPY_UINT64_ = platform_lookup<std::uint64_t, unsigned long, unsigned long long, unsigned int>(
            NPY_ULONG_, NPY_ULONGLONG_, NPY_UINT_),
    };

    typedef struct {
        Py_intptr_t *ptr;
        int len;
    } PyArray_Dims;

    static npy_api& get() {
        static npy_api api = lookup();
        return api;
    }

    bool PyArray_Check_(PyObject *obj) const {
        return (bool) PyObject_TypeCheck(obj, PyArray_Type_);
    }
    bool PyArrayDescr_Check_(PyObject *obj) const {
        return (bool) PyObject_TypeCheck(obj, PyArrayDescr_Type_);
    }

    unsigned int (*PyArray_GetNDArrayCFeatureVersion_)();
    PyObject *(*PyArray_DescrFromType_)(int);
    PyObject *(*PyArray_NewFromDescr_)
        (PyTypeObject *, PyObject *, int, Py_intptr_t const *,
         Py_intptr_t const *, void *, int, PyObject *);
    // Unused. Not removed because that affects ABI of the class.
    PyObject *(*PyArray_DescrNewFromType_)(int);
    int (*PyArray_CopyInto_)(PyObject *, PyObject *);
    PyObject *(*PyArray_NewCopy_)(PyObject *, int);
    PyTypeObject *PyArray_Type_;
    PyTypeObject *PyVoidArrType_Type_;
    PyTypeObject *PyArrayDescr_Type_;
    PyObject *(*PyArray_DescrFromScalar_)(PyObject *);
    PyObject *(*PyArray_FromAny_) (PyObject *, PyObject *, int, int, int, PyObject *);
    int (*PyArray_DescrConverter_) (PyObject *, PyObject **);
    bool (*PyArray_EquivTypes_) (PyObject *, PyObject *);
    int (*PyArray_GetArrayParamsFromObject_)(PyObject *, PyObject *, unsigned char, PyObject **, int *,
                                             Py_intptr_t *, PyObject **, PyObject *);
    PyObject *(*PyArray_Squeeze_)(PyObject *);
    // Unused. Not removed because that affects ABI of the class.
    int (*PyArray_SetBaseObject_)(PyObject *, PyObject *);
    PyObject* (*PyArray_Resize_)(PyObject*, PyArray_Dims*, int, int);
private:
    enum functions {
        API_PyArray_GetNDArrayCFeatureVersion = 211,
        API_PyArray_Type = 2,
        API_PyArrayDescr_Type = 3,
        API_PyVoidArrType_Type = 39,
        API_PyArray_DescrFromType = 45,
        API_PyArray_DescrFromScalar = 57,
        API_PyArray_FromAny = 69,
        API_PyArray_Resize = 80,
        API_PyArray_CopyInto = 82,
        API_PyArray_NewCopy = 85,
        API_PyArray_NewFromDescr = 94,
        API_PyArray_DescrNewFromType = 96,
        API_PyArray_DescrConverter = 174,
        API_PyArray_EquivTypes = 182,
        API_PyArray_GetArrayParamsFromObject = 278,
        API_PyArray_Squeeze = 136,
        API_PyArray_SetBaseObject = 282
    };

    static npy_api lookup() {
        module m = module::import("numpy.core.multiarray");
        auto c = m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
        void **api_ptr = (void **) PyCapsule_GetPointer(c.ptr(), NULL);
#else
        void **api_ptr = (void **) PyCObject_AsVoidPtr(c.ptr());
#endif
        npy_api api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_)) api_ptr[API_##Func];
        DECL_NPY_API(PyArray_GetNDArrayCFeatureVersion);
        if (api.PyArray_GetNDArrayCFeatureVersion_() < 0x7)
            pybind11_fail("pybind11 numpy support requires numpy >= 1.7.0");
        DECL_NPY_API(PyArray_Type);
        DECL_NPY_API(PyVoidArrType_Type);
        DECL_NPY_API(PyArrayDescr_Type);
        DECL_NPY_API(PyArray_DescrFromType);
        DECL_NPY_API(PyArray_DescrFromScalar);
        DECL_NPY_API(PyArray_FromAny);
        DECL_NPY_API(PyArray_Resize);
        DECL_NPY_API(PyArray_CopyInto);
        DECL_NPY_API(PyArray_NewCopy);
        DECL_NPY_API(PyArray_NewFromDescr);
        DECL_NPY_API(PyArray_DescrNewFromType);
        DECL_NPY_API(PyArray_DescrConverter);
        DECL_NPY_API(PyArray_EquivTypes);
        DECL_NPY_API(PyArray_GetArrayParamsFromObject);
        DECL_NPY_API(PyArray_Squeeze);
        DECL_NPY_API(PyArray_SetBaseObject);
#undef DECL_NPY_API
        return api;
    }
};

inline PyArray_Proxy* array_proxy(void* ptr) {
    return reinterpret_cast<PyArray_Proxy*>(ptr);
}

inline const PyArray_Proxy* array_proxy(const void* ptr) {
    return reinterpret_cast<const PyArray_Proxy*>(ptr);
}

inline PyArrayDescr_Proxy* array_descriptor_proxy(PyObject* ptr) {
   return reinterpret_cast<PyArrayDescr_Proxy*>(ptr);
}

inline const PyArrayDescr_Proxy* array_descriptor_proxy(const PyObject* ptr) {
   return reinterpret_cast<const PyArrayDescr_Proxy*>(ptr);
}

inline bool check_flags(const void* ptr, int flag) {
    return (flag == (array_proxy(ptr)->flags & flag));
}

template <typename T> struct is_std_array : std::false_type { };
template <typename T, size_t N> struct is_std_array<std::array<T, N>> : std::true_type { };
template <typename T> struct is_complex : std::false_type { };
template <typename T> struct is_complex<std::complex<T>> : std::true_type { };

template <typename T> struct array_info_scalar {
    typedef T type;
    static constexpr bool is_array = false;
    static constexpr bool is_empty = false;
    static constexpr auto extents = _("");
    static void append_extents(list& /* shape */) { }
};
// Computes underlying type and a comma-separated list of extents for array
// types (any mix of std::array and built-in arrays). An array of char is
// treated as scalar because it gets special handling.
template <typename T> struct array_info : array_info_scalar<T> { };
template <typename T, size_t N> struct array_info<std::array<T, N>> {
    using type = typename array_info<T>::type;
    static constexpr bool is_array = true;
    static constexpr bool is_empty = (N == 0) || array_info<T>::is_empty;
    static constexpr size_t extent = N;

    // appends the extents to shape
    static void append_extents(list& shape) {
        shape.append(N);
        array_info<T>::append_extents(shape);
    }

    static constexpr auto extents = _<array_info<T>::is_array>(
        concat(_<N>(), array_info<T>::extents), _<N>()
    );
};
// For numpy we have special handling for arrays of characters, so we don't include
// the size in the array extents.
template <size_t N> struct array_info<char[N]> : array_info_scalar<char[N]> { };
template <size_t N> struct array_info<std::array<char, N>> : array_info_scalar<std::array<char, N>> { };
template <typename T, size_t N> struct array_info<T[N]> : array_info<std::array<T, N>> { };
template <typename T> using remove_all_extents_t = typename array_info<T>::type;

template <typename T> using is_pod_struct = all_of<
    std::is_standard_layout<T>,     // since we're accessing directly in memory we need a standard layout type
#if !defined(__GNUG__) || defined(_LIBCPP_VERSION) || defined(_GLIBCXX_USE_CXX11_ABI)
    // _GLIBCXX_USE_CXX11_ABI indicates that we're using libstdc++ from GCC 5 or newer, independent
    // of the actual compiler (Clang can also use libstdc++, but it always defines __GNUC__ == 4).
    std::is_trivially_copyable<T>,
#else
    // GCC 4 doesn't implement is_trivially_copyable, so approximate it
    std::is_trivially_destructible<T>,
    satisfies_any_of<T, std::has_trivial_copy_constructor, std::has_trivial_copy_assign>,
#endif
    satisfies_none_of<T, std::is_reference, std::is_array, is_std_array, std::is_arithmetic, is_complex, std::is_enum>
>;

template <ssize_t Dim = 0, typename Strides> ssize_t byte_offset_unsafe(const Strides &) { return 0; }
template <ssize_t Dim = 0, typename Strides, typename... Ix>
ssize_t byte_offset_unsafe(const Strides &strides, ssize_t i, Ix... index) {
    return i * strides[Dim] + byte_offset_unsafe<Dim + 1>(strides, index...);
}

/**
 * Proxy class providing unsafe, unchecked const access to array data.  This is constructed through
 * the `unchecked<T, N>()` method of `array` or the `unchecked<N>()` method of `array_t<T>`.  `Dims`
 * will be -1 for dimensions determined at runtime.
 */
template <typename T, ssize_t Dims>
class unchecked_reference {
protected:
    static constexpr bool Dynamic = Dims < 0;
    const unsigned char *data_;
    // Storing the shape & strides in local variables (i.e. these arrays) allows the compiler to
    // make large performance gains on big, nested loops, but requires compile-time dimensions
    conditional_t<Dynamic, const ssize_t *, std::array<ssize_t, (size_t) Dims>>
            shape_, strides_;
    const ssize_t dims_;

    friend class pybind11::array;
    // Constructor for compile-time dimensions:
    template <bool Dyn = Dynamic>
    unchecked_reference(const void *data, const ssize_t *shape, const ssize_t *strides, enable_if_t<!Dyn, ssize_t>)
    : data_{reinterpret_cast<const unsigned char *>(data)}, dims_{Dims} {
        for (size_t i = 0; i < (size_t) dims_; i++) {
            shape_[i] = shape[i];
            strides_[i] = strides[i];
        }
    }
    // Constructor for runtime dimensions:
    template <bool Dyn = Dynamic>
    unchecked_reference(const void *data, const ssize_t *shape, const ssize_t *strides, enable_if_t<Dyn, ssize_t> dims)
    : data_{reinterpret_cast<const unsigned char *>(data)}, shape_{shape}, strides_{strides}, dims_{dims} {}

public:
    /**
     * Unchecked const reference access to data at the given indices.  For a compile-time known
     * number of dimensions, this requires the correct number of arguments; for run-time
     * dimensionality, this is not checked (and so is up to the caller to use safely).
     */
    template <typename... Ix> const T &operator()(Ix... index) const {
        static_assert(ssize_t{sizeof...(Ix)} == Dims || Dynamic,
                "Invalid number of indices for unchecked array reference");
        return *reinterpret_cast<const T *>(data_ + byte_offset_unsafe(strides_, ssize_t(index)...));
    }
    /**
     * Unchecked const reference access to data; this operator only participates if the reference
     * is to a 1-dimensional array.  When present, this is exactly equivalent to `obj(index)`.
     */
    template <ssize_t D = Dims, typename = enable_if_t<D == 1 || Dynamic>>
    const T &operator[](ssize_t index) const { return operator()(index); }

    /// Pointer access to the data at the given indices.
    template <typename... Ix> const T *data(Ix... ix) const { return &operator()(ssize_t(ix)...); }

    /// Returns the item size, i.e. sizeof(T)
    constexpr static ssize_t itemsize() { return sizeof(T); }

    /// Returns the shape (i.e. size) of dimension `dim`
    ssize_t shape(ssize_t dim) const { return shape_[(size_t) dim]; }

    /// Returns the number of dimensions of the array
    ssize_t ndim() const { return dims_; }

    /// Returns the total number of elements in the referenced array, i.e. the product of the shapes
    template <bool Dyn = Dynamic>
    enable_if_t<!Dyn, ssize_t> size() const {
        return std::accumulate(shape_.begin(), shape_.end(), (ssize_t) 1, std::multiplies<ssize_t>());
    }
    template <bool Dyn = Dynamic>
    enable_if_t<Dyn, ssize_t> size() const {
        return std::accumulate(shape_, shape_ + ndim(), (ssize_t) 1, std::multiplies<ssize_t>());
    }

    /// Returns the total number of bytes used by the referenced data.  Note that the actual span in
    /// memory may be larger if the referenced array has non-contiguous strides (e.g. for a slice).
    ssize_t nbytes() const {
        return size() * itemsize();
    }
};

template <typename T, ssize_t Dims>
class unchecked_mutable_reference : public unchecked_reference<T, Dims> {
    friend class pybind11::array;
    using ConstBase = unchecked_reference<T, Dims>;
    using ConstBase::ConstBase;
    using ConstBase::Dynamic;
public:
    /// Mutable, unchecked access to data at the given indices.
    template <typename... Ix> T& operator()(Ix... index) {
        static_assert(ssize_t{sizeof...(Ix)} == Dims || Dynamic,
                "Invalid number of indices for unchecked array reference");
        return const_cast<T &>(ConstBase::operator()(index...));
    }
    /**
     * Mutable, unchecked access data at the given index; this operator only participates if the
     * reference is to a 1-dimensional array (or has runtime dimensions).  When present, this is
     * exactly equivalent to `obj(index)`.
     */
    template <ssize_t D = Dims, typename = enable_if_t<D == 1 || Dynamic>>
    T &operator[](ssize_t index) { return operator()(index); }

    /// Mutable pointer access to the data at the given indices.
    template <typename... Ix> T *mutable_data(Ix... ix) { return &operator()(ssize_t(ix)...); }
};

template <typename T, ssize_t Dim>
struct type_caster<unchecked_reference<T, Dim>> {
    static_assert(Dim == 0 && Dim > 0 /* always fail */, "unchecked array proxy object is not castable");
};
template <typename T, ssize_t Dim>
struct type_caster<unchecked_mutable_reference<T, Dim>> : type_caster<unchecked_reference<T, Dim>> {};

PYBIND11_NAMESPACE_END(detail)

class dtype : public object {
public:
    PYBIND11_OBJECT_DEFAULT(dtype, object, detail::npy_api::get().PyArrayDescr_Check_);

    explicit dtype(const buffer_info &info) {
        dtype descr(_dtype_from_pep3118()(PYBIND11_STR_TYPE(info.format)));
        // If info.itemsize == 0, use the value calculated from the format string
        m_ptr = descr.strip_padding(info.itemsize ? info.itemsize : descr.itemsize()).release().ptr();
    }

    explicit dtype(const std::string &format) {
        m_ptr = from_args(pybind11::str(format)).release().ptr();
    }

    dtype(const char *format) : dtype(std::string(format)) { }

    dtype(list names, list formats, list offsets, ssize_t itemsize) {
        dict args;
        args["names"] = names;
        args["formats"] = formats;
        args["offsets"] = offsets;
        args["itemsize"] = pybind11::int_(itemsize);
        m_ptr = from_args(args).release().ptr();
    }

    /// This is essentially the same as calling numpy.dtype(args) in Python.
    static dtype from_args(object args) {
        PyObject *ptr = nullptr;
        if (!detail::npy_api::get().PyArray_DescrConverter_(args.ptr(), &ptr) || !ptr)
            throw error_already_set();
        return reinterpret_steal<dtype>(ptr);
    }

    /// Return dtype associated with a C++ type.
    template <typename T> static dtype of() {
        return detail::npy_format_descriptor<typename std::remove_cv<T>::type>::dtype();
    }

    /// Size of the data type in bytes.
    ssize_t itemsize() const {
        return detail::array_descriptor_proxy(m_ptr)->elsize;
    }

    /// Returns true for structured data types.
    bool has_fields() const {
        return detail::array_descriptor_proxy(m_ptr)->names != nullptr;
    }

    /// Single-character type code.
    char kind() const {
        return detail::array_descriptor_proxy(m_ptr)->kind;
    }

private:
    static object _dtype_from_pep3118() {
        static PyObject *obj = module::import("numpy.core._internal")
            .attr("_dtype_from_pep3118").cast<object>().release().ptr();
        return reinterpret_borrow<object>(obj);
    }

    dtype strip_padding(ssize_t itemsize) {
        // Recursively strip all void fields with empty names that are generated for
        // padding fields (as of NumPy v1.11).
        if (!has_fields())
            return *this;

        struct field_descr { PYBIND11_STR_TYPE name; object format; pybind11::int_ offset; };
        std::vector<field_descr> field_descriptors;

        for (auto field : attr("fields").attr("items")()) {
            auto spec = field.cast<tuple>();
            auto name = spec[0].cast<pybind11::str>();
            auto format = spec[1].cast<tuple>()[0].cast<dtype>();
            auto offset = spec[1].cast<tuple>()[1].cast<pybind11::int_>();
            if (!len(name) && format.kind() == 'V')
                continue;
            field_descriptors.push_back({(PYBIND11_STR_TYPE) name, format.strip_padding(format.itemsize()), offset});
        }

        std::sort(field_descriptors.begin(), field_descriptors.end(),
                  [](const field_descr& a, const field_descr& b) {
                      return a.offset.cast<int>() < b.offset.cast<int>();
                  });

        list names, formats, offsets;
        for (auto& descr : field_descriptors) {
            names.append(descr.name);
            formats.append(descr.format);
            offsets.append(descr.offset);
        }
        return dtype(names, formats, offsets, itemsize);
    }
};

class array : public buffer {
public:
    PYBIND11_OBJECT_CVT(array, buffer, detail::npy_api::get().PyArray_Check_, raw_array)

    enum {
        c_style = detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_,
        f_style = detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_,
        forcecast = detail::npy_api::NPY_ARRAY_FORCECAST_
    };

    array() : array(0, static_cast<const double *>(nullptr)) {}

    using ShapeContainer = detail::any_container<ssize_t>;
    using StridesContainer = detail::any_container<ssize_t>;

    // Constructs an array taking shape/strides from arbitrary container types
    array(const pybind11::dtype &dt, ShapeContainer shape, StridesContainer strides,
          const void *ptr = nullptr, handle base = handle()) {

        if (strides->empty())
            *strides = c_strides(*shape, dt.itemsize());

        auto ndim = shape->size();
        if (ndim != strides->size())
            pybind11_fail("NumPy: shape ndim doesn't match strides ndim");
        auto descr = dt;

        int flags = 0;
        if (base && ptr) {
            if (isinstance<array>(base))
                /* Copy flags from base (except ownership bit) */
                flags = reinterpret_borrow<array>(base).flags() & ~detail::npy_api::NPY_ARRAY_OWNDATA_;
            else
                /* Writable by default, easy to downgrade later on if needed */
                flags = detail::npy_api::NPY_ARRAY_WRITEABLE_;
        }

        auto &api = detail::npy_api::get();
        auto tmp = reinterpret_steal<object>(api.PyArray_NewFromDescr_(
            api.PyArray_Type_, descr.release().ptr(), (int) ndim, shape->data(), strides->data(),
            const_cast<void *>(ptr), flags, nullptr));
        if (!tmp)
            throw error_already_set();
        if (ptr) {
            if (base) {
                api.PyArray_SetBaseObject_(tmp.ptr(), base.inc_ref().ptr());
            } else {
                tmp = reinterpret_steal<object>(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */));
            }
        }
        m_ptr = tmp.release().ptr();
    }

    array(const pybind11::dtype &dt, ShapeContainer shape, const void *ptr = nullptr, handle base = handle())
        : array(dt, std::move(shape), {}, ptr, base) { }

    template <typename T, typename = detail::enable_if_t<std::is_integral<T>::value && !std::is_same<bool, T>::value>>
    array(const pybind11::dtype &dt, T count, const void *ptr = nullptr, handle base = handle())
        : array(dt, {{count}}, ptr, base) { }

    template <typename T>
    array(ShapeContainer shape, StridesContainer strides, const T *ptr, handle base = handle())
        : array(pybind11::dtype::of<T>(), std::move(shape), std::move(strides), ptr, base) { }

    template <typename T>
    array(ShapeContainer shape, const T *ptr, handle base = handle())
        : array(std::move(shape), {}, ptr, base) { }

    template <typename T>
    explicit array(ssize_t count, const T *ptr, handle base = handle()) : array({count}, {}, ptr, base) { }

    explicit array(const buffer_info &info, handle base = handle())
    : array(pybind11::dtype(info), info.shape, info.strides, info.ptr, base) { }

    /// Array descriptor (dtype)
    pybind11::dtype dtype() const {
        return reinterpret_borrow<pybind11::dtype>(detail::array_proxy(m_ptr)->descr);
    }

    /// Total number of elements
    ssize_t size() const {
        return std::accumulate(shape(), shape() + ndim(), (ssize_t) 1, std::multiplies<ssize_t>());
    }

    /// Byte size of a single element
    ssize_t itemsize() const {
        return detail::array_descriptor_proxy(detail::array_proxy(m_ptr)->descr)->elsize;
    }

    /// Total number of bytes
    ssize_t nbytes() const {
        return size() * itemsize();
    }

    /// Number of dimensions
    ssize_t ndim() const {
        return detail::array_proxy(m_ptr)->nd;
    }

    /// Base object
    object base() const {
        return reinterpret_borrow<object>(detail::array_proxy(m_ptr)->base);
    }

    /// Dimensions of the array
    const ssize_t* shape() const {
        return detail::array_proxy(m_ptr)->dimensions;
    }

    /// Dimension along a given axis
    ssize_t shape(ssize_t dim) const {
        if (dim >= ndim())
            fail_dim_check(dim, "invalid axis");
        return shape()[dim];
    }

    /// Strides of the array
    const ssize_t* strides() const {
        return detail::array_proxy(m_ptr)->strides;
    }

    /// Stride along a given axis
    ssize_t strides(ssize_t dim) const {
        if (dim >= ndim())
            fail_dim_check(dim, "invalid axis");
        return strides()[dim];
    }

    /// Return the NumPy array flags
    int flags() const {
        return detail::array_proxy(m_ptr)->flags;
    }

    /// If set, the array is writeable (otherwise the buffer is read-only)
    bool writeable() const {
        return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_WRITEABLE_);
    }

    /// If set, the array owns the data (will be freed when the array is deleted)
    bool owndata() const {
        return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_OWNDATA_);
    }

    /// Pointer to the contained data. If index is not provided, points to the
    /// beginning of the buffer. May throw if the index would lead to out of bounds access.
    template<typename... Ix> const void* data(Ix... index) const {
        return static_cast<const void *>(detail::array_proxy(m_ptr)->data + offset_at(index...));
    }

    /// Mutable pointer to the contained data. If index is not provided, points to the
    /// beginning of the buffer. May throw if the index would lead to out of bounds access.
    /// May throw if the array is not writeable.
    template<typename... Ix> void* mutable_data(Ix... index) {
        check_writeable();
        return static_cast<void *>(detail::array_proxy(m_ptr)->data + offset_at(index...));
    }

    /// Byte offset from beginning of the array to a given index (full or partial).
    /// May throw if the index would lead to out of bounds access.
    template<typename... Ix> ssize_t offset_at(Ix... index) const {
        if ((ssize_t) sizeof...(index) > ndim())
            fail_dim_check(sizeof...(index), "too many indices for an array");
        return byte_offset(ssize_t(index)...);
    }

    ssize_t offset_at() const { return 0; }

    /// Item count from beginning of the array to a given index (full or partial).
    /// May throw if the index would lead to out of bounds access.
    template<typename... Ix> ssize_t index_at(Ix... index) const {
        return offset_at(index...) / itemsize();
    }

    /**
     * Returns a proxy object that provides access to the array's data without bounds or
     * dimensionality checking.  Will throw if the array is missing the `writeable` flag.  Use with
     * care: the array must not be destroyed or reshaped for the duration of the returned object,
     * and the caller must take care not to access invalid dimensions or dimension indices.
     */
    template <typename T, ssize_t Dims = -1> detail::unchecked_mutable_reference<T, Dims> mutable_unchecked() & {
        if (Dims >= 0 && ndim() != Dims)
            throw std::domain_error("array has incorrect number of dimensions: " + std::to_string(ndim()) +
                    "; expected " + std::to_string(Dims));
        return detail::unchecked_mutable_reference<T, Dims>(mutable_data(), shape(), strides(), ndim());
    }

    /**
     * Returns a proxy object that provides const access to the array's data without bounds or
     * dimensionality checking.  Unlike `mutable_unchecked()`, this does not require that the
     * underlying array have the `writable` flag.  Use with care: the array must not be destroyed or
     * reshaped for the duration of the returned object, and the caller must take care not to access
     * invalid dimensions or dimension indices.
     */
    template <typename T, ssize_t Dims = -1> detail::unchecked_reference<T, Dims> unchecked() const & {
        if (Dims >= 0 && ndim() != Dims)
            throw std::domain_error("array has incorrect number of dimensions: " + std::to_string(ndim()) +
                    "; expected " + std::to_string(Dims));
        return detail::unchecked_reference<T, Dims>(data(), shape(), strides(), ndim());
    }

    /// Return a new view with all of the dimensions of length 1 removed
    array squeeze() {
        auto& api = detail::npy_api::get();
        return reinterpret_steal<array>(api.PyArray_Squeeze_(m_ptr));
    }

    /// Resize array to given shape
    /// If refcheck is true and more that one reference exist to this array
    /// then resize will succeed only if it makes a reshape, i.e. original size doesn't change
    void resize(ShapeContainer new_shape, bool refcheck = true) {
        detail::npy_api::PyArray_Dims d = {
            new_shape->data(), int(new_shape->size())
        };
        // try to resize, set ordering param to -1 cause it's not used anyway
        object new_array = reinterpret_steal<object>(
            detail::npy_api::get().PyArray_Resize_(m_ptr, &d, int(refcheck), -1)
        );
        if (!new_array) throw error_already_set();
        if (isinstance<array>(new_array)) { *this = std::move(new_array); }
    }

    /// Ensure that the argument is a NumPy array
    /// In case of an error, nullptr is returned and the Python error is cleared.
    static array ensure(handle h, int ExtraFlags = 0) {
        auto result = reinterpret_steal<array>(raw_array(h.ptr(), ExtraFlags));
        if (!result)
            PyErr_Clear();
        return result;
    }

protected:
    template<typename, typename> friend struct detail::npy_format_descriptor;

    void fail_dim_check(ssize_t dim, const std::string& msg) const {
        throw index_error(msg + ": " + std::to_string(dim) +
                          " (ndim = " + std::to_string(ndim()) + ")");
    }

    template<typename... Ix> ssize_t byte_offset(Ix... index) const {
        check_dimensions(index...);
        return detail::byte_offset_unsafe(strides(), ssize_t(index)...);
    }

    void check_writeable() const {
        if (!writeable())
            throw std::domain_error("array is not writeable");
    }

    // Default, C-style strides
    static std::vector<ssize_t> c_strides(const std::vector<ssize_t> &shape, ssize_t itemsize) {
        auto ndim = shape.size();
        std::vector<ssize_t> strides(ndim, itemsize);
        if (ndim > 0)
            for (size_t i = ndim - 1; i > 0; --i)
                strides[i - 1] = strides[i] * shape[i];
        return strides;
    }

    // F-style strides; default when constructing an array_t with `ExtraFlags & f_style`
    static std::vector<ssize_t> f_strides(const std::vector<ssize_t> &shape, ssize_t itemsize) {
        auto ndim = shape.size();
        std::vector<ssize_t> strides(ndim, itemsize);
        for (size_t i = 1; i < ndim; ++i)
            strides[i] = strides[i - 1] * shape[i - 1];
        return strides;
    }

    template<typename... Ix> void check_dimensions(Ix... index) const {
        check_dimensions_impl(ssize_t(0), shape(), ssize_t(index)...);
    }

    void check_dimensions_impl(ssize_t, const ssize_t*) const { }

    template<typename... Ix> void check_dimensions_impl(ssize_t axis, const ssize_t* shape, ssize_t i, Ix... index) const {
        if (i >= *shape) {
            throw index_error(std::string("index ") + std::to_string(i) +
                              " is out of bounds for axis " + std::to_string(axis) +
                              " with size " + std::to_string(*shape));
        }
        check_dimensions_impl(axis + 1, shape + 1, index...);
    }

    /// Create array from any object -- always returns a new reference
    static PyObject *raw_array(PyObject *ptr, int ExtraFlags = 0) {
        if (ptr == nullptr) {
            PyErr_SetString(PyExc_ValueError, "cannot create a pybind11::array from a nullptr");
            return nullptr;
        }
        return detail::npy_api::get().PyArray_FromAny_(
            ptr, nullptr, 0, 0, detail::npy_api::NPY_ARRAY_ENSUREARRAY_ | ExtraFlags, nullptr);
    }
};

template <typename T, int ExtraFlags = array::forcecast> class array_t : public array {
private:
    struct private_ctor {};
    // Delegating constructor needed when both moving and accessing in the same constructor
    array_t(private_ctor, ShapeContainer &&shape, StridesContainer &&strides, const T *ptr, handle base)
        : array(std::move(shape), std::move(strides), ptr, base) {}
public:
    static_assert(!detail::array_info<T>::is_array, "Array types cannot be used with array_t");

    using value_type = T;

    array_t() : array(0, static_cast<const T *>(nullptr)) {}
    array_t(handle h, borrowed_t) : array(h, borrowed_t{}) { }
    array_t(handle h, stolen_t) : array(h, stolen_t{}) { }

    PYBIND11_DEPRECATED("Use array_t<T>::ensure() instead")
    array_t(handle h, bool is_borrowed) : array(raw_array_t(h.ptr()), stolen_t{}) {
        if (!m_ptr) PyErr_Clear();
        if (!is_borrowed) Py_XDECREF(h.ptr());
    }

    array_t(const object &o) : array(raw_array_t(o.ptr()), stolen_t{}) {
        if (!m_ptr) throw error_already_set();
    }

    explicit array_t(const buffer_info& info, handle base = handle()) : array(info, base) { }

    array_t(ShapeContainer shape, StridesContainer strides, const T *ptr = nullptr, handle base = handle())
        : array(std::move(shape), std::move(strides), ptr, base) { }

    explicit array_t(ShapeContainer shape, const T *ptr = nullptr, handle base = handle())
        : array_t(private_ctor{}, std::move(shape),
                ExtraFlags & f_style ? f_strides(*shape, itemsize()) : c_strides(*shape, itemsize()),
                ptr, base) { }

    explicit array_t(ssize_t count, const T *ptr = nullptr, handle base = handle())
        : array({count}, {}, ptr, base) { }

    constexpr ssize_t itemsize() const {
        return sizeof(T);
    }

    template<typename... Ix> ssize_t index_at(Ix... index) const {
        return offset_at(index...) / itemsize();
    }

    template<typename... Ix> const T* data(Ix... index) const {
        return static_cast<const T*>(array::data(index...));
    }

    template<typename... Ix> T* mutable_data(Ix... index) {
        return static_cast<T*>(array::mutable_data(index...));
    }

    // Reference to element at a given index
    template<typename... Ix> const T& at(Ix... index) const {
        if ((ssize_t) sizeof...(index) != ndim())
            fail_dim_check(sizeof...(index), "index dimension mismatch");
        return *(static_cast<const T*>(array::data()) + byte_offset(ssize_t(index)...) / itemsize());
    }

    // Mutable reference to element at a given index
    template<typename... Ix> T& mutable_at(Ix... index) {
        if ((ssize_t) sizeof...(index) != ndim())
            fail_dim_check(sizeof...(index), "index dimension mismatch");
        return *(static_cast<T*>(array::mutable_data()) + byte_offset(ssize_t(index)...) / itemsize());
    }

    /**
     * Returns a proxy object that provides access to the array's data without bounds or
     * dimensionality checking.  Will throw if the array is missing the `writeable` flag.  Use with
     * care: the array must not be destroyed or reshaped for the duration of the returned object,
     * and the caller must take care not to access invalid dimensions or dimension indices.
     */
    template <ssize_t Dims = -1> detail::unchecked_mutable_reference<T, Dims> mutable_unchecked() & {
        return array::mutable_unchecked<T, Dims>();
    }

    /**
     * Returns a proxy object that provides const access to the array's data without bounds or
     * dimensionality checking.  Unlike `unchecked()`, this does not require that the underlying
     * array have the `writable` flag.  Use with care: the array must not be destroyed or reshaped
     * for the duration of the returned object, and the caller must take care not to access invalid
     * dimensions or dimension indices.
     */
    template <ssize_t Dims = -1> detail::unchecked_reference<T, Dims> unchecked() const & {
        return array::unchecked<T, Dims>();
    }

    /// Ensure that the argument is a NumPy array of the correct dtype (and if not, try to convert
    /// it).  In case of an error, nullptr is returned and the Python error is cleared.
    static array_t ensure(handle h) {
        auto result = reinterpret_steal<array_t>(raw_array_t(h.ptr()));
        if (!result)
            PyErr_Clear();
        return result;
    }

    static bool check_(handle h) {
        const auto &api = detail::npy_api::get();
        return api.PyArray_Check_(h.ptr())
               && api.PyArray_EquivTypes_(detail::array_proxy(h.ptr())->descr, dtype::of<T>().ptr());
    }

protected:
    /// Create array from any object -- always returns a new reference
    static PyObject *raw_array_t(PyObject *ptr) {
        if (ptr == nullptr) {
            PyErr_SetString(PyExc_ValueError, "cannot create a pybind11::array_t from a nullptr");
            return nullptr;
        }
        return detail::npy_api::get().PyArray_FromAny_(
            ptr, dtype::of<T>().release().ptr(), 0, 0,
            detail::npy_api::NPY_ARRAY_ENSUREARRAY_ | ExtraFlags, nullptr);
    }
};

template <typename T>
struct format_descriptor<T, detail::enable_if_t<detail::is_pod_struct<T>::value>> {
    static std::string format() {
        return detail::npy_format_descriptor<typename std::remove_cv<T>::type>::format();
    }
};

template <size_t N> struct format_descriptor<char[N]> {
    static std::string format() { return std::to_string(N) + "s"; }
};
template <size_t N> struct format_descriptor<std::array<char, N>> {
    static std::string format() { return std::to_string(N) + "s"; }
};

template <typename T>
struct format_descriptor<T, detail::enable_if_t<std::is_enum<T>::value>> {
    static std::string format() {
        return format_descriptor<
            typename std::remove_cv<typename std::underlying_type<T>::type>::type>::format();
    }
};

template <typename T>
struct format_descriptor<T, detail::enable_if_t<detail::array_info<T>::is_array>> {
    static std::string format() {
        using namespace detail;
        static constexpr auto extents = _("(") + array_info<T>::extents + _(")");
        return extents.text + format_descriptor<remove_all_extents_t<T>>::format();
    }
};

PYBIND11_NAMESPACE_BEGIN(detail)
template <typename T, int ExtraFlags>
struct pyobject_caster<array_t<T, ExtraFlags>> {
    using type = array_t<T, ExtraFlags>;

    bool load(handle src, bool convert) {
        if (!convert && !type::check_(src))
            return false;
        value = type::ensure(src);
        return static_cast<bool>(value);
    }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name);
};

template <typename T>
struct compare_buffer_info<T, detail::enable_if_t<detail::is_pod_struct<T>::value>> {
    static bool compare(const buffer_info& b) {
        return npy_api::get().PyArray_EquivTypes_(dtype::of<T>().ptr(), dtype(b).ptr());
    }
};

template <typename T, typename = void>
struct npy_format_descriptor_name;

template <typename T>
struct npy_format_descriptor_name<T, enable_if_t<std::is_integral<T>::value>> {
    static constexpr auto name = _<std::is_same<T, bool>::value>(
        _("bool"), _<std::is_signed<T>::value>("numpy.int", "numpy.uint") + _<sizeof(T)*8>()
    );
};

template <typename T>
struct npy_format_descriptor_name<T, enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr auto name = _<std::is_same<T, float>::value || std::is_same<T, double>::value>(
        _("numpy.float") + _<sizeof(T)*8>(), _("numpy.longdouble")
    );
};

template <typename T>
struct npy_format_descriptor_name<T, enable_if_t<is_complex<T>::value>> {
    static constexpr auto name = _<std::is_same<typename T::value_type, float>::value
                                   || std::is_same<typename T::value_type, double>::value>(
        _("numpy.complex") + _<sizeof(typename T::value_type)*16>(), _("numpy.longcomplex")
    );
};

template <typename T>
struct npy_format_descriptor<T, enable_if_t<satisfies_any_of<T, std::is_arithmetic, is_complex>::value>>
    : npy_format_descriptor_name<T> {
private:
    // NB: the order here must match the one in common.h
    constexpr static const int values[15] = {
        npy_api::NPY_BOOL_,
        npy_api::NPY_BYTE_,   npy_api::NPY_UBYTE_,   npy_api::NPY_INT16_,    npy_api::NPY_UINT16_,
        npy_api::NPY_INT32_,  npy_api::NPY_UINT32_,  npy_api::NPY_INT64_,    npy_api::NPY_UINT64_,
        npy_api::NPY_FLOAT_,  npy_api::NPY_DOUBLE_,  npy_api::NPY_LONGDOUBLE_,
        npy_api::NPY_CFLOAT_, npy_api::NPY_CDOUBLE_, npy_api::NPY_CLONGDOUBLE_
    };

public:
    static constexpr int value = values[detail::is_fmt_numeric<T>::index];

    static pybind11::dtype dtype() {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(value))
            return reinterpret_steal<pybind11::dtype>(ptr);
        pybind11_fail("Unsupported buffer format!");
    }
};

#define PYBIND11_DECL_CHAR_FMT \
    static constexpr auto name = _("S") + _<N>(); \
    static pybind11::dtype dtype() { return pybind11::dtype(std::string("S") + std::to_string(N)); }
template <size_t N> struct npy_format_descriptor<char[N]> { PYBIND11_DECL_CHAR_FMT };
template <size_t N> struct npy_format_descriptor<std::array<char, N>> { PYBIND11_DECL_CHAR_FMT };
#undef PYBIND11_DECL_CHAR_FMT

template<typename T> struct npy_format_descriptor<T, enable_if_t<array_info<T>::is_array>> {
private:
    using base_descr = npy_format_descriptor<typename array_info<T>::type>;
public:
    static_assert(!array_info<T>::is_empty, "Zero-sized arrays are not supported");

    static constexpr auto name = _("(") + array_info<T>::extents + _(")") + base_descr::name;
    static pybind11::dtype dtype() {
        list shape;
        array_info<T>::append_extents(shape);
        return pybind11::dtype::from_args(pybind11::make_tuple(base_descr::dtype(), shape));
    }
};

template<typename T> struct npy_format_descriptor<T, enable_if_t<std::is_enum<T>::value>> {
private:
    using base_descr = npy_format_descriptor<typename std::underlying_type<T>::type>;
public:
    static constexpr auto name = base_descr::name;
    static pybind11::dtype dtype() { return base_descr::dtype(); }
};

struct field_descriptor {
    const char *name;
    ssize_t offset;
    ssize_t size;
    std::string format;
    dtype descr;
};

inline PYBIND11_NOINLINE void register_structured_dtype(
    any_container<field_descriptor> fields,
    const std::type_info& tinfo, ssize_t itemsize,
    bool (*direct_converter)(PyObject *, void *&)) {

    auto& numpy_internals = get_numpy_internals();
    if (numpy_internals.get_type_info(tinfo, false))
        pybind11_fail("NumPy: dtype is already registered");

    // Use ordered fields because order matters as of NumPy 1.14:
    // https://docs.scipy.org/doc/numpy/release.html#multiple-field-indexing-assignment-of-structured-arrays
    std::vector<field_descriptor> ordered_fields(std::move(fields));
    std::sort(ordered_fields.begin(), ordered_fields.end(),
        [](const field_descriptor &a, const field_descriptor &b) { return a.offset < b.offset; });

    list names, formats, offsets;
    for (auto& field : ordered_fields) {
        if (!field.descr)
            pybind11_fail(std::string("NumPy: unsupported field dtype: `") +
                            field.name + "` @ " + tinfo.name());
        names.append(PYBIND11_STR_TYPE(field.name));
        formats.append(field.descr);
        offsets.append(pybind11::int_(field.offset));
    }
    auto dtype_ptr = pybind11::dtype(names, formats, offsets, itemsize).release().ptr();

    // There is an existing bug in NumPy (as of v1.11): trailing bytes are
    // not encoded explicitly into the format string. This will supposedly
    // get fixed in v1.12; for further details, see these:
    // - https://github.com/numpy/numpy/issues/7797
    // - https://github.com/numpy/numpy/pull/7798
    // Because of this, we won't use numpy's logic to generate buffer format
    // strings and will just do it ourselves.
    ssize_t offset = 0;
    std::ostringstream oss;
    // mark the structure as unaligned with '^', because numpy and C++ don't
    // always agree about alignment (particularly for complex), and we're
    // explicitly listing all our padding. This depends on none of the fields
    // overriding the endianness. Putting the ^ in front of individual fields
    // isn't guaranteed to work due to https://github.com/numpy/numpy/issues/9049
    oss << "^T{";
    for (auto& field : ordered_fields) {
        if (field.offset > offset)
            oss << (field.offset - offset) << 'x';
        oss << field.format << ':' << field.name << ':';
        offset = field.offset + field.size;
    }
    if (itemsize > offset)
        oss << (itemsize - offset) << 'x';
    oss << '}';
    auto format_str = oss.str();

    // Sanity check: verify that NumPy properly parses our buffer format string
    auto& api = npy_api::get();
    auto arr =  array(buffer_info(nullptr, itemsize, format_str, 1));
    if (!api.PyArray_EquivTypes_(dtype_ptr, arr.dtype().ptr()))
        pybind11_fail("NumPy: invalid buffer descriptor!");

    auto tindex = std::type_index(tinfo);
    numpy_internals.registered_dtypes[tindex] = { dtype_ptr, format_str };
    get_internals().direct_conversions[tindex].push_back(direct_converter);
}

template <typename T, typename SFINAE> struct npy_format_descriptor {
    static_assert(is_pod_struct<T>::value, "Attempt to use a non-POD or unimplemented POD type as a numpy dtype");

    static constexpr auto name = make_caster<T>::name;

    static pybind11::dtype dtype() {
        return reinterpret_borrow<pybind11::dtype>(dtype_ptr());
    }

    static std::string format() {
        static auto format_str = get_numpy_internals().get_type_info<T>(true)->format_str;
        return format_str;
    }

    static void register_dtype(any_container<field_descriptor> fields) {
        register_structured_dtype(std::move(fields), typeid(typename std::remove_cv<T>::type),
                                  sizeof(T), &direct_converter);
    }

private:
    static PyObject* dtype_ptr() {
        static PyObject* ptr = get_numpy_internals().get_type_info<T>(true)->dtype_ptr;
        return ptr;
    }

    static bool direct_converter(PyObject *obj, void*& value) {
        auto& api = npy_api::get();
        if (!PyObject_TypeCheck(obj, api.PyVoidArrType_Type_))
            return false;
        if (auto descr = reinterpret_steal<object>(api.PyArray_DescrFromScalar_(obj))) {
            if (api.PyArray_EquivTypes_(dtype_ptr(), descr.ptr())) {
                value = ((PyVoidScalarObject_Proxy *) obj)->obval;
                return true;
            }
        }
        return false;
    }
};

#ifdef __CLION_IDE__ // replace heavy macro with dummy code for the IDE (doesn't affect code)
# define PYBIND11_NUMPY_DTYPE(Type, ...) ((void)0)
# define PYBIND11_NUMPY_DTYPE_EX(Type, ...) ((void)0)
#else

#define PYBIND11_FIELD_DESCRIPTOR_EX(T, Field, Name)                                          \
    ::pybind11::detail::field_descriptor {                                                    \
        Name, offsetof(T, Field), sizeof(decltype(std::declval<T>().Field)),                  \
        ::pybind11::format_descriptor<decltype(std::declval<T>().Field)>::format(),           \
        ::pybind11::detail::npy_format_descriptor<decltype(std::declval<T>().Field)>::dtype() \
    }

// Extract name, offset and format descriptor for a struct field
#define PYBIND11_FIELD_DESCRIPTOR(T, Field) PYBIND11_FIELD_DESCRIPTOR_EX(T, Field, #Field)

// The main idea of this macro is borrowed from https://github.com/swansontec/map-macro
// (C) William Swanson, Paul Fultz
#define PYBIND11_EVAL0(...) __VA_ARGS__
#define PYBIND11_EVAL1(...) PYBIND11_EVAL0 (PYBIND11_EVAL0 (PYBIND11_EVAL0 (__VA_ARGS__)))
#define PYBIND11_EVAL2(...) PYBIND11_EVAL1 (PYBIND11_EVAL1 (PYBIND11_EVAL1 (__VA_ARGS__)))
#define PYBIND11_EVAL3(...) PYBIND11_EVAL2 (PYBIND11_EVAL2 (PYBIND11_EVAL2 (__VA_ARGS__)))
#define PYBIND11_EVAL4(...) PYBIND11_EVAL3 (PYBIND11_EVAL3 (PYBIND11_EVAL3 (__VA_ARGS__)))
#define PYBIND11_EVAL(...)  PYBIND11_EVAL4 (PYBIND11_EVAL4 (PYBIND11_EVAL4 (__VA_ARGS__)))
#define PYBIND11_MAP_END(...)
#define PYBIND11_MAP_OUT
#define PYBIND11_MAP_COMMA ,
#define PYBIND11_MAP_GET_END() 0, PYBIND11_MAP_END
#define PYBIND11_MAP_NEXT0(test, next, ...) next PYBIND11_MAP_OUT
#define PYBIND11_MAP_NEXT1(test, next) PYBIND11_MAP_NEXT0 (test, next, 0)
#define PYBIND11_MAP_NEXT(test, next)  PYBIND11_MAP_NEXT1 (PYBIND11_MAP_GET_END test, next)
#if defined(_MSC_VER) && !defined(__clang__) // MSVC is not as eager to expand macros, hence this workaround
#define PYBIND11_MAP_LIST_NEXT1(test, next) \
    PYBIND11_EVAL0 (PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0))
#else
#define PYBIND11_MAP_LIST_NEXT1(test, next) \
    PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0)
#endif
#define PYBIND11_MAP_LIST_NEXT(test, next) \
    PYBIND11_MAP_LIST_NEXT1 (PYBIND11_MAP_GET_END test, next)
#define PYBIND11_MAP_LIST0(f, t, x, peek, ...) \
    f(t, x) PYBIND11_MAP_LIST_NEXT (peek, PYBIND11_MAP_LIST1) (f, t, peek, __VA_ARGS__)
#define PYBIND11_MAP_LIST1(f, t, x, peek, ...) \
    f(t, x) PYBIND11_MAP_LIST_NEXT (peek, PYBIND11_MAP_LIST0) (f, t, peek, __VA_ARGS__)
// PYBIND11_MAP_LIST(f, t, a1, a2, ...) expands to f(t, a1), f(t, a2), ...
#define PYBIND11_MAP_LIST(f, t, ...) \
    PYBIND11_EVAL (PYBIND11_MAP_LIST1 (f, t, __VA_ARGS__, (), 0))

#define PYBIND11_NUMPY_DTYPE(Type, ...) \
    ::pybind11::detail::npy_format_descriptor<Type>::register_dtype \
        (::std::vector<::pybind11::detail::field_descriptor> \
         {PYBIND11_MAP_LIST (PYBIND11_FIELD_DESCRIPTOR, Type, __VA_ARGS__)})

#if defined(_MSC_VER) && !defined(__clang__)
#define PYBIND11_MAP2_LIST_NEXT1(test, next) \
    PYBIND11_EVAL0 (PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0))
#else
#define PYBIND11_MAP2_LIST_NEXT1(test, next) \
    PYBIND11_MAP_NEXT0 (test, PYBIND11_MAP_COMMA next, 0)
#endif
#define PYBIND11_MAP2_LIST_NEXT(test, next) \
    PYBIND11_MAP2_LIST_NEXT1 (PYBIND11_MAP_GET_END test, next)
#define PYBIND11_MAP2_LIST0(f, t, x1, x2, peek, ...) \
    f(t, x1, x2) PYBIND11_MAP2_LIST_NEXT (peek, PYBIND11_MAP2_LIST1) (f, t, peek, __VA_ARGS__)
#define PYBIND11_MAP2_LIST1(f, t, x1, x2, peek, ...) \
    f(t, x1, x2) PYBIND11_MAP2_LIST_NEXT (peek, PYBIND11_MAP2_LIST0) (f, t, peek, __VA_ARGS__)
// PYBIND11_MAP2_LIST(f, t, a1, a2, ...) expands to f(t, a1, a2), f(t, a3, a4), ...
#define PYBIND11_MAP2_LIST(f, t, ...) \
    PYBIND11_EVAL (PYBIND11_MAP2_LIST1 (f, t, __VA_ARGS__, (), 0))

#define PYBIND11_NUMPY_DTYPE_EX(Type, ...) \
    ::pybind11::detail::npy_format_descriptor<Type>::register_dtype \
        (::std::vector<::pybind11::detail::field_descriptor> \
         {PYBIND11_MAP2_LIST (PYBIND11_FIELD_DESCRIPTOR_EX, Type, __VA_ARGS__)})

#endif // __CLION_IDE__

template  <class T>
using array_iterator = typename std::add_pointer<T>::type;

template <class T>
array_iterator<T> array_begin(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr));
}

template <class T>
array_iterator<T> array_end(const buffer_info& buffer) {
    return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr) + buffer.size);
}

class common_iterator {
public:
    using container_type = std::vector<ssize_t>;
    using value_type = container_type::value_type;
    using size_type = container_type::size_type;

    common_iterator() : p_ptr(0), m_strides() {}

    common_iterator(void* ptr, const container_type& strides, const container_type& shape)
        : p_ptr(reinterpret_cast<char*>(ptr)), m_strides(strides.size()) {
        m_strides.back() = static_cast<value_type>(strides.back());
        for (size_type i = m_strides.size() - 1; i != 0; --i) {
            size_type j = i - 1;
            value_type s = static_cast<value_type>(shape[i]);
            m_strides[j] = strides[j] + m_strides[i] - strides[i] * s;
        }
    }

    void increment(size_type dim) {
        p_ptr += m_strides[dim];
    }

    void* data() const {
        return p_ptr;
    }

private:
    char* p_ptr;
    container_type m_strides;
};

template <size_t N> class multi_array_iterator {
public:
    using container_type = std::vector<ssize_t>;

    multi_array_iterator(const std::array<buffer_info, N> &buffers,
                         const container_type &shape)
        : m_shape(shape.size()), m_index(shape.size(), 0),
          m_common_iterator() {

        // Manual copy to avoid conversion warning if using std::copy
        for (size_t i = 0; i < shape.size(); ++i)
            m_shape[i] = shape[i];

        container_type strides(shape.size());
        for (size_t i = 0; i < N; ++i)
            init_common_iterator(buffers[i], shape, m_common_iterator[i], strides);
    }

    multi_array_iterator& operator++() {
        for (size_t j = m_index.size(); j != 0; --j) {
            size_t i = j - 1;
            if (++m_index[i] != m_shape[i]) {
                increment_common_iterator(i);
                break;
            } else {
                m_index[i] = 0;
            }
        }
        return *this;
    }

    template <size_t K, class T = void> T* data() const {
        return reinterpret_cast<T*>(m_common_iterator[K].data());
    }

private:

    using common_iter = common_iterator;

    void init_common_iterator(const buffer_info &buffer,
                              const container_type &shape,
                              common_iter &iterator,
                              container_type &strides) {
        auto buffer_shape_iter = buffer.shape.rbegin();
        auto buffer_strides_iter = buffer.strides.rbegin();
        auto shape_iter = shape.rbegin();
        auto strides_iter = strides.rbegin();

        while (buffer_shape_iter != buffer.shape.rend()) {
            if (*shape_iter == *buffer_shape_iter)
                *strides_iter = *buffer_strides_iter;
            else
                *strides_iter = 0;

            ++buffer_shape_iter;
            ++buffer_strides_iter;
            ++shape_iter;
            ++strides_iter;
        }

        std::fill(strides_iter, strides.rend(), 0);
        iterator = common_iter(buffer.ptr, strides, shape);
    }

    void increment_common_iterator(size_t dim) {
        for (auto &iter : m_common_iterator)
            iter.increment(dim);
    }

    container_type m_shape;
    container_type m_index;
    std::array<common_iter, N> m_common_iterator;
};

enum class broadcast_trivial { non_trivial, c_trivial, f_trivial };

// Populates the shape and number of dimensions for the set of buffers.  Returns a broadcast_trivial
// enum value indicating whether the broadcast is "trivial"--that is, has each buffer being either a
// singleton or a full-size, C-contiguous (`c_trivial`) or Fortran-contiguous (`f_trivial`) storage
// buffer; returns `non_trivial` otherwise.
template <size_t N>
broadcast_trivial broadcast(const std::array<buffer_info, N> &buffers, ssize_t &ndim, std::vector<ssize_t> &shape) {
    ndim = std::accumulate(buffers.begin(), buffers.end(), ssize_t(0), [](ssize_t res, const buffer_info &buf) {
        return std::max(res, buf.ndim);
    });

    shape.clear();
    shape.resize((size_t) ndim, 1);

    // Figure out the output size, and make sure all input arrays conform (i.e. are either size 1 or
    // the full size).
    for (size_t i = 0; i < N; ++i) {
        auto res_iter = shape.rbegin();
        auto end = buffers[i].shape.rend();
        for (auto shape_iter = buffers[i].shape.rbegin(); shape_iter != end; ++shape_iter, ++res_iter) {
            const auto &dim_size_in = *shape_iter;
            auto &dim_size_out = *res_iter;

            // Each input dimension can either be 1 or `n`, but `n` values must match across buffers
            if (dim_size_out == 1)
                dim_size_out = dim_size_in;
            else if (dim_size_in != 1 && dim_size_in != dim_size_out)
                pybind11_fail("pybind11::vectorize: incompatible size/dimension of inputs!");
        }
    }

    bool trivial_broadcast_c = true;
    bool trivial_broadcast_f = true;
    for (size_t i = 0; i < N && (trivial_broadcast_c || trivial_broadcast_f); ++i) {
        if (buffers[i].size == 1)
            continue;

        // Require the same number of dimensions:
        if (buffers[i].ndim != ndim)
            return broadcast_trivial::non_trivial;

        // Require all dimensions be full-size:
        if (!std::equal(buffers[i].shape.cbegin(), buffers[i].shape.cend(), shape.cbegin()))
            return broadcast_trivial::non_trivial;

        // Check for C contiguity (but only if previous inputs were also C contiguous)
        if (trivial_broadcast_c) {
            ssize_t expect_stride = buffers[i].itemsize;
            auto end = buffers[i].shape.crend();
            for (auto shape_iter = buffers[i].shape.crbegin(), stride_iter = buffers[i].strides.crbegin();
                    trivial_broadcast_c && shape_iter != end; ++shape_iter, ++stride_iter) {
                if (expect_stride == *stride_iter)
                    expect_stride *= *shape_iter;
                else
                    trivial_broadcast_c = false;
            }
        }

        // Check for Fortran contiguity (if previous inputs were also F contiguous)
        if (trivial_broadcast_f) {
            ssize_t expect_stride = buffers[i].itemsize;
            auto end = buffers[i].shape.cend();
            for (auto shape_iter = buffers[i].shape.cbegin(), stride_iter = buffers[i].strides.cbegin();
                    trivial_broadcast_f && shape_iter != end; ++shape_iter, ++stride_iter) {
                if (expect_stride == *stride_iter)
                    expect_stride *= *shape_iter;
                else
                    trivial_broadcast_f = false;
            }
        }
    }

    return
        trivial_broadcast_c ? broadcast_trivial::c_trivial :
        trivial_broadcast_f ? broadcast_trivial::f_trivial :
        broadcast_trivial::non_trivial;
}

template <typename T>
struct vectorize_arg {
    static_assert(!std::is_rvalue_reference<T>::value, "Functions with rvalue reference arguments cannot be vectorized");
    // The wrapped function gets called with this type:
    using call_type = remove_reference_t<T>;
    // Is this a vectorized argument?
    static constexpr bool vectorize =
        satisfies_any_of<call_type, std::is_arithmetic, is_complex, std::is_pod>::value &&
        satisfies_none_of<call_type, std::is_pointer, std::is_array, is_std_array, std::is_enum>::value &&
        (!std::is_reference<T>::value ||
         (std::is_lvalue_reference<T>::value && std::is_const<call_type>::value));
    // Accept this type: an array for vectorized types, otherwise the type as-is:
    using type = conditional_t<vectorize, array_t<remove_cv_t<call_type>, array::forcecast>, T>;
};

template <typename Func, typename Return, typename... Args>
struct vectorize_helper {
private:
    static constexpr size_t N = sizeof...(Args);
    static constexpr size_t NVectorized = constexpr_sum(vectorize_arg<Args>::vectorize...);
    static_assert(NVectorized >= 1,
            "pybind11::vectorize(...) requires a function with at least one vectorizable argument");

public:
    template <typename T>
    explicit vectorize_helper(T &&f) : f(std::forward<T>(f)) { }

    object operator()(typename vectorize_arg<Args>::type... args) {
        return run(args...,
                   make_index_sequence<N>(),
                   select_indices<vectorize_arg<Args>::vectorize...>(),
                   make_index_sequence<NVectorized>());
    }

private:
    remove_reference_t<Func> f;

    // Internal compiler error in MSVC 19.16.27025.1 (Visual Studio 2017 15.9.4), when compiling with "/permissive-" flag
    // when arg_call_types is manually inlined.
    using arg_call_types = std::tuple<typename vectorize_arg<Args>::call_type...>;
    template <size_t Index> using param_n_t = typename std::tuple_element<Index, arg_call_types>::type;

    // Runs a vectorized function given arguments tuple and three index sequences:
    //     - Index is the full set of 0 ... (N-1) argument indices;
    //     - VIndex is the subset of argument indices with vectorized parameters, letting us access
    //       vectorized arguments (anything not in this sequence is passed through)
    //     - BIndex is a incremental sequence (beginning at 0) of the same size as VIndex, so that
    //       we can store vectorized buffer_infos in an array (argument VIndex has its buffer at
    //       index BIndex in the array).
    template <size_t... Index, size_t... VIndex, size_t... BIndex> object run(
            typename vectorize_arg<Args>::type &...args,
            index_sequence<Index...> i_seq, index_sequence<VIndex...> vi_seq, index_sequence<BIndex...> bi_seq) {

        // Pointers to values the function was called with; the vectorized ones set here will start
        // out as array_t<T> pointers, but they will be changed them to T pointers before we make
        // call the wrapped function.  Non-vectorized pointers are left as-is.
        std::array<void *, N> params{{ &args... }};

        // The array of `buffer_info`s of vectorized arguments:
        std::array<buffer_info, NVectorized> buffers{{ reinterpret_cast<array *>(params[VIndex])->request()... }};

        /* Determine dimensions parameters of output array */
        ssize_t nd = 0;
        std::vector<ssize_t> shape(0);
        auto trivial = broadcast(buffers, nd, shape);
        size_t ndim = (size_t) nd;

        size_t size = std::accumulate(shape.begin(), shape.end(), (size_t) 1, std::multiplies<size_t>());

        // If all arguments are 0-dimension arrays (i.e. single values) return a plain value (i.e.
        // not wrapped in an array).
        if (size == 1 && ndim == 0) {
            PYBIND11_EXPAND_SIDE_EFFECTS(params[VIndex] = buffers[BIndex].ptr);
            return cast(f(*reinterpret_cast<param_n_t<Index> *>(params[Index])...));
        }

        array_t<Return> result;
        if (trivial == broadcast_trivial::f_trivial) result = array_t<Return, array::f_style>(shape);
        else result = array_t<Return>(shape);

        if (size == 0) return std::move(result);

        /* Call the function */
        if (trivial == broadcast_trivial::non_trivial)
            apply_broadcast(buffers, params, result, i_seq, vi_seq, bi_seq);
        else
            apply_trivial(buffers, params, result.mutable_data(), size, i_seq, vi_seq, bi_seq);

        return std::move(result);
    }

    template <size_t... Index, size_t... VIndex, size_t... BIndex>
    void apply_trivial(std::array<buffer_info, NVectorized> &buffers,
                       std::array<void *, N> &params,
                       Return *out,
                       size_t size,
                       index_sequence<Index...>, index_sequence<VIndex...>, index_sequence<BIndex...>) {

        // Initialize an array of mutable byte references and sizes with references set to the
        // appropriate pointer in `params`; as we iterate, we'll increment each pointer by its size
        // (except for singletons, which get an increment of 0).
        std::array<std::pair<unsigned char *&, const size_t>, NVectorized> vecparams{{
            std::pair<unsigned char *&, const size_t>(
                    reinterpret_cast<unsigned char *&>(params[VIndex] = buffers[BIndex].ptr),
                    buffers[BIndex].size == 1 ? 0 : sizeof(param_n_t<VIndex>)
            )...
        }};

        for (size_t i = 0; i < size; ++i) {
            out[i] = f(*reinterpret_cast<param_n_t<Index> *>(params[Index])...);
            for (auto &x : vecparams) x.first += x.second;
        }
    }

    template <size_t... Index, size_t... VIndex, size_t... BIndex>
    void apply_broadcast(std::array<buffer_info, NVectorized> &buffers,
                         std::array<void *, N> &params,
                         array_t<Return> &output_array,
                         index_sequence<Index...>, index_sequence<VIndex...>, index_sequence<BIndex...>) {

        buffer_info output = output_array.request();
        multi_array_iterator<NVectorized> input_iter(buffers, output.shape);

        for (array_iterator<Return> iter = array_begin<Return>(output), end = array_end<Return>(output);
             iter != end;
             ++iter, ++input_iter) {
            PYBIND11_EXPAND_SIDE_EFFECTS((
                params[VIndex] = input_iter.template data<BIndex>()
            ));
            *iter = f(*reinterpret_cast<param_n_t<Index> *>(std::get<Index>(params))...);
        }
    }
};

template <typename Func, typename Return, typename... Args>
vectorize_helper<Func, Return, Args...>
vectorize_extractor(const Func &f, Return (*) (Args ...)) {
    return detail::vectorize_helper<Func, Return, Args...>(f);
}

template <typename T, int Flags> struct handle_type_name<array_t<T, Flags>> {
    static constexpr auto name = _("numpy.ndarray[") + npy_format_descriptor<T>::name + _("]");
};

PYBIND11_NAMESPACE_END(detail)

// Vanilla pointer vectorizer:
template <typename Return, typename... Args>
detail::vectorize_helper<Return (*)(Args...), Return, Args...>
vectorize(Return (*f) (Args ...)) {
    return detail::vectorize_helper<Return (*)(Args...), Return, Args...>(f);
}

// lambda vectorizer:
template <typename Func, detail::enable_if_t<detail::is_lambda<Func>::value, int> = 0>
auto vectorize(Func &&f) -> decltype(
        detail::vectorize_extractor(std::forward<Func>(f), (detail::function_signature_t<Func> *) nullptr)) {
    return detail::vectorize_extractor(std::forward<Func>(f), (detail::function_signature_t<Func> *) nullptr);
}

// Vectorize a class method (non-const):
template <typename Return, typename Class, typename... Args,
          typename Helper = detail::vectorize_helper<decltype(std::mem_fn(std::declval<Return (Class::*)(Args...)>())), Return, Class *, Args...>>
Helper vectorize(Return (Class::*f)(Args...)) {
    return Helper(std::mem_fn(f));
}

// Vectorize a class method (const):
template <typename Return, typename Class, typename... Args,
          typename Helper = detail::vectorize_helper<decltype(std::mem_fn(std::declval<Return (Class::*)(Args...) const>())), Return, const Class *, Args...>>
Helper vectorize(Return (Class::*f)(Args...) const) {
    return Helper(std::mem_fn(f));
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif


//========= end of #include "numpy.h" ============


//========= begin of #include "eval.h" ============

/*
    pybind11/exec.h: Support for evaluating Python expressions and statements
    from strings and files

    Copyright (c) 2016 Klemens Morgenstern <klemens.morgenstern@ed-chemnitz.de> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

enum eval_mode {
    /// Evaluate a string containing an isolated expression
    eval_expr,

    /// Evaluate a string containing a single statement. Returns \c none
    eval_single_statement,

    /// Evaluate a string containing a sequence of statement. Returns \c none
    eval_statements
};

template <eval_mode mode = eval_expr>
object eval(str expr, object global = globals(), object local = object()) {
    if (!local)
        local = global;

    /* PyRun_String does not accept a PyObject / encoding specifier,
       this seems to be the only alternative */
    std::string buffer = "# -*- coding: utf-8 -*-\n" + (std::string) expr;

    int start;
    switch (mode) {
        case eval_expr:             start = Py_eval_input;   break;
        case eval_single_statement: start = Py_single_input; break;
        case eval_statements:       start = Py_file_input;   break;
        default: pybind11_fail("invalid evaluation mode");
    }

    PyObject *result = PyRun_String(buffer.c_str(), start, global.ptr(), local.ptr());
    if (!result)
        throw error_already_set();
    return reinterpret_steal<object>(result);
}

template <eval_mode mode = eval_expr, size_t N>
object eval(const char (&s)[N], object global = globals(), object local = object()) {
    /* Support raw string literals by removing common leading whitespace */
    auto expr = (s[0] == '\n') ? str(module::import("textwrap").attr("dedent")(s))
                               : str(s);
    return eval<mode>(expr, global, local);
}

inline void exec(str expr, object global = globals(), object local = object()) {
    eval<eval_statements>(expr, global, local);
}

template <size_t N>
void exec(const char (&s)[N], object global = globals(), object local = object()) {
    eval<eval_statements>(s, global, local);
}

#if defined(PYPY_VERSION) && PY_VERSION_HEX >= 0x3000000
template <eval_mode mode = eval_statements>
object eval_file(str, object, object) {
    pybind11_fail("eval_file not supported in PyPy3. Use eval");
}
template <eval_mode mode = eval_statements>
object eval_file(str, object) {
    pybind11_fail("eval_file not supported in PyPy3. Use eval");
}
template <eval_mode mode = eval_statements>
object eval_file(str) {
    pybind11_fail("eval_file not supported in PyPy3. Use eval");
}
#else
template <eval_mode mode = eval_statements>
object eval_file(str fname, object global = globals(), object local = object()) {
    if (!local)
        local = global;

    int start;
    switch (mode) {
        case eval_expr:             start = Py_eval_input;   break;
        case eval_single_statement: start = Py_single_input; break;
        case eval_statements:       start = Py_file_input;   break;
        default: pybind11_fail("invalid evaluation mode");
    }

    int closeFile = 1;
    std::string fname_str = (std::string) fname;
#if PY_VERSION_HEX >= 0x03040000
    FILE *f = _Py_fopen_obj(fname.ptr(), "r");
#elif PY_VERSION_HEX >= 0x03000000
    FILE *f = _Py_fopen(fname.ptr(), "r");
#else
    /* No unicode support in open() :( */
    auto fobj = reinterpret_steal<object>(PyFile_FromString(
        const_cast<char *>(fname_str.c_str()),
        const_cast<char*>("r")));
    FILE *f = nullptr;
    if (fobj)
        f = PyFile_AsFile(fobj.ptr());
    closeFile = 0;
#endif
    if (!f) {
        PyErr_Clear();
        pybind11_fail("File \"" + fname_str + "\" could not be opened!");
    }

#if PY_VERSION_HEX < 0x03000000 && defined(PYPY_VERSION)
    PyObject *result = PyRun_File(f, fname_str.c_str(), start, global.ptr(),
                                  local.ptr());
    (void) closeFile;
#else
    PyObject *result = PyRun_FileEx(f, fname_str.c_str(), start, global.ptr(),
                                    local.ptr(), closeFile);
#endif

    if (!result)
        throw error_already_set();
    return reinterpret_steal<object>(result);
}
#endif

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "eval.h" ============


//========= begin of #include "embed.h" ============

/*
    pybind11/embed.h: Support for embedding the interpreter

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"
// ans ignore: #include "eval.h"

#if defined(PYPY_VERSION)
#  error Embedding the interpreter is not supported with PyPy
#endif

#if PY_MAJOR_VERSION >= 3
#  define PYBIND11_EMBEDDED_MODULE_IMPL(name)            \
      extern "C" PyObject *pybind11_init_impl_##name();  \
      extern "C" PyObject *pybind11_init_impl_##name() { \
          return pybind11_init_wrapper_##name();         \
      }
#else
#  define PYBIND11_EMBEDDED_MODULE_IMPL(name)            \
      extern "C" void pybind11_init_impl_##name();       \
      extern "C" void pybind11_init_impl_##name() {      \
          pybind11_init_wrapper_##name();                \
      }
#endif

/** \rst
    Add a new module to the table of builtins for the interpreter. Must be
    defined in global scope. The first macro parameter is the name of the
    module (without quotes). The second parameter is the variable which will
    be used as the interface to add functions and classes to the module.

    .. code-block:: cpp

        PYBIND11_EMBEDDED_MODULE(example, m) {
            // ... initialize functions and classes here
            m.def("foo", []() {
                return "Hello, World!";
            });
        }
 \endrst */
#define PYBIND11_EMBEDDED_MODULE(name, variable)                              \
    static void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &);    \
    static PyObject PYBIND11_CONCAT(*pybind11_init_wrapper_, name)() {        \
        auto m = pybind11::module(PYBIND11_TOSTRING(name));                   \
        try {                                                                 \
            PYBIND11_CONCAT(pybind11_init_, name)(m);                         \
            return m.ptr();                                                   \
        } catch (pybind11::error_already_set &e) {                            \
            PyErr_SetString(PyExc_ImportError, e.what());                     \
            return nullptr;                                                   \
        } catch (const std::exception &e) {                                   \
            PyErr_SetString(PyExc_ImportError, e.what());                     \
            return nullptr;                                                   \
        }                                                                     \
    }                                                                         \
    PYBIND11_EMBEDDED_MODULE_IMPL(name)                                       \
    pybind11::detail::embedded_module PYBIND11_CONCAT(pybind11_module_, name) \
                              (PYBIND11_TOSTRING(name),             \
                               PYBIND11_CONCAT(pybind11_init_impl_, name));   \
    void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::module &variable)


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Python 2.7/3.x compatible version of `PyImport_AppendInittab` and error checks.
struct embedded_module {
#if PY_MAJOR_VERSION >= 3
    using init_t = PyObject *(*)();
#else
    using init_t = void (*)();
#endif
    embedded_module(const char *name, init_t init) {
        if (Py_IsInitialized())
            pybind11_fail("Can't add new modules after the interpreter has been initialized");

        auto result = PyImport_AppendInittab(name, init);
        if (result == -1)
            pybind11_fail("Insufficient memory to add a new module");
    }
};

PYBIND11_NAMESPACE_END(detail)

/** \rst
    Initialize the Python interpreter. No other pybind11 or CPython API functions can be
    called before this is done; with the exception of `PYBIND11_EMBEDDED_MODULE`. The
    optional parameter can be used to skip the registration of signal handlers (see the
    `Python documentation`_ for details). Calling this function again after the interpreter
    has already been initialized is a fatal error.

    If initializing the Python interpreter fails, then the program is terminated.  (This
    is controlled by the CPython runtime and is an exception to pybind11's normal behavior
    of throwing exceptions on errors.)

    .. _Python documentation: https://docs.python.org/3/c-api/init.html#c.Py_InitializeEx
 \endrst */
inline void initialize_interpreter(bool init_signal_handlers = true) {
    if (Py_IsInitialized())
        pybind11_fail("The interpreter is already running");

    Py_InitializeEx(init_signal_handlers ? 1 : 0);

    // Make .py files in the working directory available by default
    module::import("sys").attr("path").cast<list>().append(".");
}

/** \rst
    Shut down the Python interpreter. No pybind11 or CPython API functions can be called
    after this. In addition, pybind11 objects must not outlive the interpreter:

    .. code-block:: cpp

        { // BAD
            py::initialize_interpreter();
            auto hello = py::str("Hello, World!");
            py::finalize_interpreter();
        } // <-- BOOM, hello's destructor is called after interpreter shutdown

        { // GOOD
            py::initialize_interpreter();
            { // scoped
                auto hello = py::str("Hello, World!");
            } // <-- OK, hello is cleaned up properly
            py::finalize_interpreter();
        }

        { // BETTER
            py::scoped_interpreter guard{};
            auto hello = py::str("Hello, World!");
        }

    .. warning::

        The interpreter can be restarted by calling `initialize_interpreter` again.
        Modules created using pybind11 can be safely re-initialized. However, Python
        itself cannot completely unload binary extension modules and there are several
        caveats with regard to interpreter restarting. All the details can be found
        in the CPython documentation. In short, not all interpreter memory may be
        freed, either due to reference cycles or user-created global data.

 \endrst */
inline void finalize_interpreter() {
    handle builtins(PyEval_GetBuiltins());
    const char *id = PYBIND11_INTERNALS_ID;

    // Get the internals pointer (without creating it if it doesn't exist).  It's possible for the
    // internals to be created during Py_Finalize() (e.g. if a py::capsule calls `get_internals()`
    // during destruction), so we get the pointer-pointer here and check it after Py_Finalize().
    detail::internals **internals_ptr_ptr = detail::get_internals_pp();
    // It could also be stashed in builtins, so look there too:
    if (builtins.contains(id) && isinstance<capsule>(builtins[id]))
        internals_ptr_ptr = capsule(builtins[id]);

    Py_Finalize();

    if (internals_ptr_ptr) {
        delete *internals_ptr_ptr;
        *internals_ptr_ptr = nullptr;
    }
}

/** \rst
    Scope guard version of `initialize_interpreter` and `finalize_interpreter`.
    This a move-only guard and only a single instance can exist.

    .. code-block:: cpp

        #include <pybind11/embed.h>

        int main() {
            py::scoped_interpreter guard{};
            py::print(Hello, World!);
        } // <-- interpreter shutdown
 \endrst */
class scoped_interpreter {
public:
    scoped_interpreter(bool init_signal_handlers = true) {
        initialize_interpreter(init_signal_handlers);
    }

    scoped_interpreter(const scoped_interpreter &) = delete;
    scoped_interpreter(scoped_interpreter &&other) noexcept { other.is_valid = false; }
    scoped_interpreter &operator=(const scoped_interpreter &) = delete;
    scoped_interpreter &operator=(scoped_interpreter &&) = delete;

    ~scoped_interpreter() {
        if (is_valid)
            finalize_interpreter();
    }

private:
    bool is_valid = true;
};

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "embed.h" ============


//========= begin of #include "functional.h" ============

/*
    pybind11/functional.h: std::function<> support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"
#include <functional>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <typename Return, typename... Args>
struct type_caster<std::function<Return(Args...)>> {
    using type = std::function<Return(Args...)>;
    using retval_type = conditional_t<std::is_same<Return, void>::value, void_type, Return>;
    using function_type = Return (*) (Args...);

public:
    bool load(handle src, bool convert) {
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) return false;
            return true;
        }

        if (!isinstance<function>(src))
            return false;

        auto func = reinterpret_borrow<function>(src);

        /*
           When passing a C++ function as an argument to another C++
           function via Python, every function call would normally involve
           a full C++ -> Python -> C++ roundtrip, which can be prohibitive.
           Here, we try to at least detect the case where the function is
           stateless (i.e. function pointer or lambda function without
           captured variables), in which case the roundtrip can be avoided.
         */
        if (auto cfunc = func.cpp_function()) {
            auto c = reinterpret_borrow<capsule>(PyCFunction_GET_SELF(cfunc.ptr()));
            auto rec = (function_record *) c;

            if (rec && rec->is_stateless &&
                    same_type(typeid(function_type), *reinterpret_cast<const std::type_info *>(rec->data[1]))) {
                struct capture { function_type f; };
                value = ((capture *) &rec->data)->f;
                return true;
            }
        }

        // ensure GIL is held during functor destruction
        struct func_handle {
            function f;
            func_handle(function&& f_) : f(std::move(f_)) {}
            func_handle(const func_handle&) = default;
            ~func_handle() {
                gil_scoped_acquire acq;
                function kill_f(std::move(f));
            }
        };

        // to emulate 'move initialization capture' in C++11
        struct func_wrapper {
            func_handle hfunc;
            func_wrapper(func_handle&& hf): hfunc(std::move(hf)) {}
            Return operator()(Args... args) const {
                gil_scoped_acquire acq;
                object retval(hfunc.f(std::forward<Args>(args)...));
                /* Visual studio 2015 parser issue: need parentheses around this expression */
                return (retval.template cast<Return>());
            }
        };

        value = func_wrapper(func_handle(std::move(func)));
        return true;
    }

    template <typename Func>
    static handle cast(Func &&f_, return_value_policy policy, handle /* parent */) {
        if (!f_)
            return none().inc_ref();

        auto result = f_.template target<function_type>();
        if (result)
            return cpp_function(*result, policy).release();
        else
            return cpp_function(std::forward<Func>(f_), policy).release();
    }

    PYBIND11_TYPE_CASTER(type, _("Callable[[") + concat(make_caster<Args>::name...) + _("], ")
                               + make_caster<retval_type>::name + _("]"));
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "functional.h" ============


//========= begin of #include "iostream.h" ============

/*
    pybind11/iostream.h -- Tools to assist with redirecting cout and cerr to Python

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"

#include <streambuf>
#include <ostream>
#include <string>
#include <memory>
#include <iostream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// Buffer that writes to Python instead of C++
class pythonbuf : public std::streambuf {
private:
    using traits_type = std::streambuf::traits_type;

    const size_t buf_size;
    std::unique_ptr<char[]> d_buffer;
    object pywrite;
    object pyflush;

    int overflow(int c) {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(c);
            pbump(1);
        }
        return sync() == 0 ? traits_type::not_eof(c) : traits_type::eof();
    }

    int sync() {
        if (pbase() != pptr()) {
            // This subtraction cannot be negative, so dropping the sign
            str line(pbase(), static_cast<size_t>(pptr() - pbase()));

            {
                gil_scoped_acquire tmp;
                pywrite(line);
                pyflush();
            }

            setp(pbase(), epptr());
        }
        return 0;
    }

public:

    pythonbuf(object pyostream, size_t buffer_size = 1024)
        : buf_size(buffer_size),
          d_buffer(new char[buf_size]),
          pywrite(pyostream.attr("write")),
          pyflush(pyostream.attr("flush")) {
        setp(d_buffer.get(), d_buffer.get() + buf_size - 1);
    }

    pythonbuf(pythonbuf&&) = default;

    /// Sync before destroy
    ~pythonbuf() {
        sync();
    }
};

PYBIND11_NAMESPACE_END(detail)


/** \rst
    This a move-only guard that redirects output.

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        ...

        {
            py::scoped_ostream_redirect output;
            std::cout << "Hello, World!"; // Python stdout
        } // <-- return std::cout to normal

    You can explicitly pass the c++ stream and the python object,
    for example to guard stderr instead.

    .. code-block:: cpp

        {
            py::scoped_ostream_redirect output{std::cerr, py::module::import("sys").attr("stderr")};
            std::cerr << "Hello, World!";
        }
 \endrst */
class scoped_ostream_redirect {
protected:
    std::streambuf *old;
    std::ostream &costream;
    detail::pythonbuf buffer;

public:
    scoped_ostream_redirect(
            std::ostream &costream = std::cout,
            object pyostream = module::import("sys").attr("stdout"))
        : costream(costream), buffer(pyostream) {
        old = costream.rdbuf(&buffer);
    }

    ~scoped_ostream_redirect() {
        costream.rdbuf(old);
    }

    scoped_ostream_redirect(const scoped_ostream_redirect &) = delete;
    scoped_ostream_redirect(scoped_ostream_redirect &&other) = default;
    scoped_ostream_redirect &operator=(const scoped_ostream_redirect &) = delete;
    scoped_ostream_redirect &operator=(scoped_ostream_redirect &&) = delete;
};


/** \rst
    Like `scoped_ostream_redirect`, but redirects cerr by default. This class
    is provided primary to make ``py::call_guard`` easier to make.

    .. code-block:: cpp

     m.def("noisy_func", &noisy_func,
           py::call_guard<scoped_ostream_redirect,
                          scoped_estream_redirect>());

\endrst */
class scoped_estream_redirect : public scoped_ostream_redirect {
public:
    scoped_estream_redirect(
            std::ostream &costream = std::cerr,
            object pyostream = module::import("sys").attr("stderr"))
        : scoped_ostream_redirect(costream,pyostream) {}
};


PYBIND11_NAMESPACE_BEGIN(detail)

// Class to redirect output as a context manager. C++ backend.
class OstreamRedirect {
    bool do_stdout_;
    bool do_stderr_;
    std::unique_ptr<scoped_ostream_redirect> redirect_stdout;
    std::unique_ptr<scoped_estream_redirect> redirect_stderr;

public:
    OstreamRedirect(bool do_stdout = true, bool do_stderr = true)
        : do_stdout_(do_stdout), do_stderr_(do_stderr) {}

    void enter() {
        if (do_stdout_)
            redirect_stdout.reset(new scoped_ostream_redirect());
        if (do_stderr_)
            redirect_stderr.reset(new scoped_estream_redirect());
    }

    void exit() {
        redirect_stdout.reset();
        redirect_stderr.reset();
    }
};

PYBIND11_NAMESPACE_END(detail)

/** \rst
    This is a helper function to add a C++ redirect context manager to Python
    instead of using a C++ guard. To use it, add the following to your binding code:

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        ...

        py::add_ostream_redirect(m, "ostream_redirect");

    You now have a Python context manager that redirects your output:

    .. code-block:: python

        with m.ostream_redirect():
            m.print_to_cout_function()

    This manager can optionally be told which streams to operate on:

    .. code-block:: python

        with m.ostream_redirect(stdout=true, stderr=true):
            m.noisy_function_with_error_printing()

 \endrst */
inline class_<detail::OstreamRedirect> add_ostream_redirect(module m, std::string name = "ostream_redirect") {
    return class_<detail::OstreamRedirect>(m, name.c_str(), module_local())
        .def(init<bool,bool>(), arg("stdout")=true, arg("stderr")=true)
        .def("__enter__", &detail::OstreamRedirect::enter)
        .def("__exit__", [](detail::OstreamRedirect &self_, args) { self_.exit(); });
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "iostream.h" ============


//========= begin of #include "operators.h" ============

/*
    pybind11/operator.h: Metatemplates for operator overloading

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"

#if defined(__clang__) && !defined(__INTEL_COMPILER)
#  pragma clang diagnostic ignored "-Wunsequenced" // multiple unsequenced modifications to 'self' (when using def(py::self OP Type()))
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Enumeration with all supported operator types
enum op_id : int {
    op_add, op_sub, op_mul, op_div, op_mod, op_divmod, op_pow, op_lshift,
    op_rshift, op_and, op_xor, op_or, op_neg, op_pos, op_abs, op_invert,
    op_int, op_long, op_float, op_str, op_cmp, op_gt, op_ge, op_lt, op_le,
    op_eq, op_ne, op_iadd, op_isub, op_imul, op_idiv, op_imod, op_ilshift,
    op_irshift, op_iand, op_ixor, op_ior, op_complex, op_bool, op_nonzero,
    op_repr, op_truediv, op_itruediv, op_hash
};

enum op_type : int {
    op_l, /* base type on left */
    op_r, /* base type on right */
    op_u  /* unary operator */
};

struct self_t { };
static const self_t self = self_t();

/// Type for an unused type slot
struct undefined_t { };

/// Don't warn about an unused variable
inline self_t __self() { return self; }

/// base template of operator implementations
template <op_id, op_type, typename B, typename L, typename R> struct op_impl { };

/// Operator implementation generator
template <op_id id, op_type ot, typename L, typename R> struct op_ {
    template <typename Class, typename... Extra> void execute(Class &cl, const Extra&... extra) const {
        using Base = typename Class::type;
        using L_type = conditional_t<std::is_same<L, self_t>::value, Base, L>;
        using R_type = conditional_t<std::is_same<R, self_t>::value, Base, R>;
        using op = op_impl<id, ot, Base, L_type, R_type>;
        cl.def(op::name(), &op::execute, is_operator(), extra...);
        #if PY_MAJOR_VERSION < 3
        if (id == op_truediv || id == op_itruediv)
            cl.def(id == op_itruediv ? "__idiv__" : ot == op_l ? "__div__" : "__rdiv__",
                    &op::execute, is_operator(), extra...);
        #endif
    }
    template <typename Class, typename... Extra> void execute_cast(Class &cl, const Extra&... extra) const {
        using Base = typename Class::type;
        using L_type = conditional_t<std::is_same<L, self_t>::value, Base, L>;
        using R_type = conditional_t<std::is_same<R, self_t>::value, Base, R>;
        using op = op_impl<id, ot, Base, L_type, R_type>;
        cl.def(op::name(), &op::execute_cast, is_operator(), extra...);
        #if PY_MAJOR_VERSION < 3
        if (id == op_truediv || id == op_itruediv)
            cl.def(id == op_itruediv ? "__idiv__" : ot == op_l ? "__div__" : "__rdiv__",
                    &op::execute, is_operator(), extra...);
        #endif
    }
};

#define PYBIND11_BINARY_OPERATOR(id, rid, op, expr)                                    \
template <typename B, typename L, typename R> struct op_impl<op_##id, op_l, B, L, R> { \
    static char const* name() { return "__" #id "__"; }                                \
    static auto execute(const L &l, const R &r) -> decltype(expr) { return (expr); }   \
    static B execute_cast(const L &l, const R &r) { return B(expr); }                  \
};                                                                                     \
template <typename B, typename L, typename R> struct op_impl<op_##id, op_r, B, L, R> { \
    static char const* name() { return "__" #rid "__"; }                               \
    static auto execute(const R &r, const L &l) -> decltype(expr) { return (expr); }   \
    static B execute_cast(const R &r, const L &l) { return B(expr); }                  \
};                                                                                     \
inline op_<op_##id, op_l, self_t, self_t> op(const self_t &, const self_t &) {         \
    return op_<op_##id, op_l, self_t, self_t>();                                       \
}                                                                                      \
template <typename T> op_<op_##id, op_l, self_t, T> op(const self_t &, const T &) {    \
    return op_<op_##id, op_l, self_t, T>();                                            \
}                                                                                      \
template <typename T> op_<op_##id, op_r, T, self_t> op(const T &, const self_t &) {    \
    return op_<op_##id, op_r, T, self_t>();                                            \
}

#define PYBIND11_INPLACE_OPERATOR(id, op, expr)                                        \
template <typename B, typename L, typename R> struct op_impl<op_##id, op_l, B, L, R> { \
    static char const* name() { return "__" #id "__"; }                                \
    static auto execute(L &l, const R &r) -> decltype(expr) { return expr; }           \
    static B execute_cast(L &l, const R &r) { return B(expr); }                        \
};                                                                                     \
template <typename T> op_<op_##id, op_l, self_t, T> op(const self_t &, const T &) {    \
    return op_<op_##id, op_l, self_t, T>();                                            \
}

#define PYBIND11_UNARY_OPERATOR(id, op, expr)                                          \
template <typename B, typename L> struct op_impl<op_##id, op_u, B, L, undefined_t> {   \
    static char const* name() { return "__" #id "__"; }                                \
    static auto execute(const L &l) -> decltype(expr) { return expr; }                 \
    static B execute_cast(const L &l) { return B(expr); }                              \
};                                                                                     \
inline op_<op_##id, op_u, self_t, undefined_t> op(const self_t &) {                    \
    return op_<op_##id, op_u, self_t, undefined_t>();                                  \
}

PYBIND11_BINARY_OPERATOR(sub,       rsub,         operator-,    l - r)
PYBIND11_BINARY_OPERATOR(add,       radd,         operator+,    l + r)
PYBIND11_BINARY_OPERATOR(mul,       rmul,         operator*,    l * r)
PYBIND11_BINARY_OPERATOR(truediv,   rtruediv,     operator/,    l / r)
PYBIND11_BINARY_OPERATOR(mod,       rmod,         operator%,    l % r)
PYBIND11_BINARY_OPERATOR(lshift,    rlshift,      operator<<,   l << r)
PYBIND11_BINARY_OPERATOR(rshift,    rrshift,      operator>>,   l >> r)
PYBIND11_BINARY_OPERATOR(and,       rand,         operator&,    l & r)
PYBIND11_BINARY_OPERATOR(xor,       rxor,         operator^,    l ^ r)
PYBIND11_BINARY_OPERATOR(eq,        eq,           operator==,   l == r)
PYBIND11_BINARY_OPERATOR(ne,        ne,           operator!=,   l != r)
PYBIND11_BINARY_OPERATOR(or,        ror,          operator|,    l | r)
PYBIND11_BINARY_OPERATOR(gt,        lt,           operator>,    l > r)
PYBIND11_BINARY_OPERATOR(ge,        le,           operator>=,   l >= r)
PYBIND11_BINARY_OPERATOR(lt,        gt,           operator<,    l < r)
PYBIND11_BINARY_OPERATOR(le,        ge,           operator<=,   l <= r)
//PYBIND11_BINARY_OPERATOR(pow,       rpow,         pow,          std::pow(l,  r))
PYBIND11_INPLACE_OPERATOR(iadd,     operator+=,   l += r)
PYBIND11_INPLACE_OPERATOR(isub,     operator-=,   l -= r)
PYBIND11_INPLACE_OPERATOR(imul,     operator*=,   l *= r)
PYBIND11_INPLACE_OPERATOR(itruediv, operator/=,   l /= r)
PYBIND11_INPLACE_OPERATOR(imod,     operator%=,   l %= r)
PYBIND11_INPLACE_OPERATOR(ilshift,  operator<<=,  l <<= r)
PYBIND11_INPLACE_OPERATOR(irshift,  operator>>=,  l >>= r)
PYBIND11_INPLACE_OPERATOR(iand,     operator&=,   l &= r)
PYBIND11_INPLACE_OPERATOR(ixor,     operator^=,   l ^= r)
PYBIND11_INPLACE_OPERATOR(ior,      operator|=,   l |= r)
PYBIND11_UNARY_OPERATOR(neg,        operator-,    -l)
PYBIND11_UNARY_OPERATOR(pos,        operator+,    +l)
// WARNING: This usage of `abs` should only be done for existing STL overloads.
// Adding overloads directly in to the `std::` namespace is advised against:
// https://en.cppreference.com/w/cpp/language/extending_std
PYBIND11_UNARY_OPERATOR(abs,        abs,          std::abs(l))
PYBIND11_UNARY_OPERATOR(hash,       hash,         std::hash<L>()(l))
PYBIND11_UNARY_OPERATOR(invert,     operator~,    (~l))
PYBIND11_UNARY_OPERATOR(bool,       operator!,    !!l)
PYBIND11_UNARY_OPERATOR(int,        int_,         (int) l)
PYBIND11_UNARY_OPERATOR(float,      float_,       (double) l)

#undef PYBIND11_BINARY_OPERATOR
#undef PYBIND11_INPLACE_OPERATOR
#undef PYBIND11_UNARY_OPERATOR
PYBIND11_NAMESPACE_END(detail)

using detail::self;
// Add named operators so that they are accessible via `py::`.
using detail::hash;

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif


//========= end of #include "operators.h" ============


//========= begin of #include "stl.h" ============

/*
    pybind11/stl.h: Transparent conversion for STL data types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "pybind11.h"
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <iostream>
#include <list>
#include <deque>
#include <valarray>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

#ifdef __has_include
// std::optional (but including it in c++14 mode isn't allowed)
#  if defined(PYBIND11_CPP17) && __has_include(<optional>)
#    include <optional>
#    define PYBIND11_HAS_OPTIONAL 1
#  endif
// std::experimental::optional (but not allowed in c++11 mode)
#  if defined(PYBIND11_CPP14) && (__has_include(<experimental/optional>) && \
                                 !__has_include(<optional>))
#    include <experimental/optional>
#    define PYBIND11_HAS_EXP_OPTIONAL 1
#  endif
// std::variant
#  if defined(PYBIND11_CPP17) && __has_include(<variant>)
#    include <variant>
#    define PYBIND11_HAS_VARIANT 1
#  endif
#elif defined(_MSC_VER) && defined(PYBIND11_CPP17)
#  include <optional>
#  include <variant>
#  define PYBIND11_HAS_OPTIONAL 1
#  define PYBIND11_HAS_VARIANT 1
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/// Extracts an const lvalue reference or rvalue reference for U based on the type of T (e.g. for
/// forwarding a container element).  Typically used indirect via forwarded_type(), below.
template <typename T, typename U>
using forwarded_type = conditional_t<
    std::is_lvalue_reference<T>::value, remove_reference_t<U> &, remove_reference_t<U> &&>;

/// Forwards a value U as rvalue or lvalue according to whether T is rvalue or lvalue; typically
/// used for forwarding a container's elements.
template <typename T, typename U>
forwarded_type<T, U> forward_like(U &&u) {
    return std::forward<detail::forwarded_type<T, U>>(std::forward<U>(u));
}

template <typename Type, typename Key> struct set_caster {
    using type = Type;
    using key_conv = make_caster<Key>;

    bool load(handle src, bool convert) {
        if (!isinstance<pybind11::set>(src))
            return false;
        auto s = reinterpret_borrow<pybind11::set>(src);
        value.clear();
        for (auto entry : s) {
            key_conv conv;
            if (!conv.load(entry, convert))
                return false;
            value.insert(cast_op<Key &&>(std::move(conv)));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        if (!std::is_lvalue_reference<T>::value)
            policy = return_value_policy_override<Key>::policy(policy);
        pybind11::set s;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(key_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_ || !s.add(value_))
                return handle();
        }
        return s.release();
    }

    PYBIND11_TYPE_CASTER(type, _("Set[") + key_conv::name + _("]"));
};

template <typename Type, typename Key, typename Value> struct map_caster {
    using key_conv   = make_caster<Key>;
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<dict>(src))
            return false;
        auto d = reinterpret_borrow<dict>(src);
        value.clear();
        for (auto it : d) {
            key_conv kconv;
            value_conv vconv;
            if (!kconv.load(it.first.ptr(), convert) ||
                !vconv.load(it.second.ptr(), convert))
                return false;
            value.emplace(cast_op<Key &&>(std::move(kconv)), cast_op<Value &&>(std::move(vconv)));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        dict d;
        return_value_policy policy_key = policy;
        return_value_policy policy_value = policy;
        if (!std::is_lvalue_reference<T>::value) {
            policy_key = return_value_policy_override<Key>::policy(policy_key);
            policy_value = return_value_policy_override<Value>::policy(policy_value);
        }
        for (auto &&kv : src) {
            auto key = reinterpret_steal<object>(key_conv::cast(forward_like<T>(kv.first), policy_key, parent));
            auto value = reinterpret_steal<object>(value_conv::cast(forward_like<T>(kv.second), policy_value, parent));
            if (!key || !value)
                return handle();
            d[key] = value;
        }
        return d.release();
    }

    PYBIND11_TYPE_CASTER(Type, _("Dict[") + key_conv::name + _(", ") + value_conv::name + _("]"));
};

template <typename Type, typename Value> struct list_caster {
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src) || isinstance<str>(src))
            return false;
        auto s = reinterpret_borrow<sequence>(src);
        value.clear();
        reserve_maybe(s, &value);
        for (auto it : s) {
            value_conv conv;
            if (!conv.load(it, convert))
                return false;
            value.push_back(cast_op<Value &&>(std::move(conv)));
        }
        return true;
    }

private:
    template <typename T = Type,
              enable_if_t<std::is_same<decltype(std::declval<T>().reserve(0)), void>::value, int> = 0>
    void reserve_maybe(sequence s, Type *) { value.reserve(s.size()); }
    void reserve_maybe(sequence, void *) { }

public:
    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        if (!std::is_lvalue_reference<T>::value)
            policy = return_value_policy_override<Value>::policy(policy);
        list l(src.size());
        size_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(value_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    PYBIND11_TYPE_CASTER(Type, _("List[") + value_conv::name + _("]"));
};

template <typename Type, typename Alloc> struct type_caster<std::vector<Type, Alloc>>
 : list_caster<std::vector<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::deque<Type, Alloc>>
 : list_caster<std::deque<Type, Alloc>, Type> { };

template <typename Type, typename Alloc> struct type_caster<std::list<Type, Alloc>>
 : list_caster<std::list<Type, Alloc>, Type> { };

template <typename ArrayType, typename Value, bool Resizable, size_t Size = 0> struct array_caster {
    using value_conv = make_caster<Value>;

private:
    template <bool R = Resizable>
    bool require_size(enable_if_t<R, size_t> size) {
        if (value.size() != size)
            value.resize(size);
        return true;
    }
    template <bool R = Resizable>
    bool require_size(enable_if_t<!R, size_t> size) {
        return size == Size;
    }

public:
    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src))
            return false;
        auto l = reinterpret_borrow<sequence>(src);
        if (!require_size(l.size()))
            return false;
        size_t ctr = 0;
        for (auto it : l) {
            value_conv conv;
            if (!conv.load(it, convert))
                return false;
            value[ctr++] = cast_op<Value &&>(std::move(conv));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        list l(src.size());
        size_t index = 0;
        for (auto &&value : src) {
            auto value_ = reinterpret_steal<object>(value_conv::cast(forward_like<T>(value), policy, parent));
            if (!value_)
                return handle();
            PyList_SET_ITEM(l.ptr(), (ssize_t) index++, value_.release().ptr()); // steals a reference
        }
        return l.release();
    }

    PYBIND11_TYPE_CASTER(ArrayType, _("List[") + value_conv::name + _<Resizable>(_(""), _("[") + _<Size>() + _("]")) + _("]"));
};

template <typename Type, size_t Size> struct type_caster<std::array<Type, Size>>
 : array_caster<std::array<Type, Size>, Type, false, Size> { };

template <typename Type> struct type_caster<std::valarray<Type>>
 : array_caster<std::valarray<Type>, Type, true> { };

template <typename Key, typename Compare, typename Alloc> struct type_caster<std::set<Key, Compare, Alloc>>
  : set_caster<std::set<Key, Compare, Alloc>, Key> { };

template <typename Key, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_set<Key, Hash, Equal, Alloc>>
  : set_caster<std::unordered_set<Key, Hash, Equal, Alloc>, Key> { };

template <typename Key, typename Value, typename Compare, typename Alloc> struct type_caster<std::map<Key, Value, Compare, Alloc>>
  : map_caster<std::map<Key, Value, Compare, Alloc>, Key, Value> { };

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc> struct type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
  : map_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> { };

// This type caster is intended to be used for std::optional and std::experimental::optional
template<typename T> struct optional_caster {
    using value_conv = make_caster<typename T::value_type>;

    template <typename T_>
    static handle cast(T_ &&src, return_value_policy policy, handle parent) {
        if (!src)
            return none().inc_ref();
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<T>::policy(policy);
        }
        return value_conv::cast(*std::forward<T_>(src), policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        } else if (src.is_none()) {
            return true;  // default-constructed value is already empty
        }
        value_conv inner_caster;
        if (!inner_caster.load(src, convert))
            return false;

        value.emplace(cast_op<typename T::value_type &&>(std::move(inner_caster)));
        return true;
    }

    PYBIND11_TYPE_CASTER(T, _("Optional[") + value_conv::name + _("]"));
};

#if PYBIND11_HAS_OPTIONAL
template<typename T> struct type_caster<std::optional<T>>
    : public optional_caster<std::optional<T>> {};

template<> struct type_caster<std::nullopt_t>
    : public void_caster<std::nullopt_t> {};
#endif

#if PYBIND11_HAS_EXP_OPTIONAL
template<typename T> struct type_caster<std::experimental::optional<T>>
    : public optional_caster<std::experimental::optional<T>> {};

template<> struct type_caster<std::experimental::nullopt_t>
    : public void_caster<std::experimental::nullopt_t> {};
#endif

/// Visit a variant and cast any found type to Python
struct variant_caster_visitor {
    return_value_policy policy;
    handle parent;

    using result_type = handle; // required by boost::variant in C++11

    template <typename T>
    result_type operator()(T &&src) const {
        return make_caster<T>::cast(std::forward<T>(src), policy, parent);
    }
};

/// Helper class which abstracts away variant's `visit` function. `std::variant` and similar
/// `namespace::variant` types which provide a `namespace::visit()` function are handled here
/// automatically using argument-dependent lookup. Users can provide specializations for other
/// variant-like classes, e.g. `boost::variant` and `boost::apply_visitor`.
template <template<typename...> class Variant>
struct visit_helper {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(visit(std::forward<Args>(args)...)) {
        return visit(std::forward<Args>(args)...);
    }
};

/// Generic variant caster
template <typename Variant> struct variant_caster;

template <template<typename...> class V, typename... Ts>
struct variant_caster<V<Ts...>> {
    static_assert(sizeof...(Ts) > 0, "Variant must consist of at least one alternative.");

    template <typename U, typename... Us>
    bool load_alternative(handle src, bool convert, type_list<U, Us...>) {
        auto caster = make_caster<U>();
        if (caster.load(src, convert)) {
            value = cast_op<U>(caster);
            return true;
        }
        return load_alternative(src, convert, type_list<Us...>{});
    }

    bool load_alternative(handle, bool, type_list<>) { return false; }

    bool load(handle src, bool convert) {
        // Do a first pass without conversions to improve constructor resolution.
        // E.g. `py::int_(1).cast<variant<double, int>>()` needs to fill the `int`
        // slot of the variant. Without two-pass loading `double` would be filled
        // because it appears first and a conversion is possible.
        if (convert && load_alternative(src, false, type_list<Ts...>{}))
            return true;
        return load_alternative(src, convert, type_list<Ts...>{});
    }

    template <typename Variant>
    static handle cast(Variant &&src, return_value_policy policy, handle parent) {
        return visit_helper<V>::call(variant_caster_visitor{policy, parent},
                                     std::forward<Variant>(src));
    }

    using Type = V<Ts...>;
    PYBIND11_TYPE_CASTER(Type, _("Union[") + detail::concat(make_caster<Ts>::name...) + _("]"));
};

#if PYBIND11_HAS_VARIANT
template <typename... Ts>
struct type_caster<std::variant<Ts...>> : variant_caster<std::variant<Ts...>> { };
#endif

PYBIND11_NAMESPACE_END(detail)

inline std::ostream &operator<<(std::ostream &os, const handle &obj) {
    os << (std::string) str(obj);
    return os;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif


//========= end of #include "stl.h" ============


//========= begin of #include "stl_bind.h" ============

/*
    pybind11/std_bind.h: Binding generators for STL data types

    Copyright (c) 2016 Sergey Lyskov and Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

// ans ignore: #include "detail/common.h"
// ans ignore: #include "operators.h"

#include <algorithm>
#include <sstream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

/* SFINAE helper class used by 'is_comparable */
template <typename T>  struct container_traits {
    template <typename T2> static std::true_type test_comparable(decltype(std::declval<const T2 &>() == std::declval<const T2 &>())*);
    template <typename T2> static std::false_type test_comparable(...);
    template <typename T2> static std::true_type test_value(typename T2::value_type *);
    template <typename T2> static std::false_type test_value(...);
    template <typename T2> static std::true_type test_pair(typename T2::first_type *, typename T2::second_type *);
    template <typename T2> static std::false_type test_pair(...);

    static constexpr const bool is_comparable = std::is_same<std::true_type, decltype(test_comparable<T>(nullptr))>::value;
    static constexpr const bool is_pair = std::is_same<std::true_type, decltype(test_pair<T>(nullptr, nullptr))>::value;
    static constexpr const bool is_vector = std::is_same<std::true_type, decltype(test_value<T>(nullptr))>::value;
    static constexpr const bool is_element = !is_pair && !is_vector;
};

/* Default: is_comparable -> std::false_type */
template <typename T, typename SFINAE = void>
struct is_comparable : std::false_type { };

/* For non-map data structures, check whether operator== can be instantiated */
template <typename T>
struct is_comparable<
    T, enable_if_t<container_traits<T>::is_element &&
                   container_traits<T>::is_comparable>>
    : std::true_type { };

/* For a vector/map data structure, recursively check the value type (which is std::pair for maps) */
template <typename T>
struct is_comparable<T, enable_if_t<container_traits<T>::is_vector>> {
    static constexpr const bool value =
        is_comparable<typename T::value_type>::value;
};

/* For pairs, recursively check the two data types */
template <typename T>
struct is_comparable<T, enable_if_t<container_traits<T>::is_pair>> {
    static constexpr const bool value =
        is_comparable<typename T::first_type>::value &&
        is_comparable<typename T::second_type>::value;
};

/* Fallback functions */
template <typename, typename, typename... Args> void vector_if_copy_constructible(const Args &...) { }
template <typename, typename, typename... Args> void vector_if_equal_operator(const Args &...) { }
template <typename, typename, typename... Args> void vector_if_insertion_operator(const Args &...) { }
template <typename, typename, typename... Args> void vector_modifiers(const Args &...) { }

template<typename Vector, typename Class_>
void vector_if_copy_constructible(enable_if_t<is_copy_constructible<Vector>::value, Class_> &cl) {
    cl.def(init<const Vector &>(), "Copy constructor");
}

template<typename Vector, typename Class_>
void vector_if_equal_operator(enable_if_t<is_comparable<Vector>::value, Class_> &cl) {
    using T = typename Vector::value_type;

    cl.def(self == self);
    cl.def(self != self);

    cl.def("count",
        [](const Vector &v, const T &x) {
            return std::count(v.begin(), v.end(), x);
        },
        arg("x"),
        "Return the number of times ``x`` appears in the list"
    );

    cl.def("remove", [](Vector &v, const T &x) {
            auto p = std::find(v.begin(), v.end(), x);
            if (p != v.end())
                v.erase(p);
            else
                throw value_error();
        },
        arg("x"),
        "Remove the first item from the list whose value is x. "
        "It is an error if there is no such item."
    );

    cl.def("__contains__",
        [](const Vector &v, const T &x) {
            return std::find(v.begin(), v.end(), x) != v.end();
        },
        arg("x"),
        "Return true the container contains ``x``"
    );
}

// Vector modifiers -- requires a copyable vector_type:
// (Technically, some of these (pop and __delitem__) don't actually require copyability, but it seems
// silly to allow deletion but not insertion, so include them here too.)
template <typename Vector, typename Class_>
void vector_modifiers(enable_if_t<is_copy_constructible<typename Vector::value_type>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;

    auto wrap_i = [](DiffType i, SizeType n) {
        if (i < 0)
            i += n;
        if (i < 0 || (SizeType)i >= n)
            throw index_error();
        return i;
    };

    cl.def("append",
           [](Vector &v, const T &value) { v.push_back(value); },
           arg("x"),
           "Add an item to the end of the list");

    cl.def(init([](iterable it) {
        auto v = std::unique_ptr<Vector>(new Vector());
        v->reserve(len_hint(it));
        for (handle h : it)
           v->push_back(h.cast<T>());
        return v.release();
    }));

    cl.def("clear",
        [](Vector &v) {
            v.clear();
        },
        "Clear the contents"
    );

    cl.def("extend",
       [](Vector &v, const Vector &src) {
           v.insert(v.end(), src.begin(), src.end());
       },
       arg("L"),
       "Extend the list by appending all the items in the given list"
    );

    cl.def("extend",
       [](Vector &v, iterable it) {
           const size_t old_size = v.size();
           v.reserve(old_size + len_hint(it));
           try {
               for (handle h : it) {
                   v.push_back(h.cast<T>());
               }
           } catch (const cast_error &) {
               v.erase(v.begin() + static_cast<typename Vector::difference_type>(old_size), v.end());
               try {
                   v.shrink_to_fit();
               } catch (const std::exception &) {
                   // Do nothing
               }
               throw;
           }
       },
       arg("L"),
       "Extend the list by appending all the items in the given list"
    );

    cl.def("insert",
        [](Vector &v, DiffType i, const T &x) {
            // Can't use wrap_i; i == v.size() is OK
            if (i < 0)
                i += v.size();
            if (i < 0 || (SizeType)i > v.size())
                throw index_error();
            v.insert(v.begin() + i, x);
        },
        arg("i") , arg("x"),
        "Insert an item at a given position."
    );

    cl.def("pop",
        [](Vector &v) {
            if (v.empty())
                throw index_error();
            T t = v.back();
            v.pop_back();
            return t;
        },
        "Remove and return the last item"
    );

    cl.def("pop",
        [wrap_i](Vector &v, DiffType i) {
            i = wrap_i(i, v.size());
            T t = v[(SizeType) i];
            v.erase(v.begin() + i);
            return t;
        },
        arg("i"),
        "Remove and return the item at index ``i``"
    );

    cl.def("__setitem__",
        [wrap_i](Vector &v, DiffType i, const T &t) {
            i = wrap_i(i, v.size());
            v[(SizeType)i] = t;
        }
    );

    /// Slicing protocol
    cl.def("__getitem__",
        [](const Vector &v, slice slice) -> Vector * {
            size_t start, stop, step, slicelength;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw error_already_set();

            Vector *seq = new Vector();
            seq->reserve((size_t) slicelength);

            for (size_t i=0; i<slicelength; ++i) {
                seq->push_back(v[start]);
                start += step;
            }
            return seq;
        },
        arg("s"),
        "Retrieve list elements using a slice object"
    );

    cl.def("__setitem__",
        [](Vector &v, slice slice,  const Vector &value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw error_already_set();

            if (slicelength != value.size())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");

            for (size_t i=0; i<slicelength; ++i) {
                v[start] = value[i];
                start += step;
            }
        },
        "Assign list elements using a slice object"
    );

    cl.def("__delitem__",
        [wrap_i](Vector &v, DiffType i) {
            i = wrap_i(i, v.size());
            v.erase(v.begin() + i);
        },
        "Delete the list elements at index ``i``"
    );

    cl.def("__delitem__",
        [](Vector &v, slice slice) {
            size_t start, stop, step, slicelength;

            if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
                throw error_already_set();

            if (step == 1 && false) {
                v.erase(v.begin() + (DiffType) start, v.begin() + DiffType(start + slicelength));
            } else {
                for (size_t i = 0; i < slicelength; ++i) {
                    v.erase(v.begin() + DiffType(start));
                    start += step - 1;
                }
            }
        },
        "Delete list elements using a slice object"
    );

}

// If the type has an operator[] that doesn't return a reference (most notably std::vector<bool>),
// we have to access by copying; otherwise we return by reference.
template <typename Vector> using vector_needs_copy = negation<
    std::is_same<decltype(std::declval<Vector>()[typename Vector::size_type()]), typename Vector::value_type &>>;

// The usual case: access and iterate by reference
template <typename Vector, typename Class_>
void vector_accessor(enable_if_t<!vector_needs_copy<Vector>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;
    using ItType   = typename Vector::iterator;

    auto wrap_i = [](DiffType i, SizeType n) {
        if (i < 0)
            i += n;
        if (i < 0 || (SizeType)i >= n)
            throw index_error();
        return i;
    };

    cl.def("__getitem__",
        [wrap_i](Vector &v, DiffType i) -> T & {
            i = wrap_i(i, v.size());
            return v[(SizeType)i];
        },
        return_value_policy::reference_internal // ref + keepalive
    );

    cl.def("__iter__",
           [](Vector &v) {
               return make_iterator<
                   return_value_policy::reference_internal, ItType, ItType, T&>(
                   v.begin(), v.end());
           },
           keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );
}

// The case for special objects, like std::vector<bool>, that have to be returned-by-copy:
template <typename Vector, typename Class_>
void vector_accessor(enable_if_t<vector_needs_copy<Vector>::value, Class_> &cl) {
    using T = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;
    using ItType   = typename Vector::iterator;
    cl.def("__getitem__",
        [](const Vector &v, DiffType i) -> T {
            if (i < 0 && (i += v.size()) < 0)
                throw index_error();
            if ((SizeType)i >= v.size())
                throw index_error();
            return v[(SizeType)i];
        }
    );

    cl.def("__iter__",
           [](Vector &v) {
               return make_iterator<
                   return_value_policy::copy, ItType, ItType, T>(
                   v.begin(), v.end());
           },
           keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );
}

template <typename Vector, typename Class_> auto vector_if_insertion_operator(Class_ &cl, std::string const &name)
    -> decltype(std::declval<std::ostream&>() << std::declval<typename Vector::value_type>(), void()) {
    using size_type = typename Vector::size_type;

    cl.def("__repr__",
           [name](Vector &v) {
            std::ostringstream s;
            s << name << '[';
            for (size_type i=0; i < v.size(); ++i) {
                s << v[i];
                if (i != v.size() - 1)
                    s << ", ";
            }
            s << ']';
            return s.str();
        },
        "Return the canonical string representation of this list."
    );
}

// Provide the buffer interface for vectors if we have data() and we have a format for it
// GCC seems to have "void std::vector<bool>::data()" - doing SFINAE on the existence of data() is insufficient, we need to check it returns an appropriate pointer
template <typename Vector, typename = void>
struct vector_has_data_and_format : std::false_type {};
template <typename Vector>
struct vector_has_data_and_format<Vector, enable_if_t<std::is_same<decltype(format_descriptor<typename Vector::value_type>::format(), std::declval<Vector>().data()), typename Vector::value_type*>::value>> : std::true_type {};

// Add the buffer interface to a vector
template <typename Vector, typename Class_, typename... Args>
enable_if_t<detail::any_of<std::is_same<Args, buffer_protocol>...>::value>
vector_buffer(Class_& cl) {
    using T = typename Vector::value_type;

    static_assert(vector_has_data_and_format<Vector>::value, "There is not an appropriate format descriptor for this vector");

    // numpy.h declares this for arbitrary types, but it may raise an exception and crash hard at runtime if PYBIND11_NUMPY_DTYPE hasn't been called, so check here
    format_descriptor<T>::format();

    cl.def_buffer([](Vector& v) -> buffer_info {
        return buffer_info(v.data(), static_cast<ssize_t>(sizeof(T)), format_descriptor<T>::format(), 1, {v.size()}, {sizeof(T)});
    });

    cl.def(init([](buffer buf) {
        auto info = buf.request();
        if (info.ndim != 1 || info.strides[0] % static_cast<ssize_t>(sizeof(T)))
            throw type_error("Only valid 1D buffers can be copied to a vector");
        if (!detail::compare_buffer_info<T>::compare(info) || (ssize_t) sizeof(T) != info.itemsize)
            throw type_error("Format mismatch (Python: " + info.format + " C++: " + format_descriptor<T>::format() + ")");

        T *p = static_cast<T*>(info.ptr);
        ssize_t step = info.strides[0] / static_cast<ssize_t>(sizeof(T));
        T *end = p + info.shape[0] * step;
        if (step == 1) {
            return Vector(p, end);
        }
        else {
            Vector vec;
            vec.reserve((size_t) info.shape[0]);
            for (; p != end; p += step)
                vec.push_back(*p);
            return vec;
        }
    }));

    return;
}

template <typename Vector, typename Class_, typename... Args>
enable_if_t<!detail::any_of<std::is_same<Args, buffer_protocol>...>::value> vector_buffer(Class_&) {}

PYBIND11_NAMESPACE_END(detail)

//
// std::vector
//
template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
class_<Vector, holder_type> bind_vector(handle scope, std::string const &name, Args&&... args) {
    using Class_ = class_<Vector, holder_type>;

    // If the value_type is unregistered (e.g. a converting type) or is itself registered
    // module-local then make the vector binding module-local as well:
    using vtype = typename Vector::value_type;
    auto vtype_info = detail::get_type_info(typeid(vtype));
    bool local = !vtype_info || vtype_info->module_local;

    Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

    // Declare the buffer interface if a buffer_protocol() is passed in
    detail::vector_buffer<Vector, Class_, Args...>(cl);

    cl.def(init<>());

    // Register copy constructor (if possible)
    detail::vector_if_copy_constructible<Vector, Class_>(cl);

    // Register comparison-related operators and functions (if possible)
    detail::vector_if_equal_operator<Vector, Class_>(cl);

    // Register stream insertion operator (if possible)
    detail::vector_if_insertion_operator<Vector, Class_>(cl, name);

    // Modifiers require copyable vector value type
    detail::vector_modifiers<Vector, Class_>(cl);

    // Accessor and iterator; return by value if copyable, otherwise we return by ref + keep-alive
    detail::vector_accessor<Vector, Class_>(cl);

    cl.def("__bool__",
        [](const Vector &v) -> bool {
            return !v.empty();
        },
        "Check whether the list is nonempty"
    );

    cl.def("__len__", &Vector::size);




#if 0
    // C++ style functions deprecated, leaving it here as an example
    cl.def(init<size_type>());

    cl.def("resize",
         (void (Vector::*) (size_type count)) & Vector::resize,
         "changes the number of elements stored");

    cl.def("erase",
        [](Vector &v, SizeType i) {
        if (i >= v.size())
            throw index_error();
        v.erase(v.begin() + i);
    }, "erases element at index ``i``");

    cl.def("empty",         &Vector::empty,         "checks whether the container is empty");
    cl.def("size",          &Vector::size,          "returns the number of elements");
    cl.def("push_back", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
    cl.def("pop_back",                               &Vector::pop_back, "removes the last element");

    cl.def("max_size",      &Vector::max_size,      "returns the maximum possible number of elements");
    cl.def("reserve",       &Vector::reserve,       "reserves storage");
    cl.def("capacity",      &Vector::capacity,      "returns the number of elements that can be held in currently allocated storage");
    cl.def("shrink_to_fit", &Vector::shrink_to_fit, "reduces memory usage by freeing unused memory");

    cl.def("clear", &Vector::clear, "clears the contents");
    cl.def("swap",   &Vector::swap, "swaps the contents");

    cl.def("front", [](Vector &v) {
        if (v.size()) return v.front();
        else throw index_error();
    }, "access the first element");

    cl.def("back", [](Vector &v) {
        if (v.size()) return v.back();
        else throw index_error();
    }, "access the last element ");

#endif

    return cl;
}



//
// std::map, std::unordered_map
//

PYBIND11_NAMESPACE_BEGIN(detail)

/* Fallback functions */
template <typename, typename, typename... Args> void map_if_insertion_operator(const Args &...) { }
template <typename, typename, typename... Args> void map_assignment(const Args &...) { }

// Map assignment when copy-assignable: just copy the value
template <typename Map, typename Class_>
void map_assignment(enable_if_t<is_copy_assignable<typename Map::mapped_type>::value, Class_> &cl) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;

    cl.def("__setitem__",
           [](Map &m, const KeyType &k, const MappedType &v) {
               auto it = m.find(k);
               if (it != m.end()) it->second = v;
               else m.emplace(k, v);
           }
    );
}

// Not copy-assignable, but still copy-constructible: we can update the value by erasing and reinserting
template<typename Map, typename Class_>
void map_assignment(enable_if_t<
        !is_copy_assignable<typename Map::mapped_type>::value &&
        is_copy_constructible<typename Map::mapped_type>::value,
        Class_> &cl) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;

    cl.def("__setitem__",
           [](Map &m, const KeyType &k, const MappedType &v) {
               // We can't use m[k] = v; because value type might not be default constructable
               auto r = m.emplace(k, v);
               if (!r.second) {
                   // value type is not copy assignable so the only way to insert it is to erase it first...
                   m.erase(r.first);
                   m.emplace(k, v);
               }
           }
    );
}


template <typename Map, typename Class_> auto map_if_insertion_operator(Class_ &cl, std::string const &name)
-> decltype(std::declval<std::ostream&>() << std::declval<typename Map::key_type>() << std::declval<typename Map::mapped_type>(), void()) {

    cl.def("__repr__",
           [name](Map &m) {
            std::ostringstream s;
            s << name << '{';
            bool f = false;
            for (auto const &kv : m) {
                if (f)
                    s << ", ";
                s << kv.first << ": " << kv.second;
                f = true;
            }
            s << '}';
            return s.str();
        },
        "Return the canonical string representation of this map."
    );
}


PYBIND11_NAMESPACE_END(detail)

template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
class_<Map, holder_type> bind_map(handle scope, const std::string &name, Args&&... args) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;
    using Class_ = class_<Map, holder_type>;

    // If either type is a non-module-local bound type then make the map binding non-local as well;
    // otherwise (e.g. both types are either module-local or converting) the map will be
    // module-local.
    auto tinfo = detail::get_type_info(typeid(MappedType));
    bool local = !tinfo || tinfo->module_local;
    if (local) {
        tinfo = detail::get_type_info(typeid(KeyType));
        local = !tinfo || tinfo->module_local;
    }

    Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

    cl.def(init<>());

    // Register stream insertion operator (if possible)
    detail::map_if_insertion_operator<Map, Class_>(cl, name);

    cl.def("__bool__",
        [](const Map &m) -> bool { return !m.empty(); },
        "Check whether the map is nonempty"
    );

    cl.def("__iter__",
           [](Map &m) { return make_key_iterator(m.begin(), m.end()); },
           keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    cl.def("items",
           [](Map &m) { return make_iterator(m.begin(), m.end()); },
           keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    cl.def("__getitem__",
        [](Map &m, const KeyType &k) -> MappedType & {
            auto it = m.find(k);
            if (it == m.end())
              throw key_error();
           return it->second;
        },
        return_value_policy::reference_internal // ref + keepalive
    );

    cl.def("__contains__",
        [](Map &m, const KeyType &k) -> bool {
            auto it = m.find(k);
            if (it == m.end())
              return false;
           return true;
        }
    );

    // Assignment provided only if the type is copyable
    detail::map_assignment<Map, Class_>(cl);

    cl.def("__delitem__",
           [](Map &m, const KeyType &k) {
               auto it = m.find(k);
               if (it == m.end())
                   throw key_error();
               m.erase(it);
           }
    );

    cl.def("__len__", &Map::size);

    return cl;
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


//========= end of #include "stl_bind.h" ============


#endif  // __UNION_PYBIND11_HPP


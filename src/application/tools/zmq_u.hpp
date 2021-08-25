#pragma once
#ifndef __UNION_EMQ_HPP
#define __UNION_EMQ_HPP
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define ZMQ_STATIC

//========= begin of #include "platform.hpp" ============

#ifndef __ZMQ_PLATFORM_HPP_INCLUDED__
#define __ZMQ_PLATFORM_HPP_INCLUDED__

#ifdef _WIN32

  /* #undef ZMQ_USE_CV_IMPL_STL11 */
  #define ZMQ_USE_CV_IMPL_WIN32API
  /* #undef ZMQ_USE_CV_IMPL_PTHREADS */
  /* #undef ZMQ_USE_CV_IMPL_NONE */

  /* #undef ZMQ_IOTHREAD_POLLER_USE_KQUEUE */
  /* #undef ZMQ_IOTHREAD_POLLER_USE_EPOLL */
  /* #undef ZMQ_IOTHREAD_POLLER_USE_EPOLL_CLOEXEC */
  /* #undef ZMQ_IOTHREAD_POLLER_USE_DEVPOLL */
  /* #undef ZMQ_IOTHREAD_POLLER_USE_POLL */
  #define ZMQ_IOTHREAD_POLLER_USE_SELECT

  #define ZMQ_POLL_BASED_ON_SELECT
  /* #undef ZMQ_POLL_BASED_ON_POLL */

  #define ZMQ_CACHELINE_SIZE 64

  /* #undef ZMQ_FORCE_MUTEXES */

  /* #undef HAVE_FORK */
  /* #undef HAVE_CLOCK_GETTIME */
  /* #undef HAVE_GETHRTIME */
  /* #undef HAVE_MKDTEMP */
  /* #undef ZMQ_HAVE_UIO */

  #define ZMQ_HAVE_NOEXCEPT

  /* #undef ZMQ_HAVE_EVENTFD */
  /* #undef ZMQ_HAVE_EVENTFD_CLOEXEC */
  /* #undef ZMQ_HAVE_IFADDRS */
  /* #undef ZMQ_HAVE_SO_BINDTODEVICE */

  /* #undef ZMQ_HAVE_SO_PEERCRED */
  /* #undef ZMQ_HAVE_LOCAL_PEERCRED */

  /* #undef ZMQ_HAVE_O_CLOEXEC */

  /* #undef ZMQ_HAVE_SOCK_CLOEXEC */
  /* #undef ZMQ_HAVE_SO_KEEPALIVE */
  /* #undef ZMQ_HAVE_TCP_KEEPCNT */
  /* #undef ZMQ_HAVE_TCP_KEEPIDLE */
  /* #undef ZMQ_HAVE_TCP_KEEPINTVL */
  /* #undef ZMQ_HAVE_TCP_KEEPALIVE */
  /* #undef ZMQ_HAVE_PTHREAD_SETNAME_1 */
  /* #undef ZMQ_HAVE_PTHREAD_SETNAME_2 */
  /* #undef ZMQ_HAVE_PTHREAD_SETNAME_3 */
  /* #undef ZMQ_HAVE_PTHREAD_SET_NAME */
  /* #undef HAVE_ACCEPT4 */
  /* #undef HAVE_STRNLEN */

  /* #undef ZMQ_HAVE_OPENPGM */
  /* #undef ZMQ_MAKE_VALGRIND_HAPPY */

  #define ZMQ_HAVE_CURVE
  #define ZMQ_USE_TWEETNACL
  /* #undef ZMQ_USE_LIBSODIUM */
  /* #undef SODIUM_STATIC */

  #ifdef _AIX
    #define ZMQ_HAVE_AIX
  #endif

  #if defined __ANDROID__
    #define ZMQ_HAVE_ANDROID
  #endif

  #if defined __CYGWIN__
    #define ZMQ_HAVE_CYGWIN
  #endif

  #if defined __MINGW32__
    #define ZMQ_HAVE_MINGW32
  #endif

  #if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__FreeBSD_kernel__)
    #define ZMQ_HAVE_FREEBSD
  #endif

  #if defined __hpux
    #define ZMQ_HAVE_HPUX
  #endif

  #if defined __linux__
    #define ZMQ_HAVE_LINUX
  #endif

  #if defined __NetBSD__
    #define ZMQ_HAVE_NETBSD
  #endif

  #if defined __OpenBSD__
    #define ZMQ_HAVE_OPENBSD
  #endif

  #if defined __VMS
    #define ZMQ_HAVE_OPENVMS
  #endif

  #if defined __APPLE__
    #define ZMQ_HAVE_OSX
  #endif

  #if defined __QNXNTO__
    #define ZMQ_HAVE_QNXNTO
  #endif

  #if defined(sun) || defined(__sun)
    #define ZMQ_HAVE_SOLARIS
  #endif

  #define ZMQ_HAVE_WINDOWS
  /* #undef ZMQ_HAVE_WINDOWS_UWP */

#else   // if _WIN32 else


  #define ZMQ_USE_CV_IMPL_STL11
  /* #undef ZMQ_USE_CV_IMPL_WIN32API */
  /* #undef ZMQ_USE_CV_IMPL_PTHREADS */
  /* #undef ZMQ_USE_CV_IMPL_NONE */

  /* #undef ZMQ_IOTHREAD_POLLER_USE_KQUEUE */
  #define ZMQ_IOTHREAD_POLLER_USE_EPOLL
  #define ZMQ_IOTHREAD_POLLER_USE_EPOLL_CLOEXEC
  /* #undef ZMQ_IOTHREAD_POLLER_USE_DEVPOLL */
  /* #undef ZMQ_IOTHREAD_POLLER_USE_POLL */
  /* #undef ZMQ_IOTHREAD_POLLER_USE_SELECT */

  /* #undef ZMQ_POLL_BASED_ON_SELECT */
  #define ZMQ_POLL_BASED_ON_POLL

  #define ZMQ_CACHELINE_SIZE 64

  /* #undef ZMQ_FORCE_MUTEXES */

  #define HAVE_FORK
  #define HAVE_CLOCK_GETTIME
  /* #undef HAVE_GETHRTIME */
  #define HAVE_MKDTEMP
  #define ZMQ_HAVE_UIO

  #define ZMQ_HAVE_NOEXCEPT

  #define ZMQ_HAVE_EVENTFD
  #define ZMQ_HAVE_EVENTFD_CLOEXEC
  #define ZMQ_HAVE_IFADDRS
  #define ZMQ_HAVE_SO_BINDTODEVICE

  #define ZMQ_HAVE_SO_PEERCRED
  /* #undef ZMQ_HAVE_LOCAL_PEERCRED */

  #define ZMQ_HAVE_O_CLOEXEC

  #define ZMQ_HAVE_SOCK_CLOEXEC
  #define ZMQ_HAVE_SO_KEEPALIVE
  #define ZMQ_HAVE_TCP_KEEPCNT
  #define ZMQ_HAVE_TCP_KEEPIDLE
  #define ZMQ_HAVE_TCP_KEEPINTVL
  /* #undef ZMQ_HAVE_TCP_KEEPALIVE */
  /* #undef ZMQ_HAVE_PTHREAD_SETNAME_1 */
  #define ZMQ_HAVE_PTHREAD_SETNAME_2
  /* #undef ZMQ_HAVE_PTHREAD_SETNAME_3 */
  /* #undef ZMQ_HAVE_PTHREAD_SET_NAME */
  #define HAVE_ACCEPT4
  #define HAVE_STRNLEN

  /* #undef ZMQ_HAVE_OPENPGM */
  /* #undef ZMQ_MAKE_VALGRIND_HAPPY */

  #define ZMQ_HAVE_CURVE
  #define ZMQ_USE_TWEETNACL
  /* #undef ZMQ_USE_LIBSODIUM */
  /* #undef SODIUM_STATIC */

  #ifdef _AIX
    #define ZMQ_HAVE_AIX
  #endif

  #if defined __ANDROID__
    #define ZMQ_HAVE_ANDROID
  #endif

  #if defined __CYGWIN__
    #define ZMQ_HAVE_CYGWIN
  #endif

  #if defined __MINGW32__
    #define ZMQ_HAVE_MINGW32
  #endif

  #if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__FreeBSD_kernel__)
    #define ZMQ_HAVE_FREEBSD
  #endif

  #if defined __hpux
    #define ZMQ_HAVE_HPUX
  #endif

  #if defined __linux__
    #define ZMQ_HAVE_LINUX
  #endif

  #if defined __NetBSD__
    #define ZMQ_HAVE_NETBSD
  #endif

  #if defined __OpenBSD__
    #define ZMQ_HAVE_OPENBSD
  #endif

  #if defined __VMS
    #define ZMQ_HAVE_OPENVMS
  #endif

  #if defined __APPLE__
    #define ZMQ_HAVE_OSX
  #endif

  #if defined __QNXNTO__
    #define ZMQ_HAVE_QNXNTO
  #endif

  #if defined(sun) || defined(__sun)
    #define ZMQ_HAVE_SOLARIS
  #endif

  /* #undef ZMQ_HAVE_WINDOWS */
  /* #undef ZMQ_HAVE_WINDOWS_UWP */

#endif  // ifdefined _WIN32
#endif  // __ZMQ_PLATFORM_HPP_INCLUDED__


//========= end of #include "platform.hpp" ============


//========= begin of #include "windows.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef _WIN32
#ifndef __ZMQ_WINDOWS_HPP_INCLUDED__
#define __ZMQ_WINDOWS_HPP_INCLUDED__

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef NOMINMAX
#define NOMINMAX // Macros min(a,b) and max(a,b)
#endif

//  Set target version to Windows Server 2008, Windows Vista or higher.
//  Windows XP (0x0501) is supported but without client & server socket types.
#if !defined _WIN32_WINNT && !defined ZMQ_HAVE_WINDOWS_UWP
#define _WIN32_WINNT 0x0600
#endif

#if defined ZMQ_HAVE_WINDOWS_UWP
#define _WIN32_WINNT _WIN32_WINNT_WIN10
#endif

#ifdef __MINGW32__
//  Require Windows XP or higher with MinGW for getaddrinfo().
#if (_WIN32_WINNT >= 0x0501)
#else
#error You need at least Windows XP target
#endif
#endif

#include <winsock2.h>
#include <windows.h>
#include <mswsock.h>
#include <iphlpapi.h>

#if !defined __MINGW32__
#include <mstcpip.h>
#endif

//  Workaround missing mstcpip.h in mingw32 (MinGW64 provides this)
//  __MINGW64_VERSION_MAJOR is only defined when using in mingw-w64
#if defined __MINGW32__ && !defined SIO_KEEPALIVE_VALS                         \
  && !defined __MINGW64_VERSION_MAJOR
struct tcp_keepalive
{
    u_long onoff;
    u_long keepalivetime;
    u_long keepaliveinterval;
};
#define SIO_KEEPALIVE_VALS _WSAIOW (IOC_VENDOR, 4)
#endif

#include <ws2tcpip.h>
#include <ipexport.h>
#if !defined _WIN32_WCE
#include <process.h>
#endif

#if defined ZMQ_IOTHREAD_POLLER_USE_POLL || defined ZMQ_POLL_BASED_ON_POLL
static inline int poll (struct pollfd *pfd, unsigned long nfds, int timeout)
{
    return WSAPoll (pfd, nfds, timeout);
}
#endif

//  In MinGW environment AI_NUMERICSERV is not defined.
#ifndef AI_NUMERICSERV
#define AI_NUMERICSERV 0x0400
#endif
#endif

//  In MSVC prior to v14, snprintf is not available
//  The closest implementation is the _snprintf_s function
#if defined(_MSC_VER) && _MSC_VER < 1900
#define snprintf(buffer_, count_, format_, ...)                                \
    _snprintf_s (buffer_, count_, _TRUNCATE, format_, __VA_ARGS__)
#endif
#endif // #ifdef _WIN32

//========= end of #include "windows.hpp" ============


//========= begin of #include "zmq.h" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    *************************************************************************
    NOTE to contributors. This file comprises the principal public contract
    for ZeroMQ API users. Any change to this file supplied in a stable
    release SHOULD not break existing applications.
    In practice this means that the value of constants must not change, and
    that old values may not be reused for new constants.
    *************************************************************************
*/

#ifndef __ZMQ_H_INCLUDED__
#define __ZMQ_H_INCLUDED__

// ans ignore: #include "platform.hpp"

/*  Version macros for compile-time API version detection                     */
#define ZMQ_VERSION_MAJOR 4
#define ZMQ_VERSION_MINOR 3
#define ZMQ_VERSION_PATCH 2

#define ZMQ_MAKE_VERSION(major, minor, patch)                                  \
    ((major) *10000 + (minor) *100 + (patch))
#define ZMQ_VERSION                                                            \
    ZMQ_MAKE_VERSION (ZMQ_VERSION_MAJOR, ZMQ_VERSION_MINOR, ZMQ_VERSION_PATCH)

#ifdef __cplusplus
extern "C" {
#endif

#if !defined _WIN32_WCE
#include <errno.h>
#endif
#include <stddef.h>
#include <stdio.h>
#if defined _WIN32
//  Set target version to Windows Server 2008, Windows Vista or higher.
//  Windows XP (0x0501) is supported but without client & server socket types.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#ifdef __MINGW32__
//  Require Windows XP or higher with MinGW for getaddrinfo().
#if (_WIN32_WINNT >= 0x0501)
#else
#error You need at least Windows XP target
#endif
#endif
#include <winsock2.h>
#endif

/*  Handle DSO symbol visibility                                             */
#if defined _WIN32
#if defined ZMQ_STATIC
#define ZMQ_EXPORT
#elif defined DLL_EXPORT
#define ZMQ_EXPORT __declspec(dllexport)
#else
#define ZMQ_EXPORT __declspec(dllimport)
#endif
#else
#if defined __SUNPRO_C || defined __SUNPRO_CC
#define ZMQ_EXPORT __global
#elif (defined __GNUC__ && __GNUC__ >= 4) || defined __INTEL_COMPILER
#define ZMQ_EXPORT __attribute__ ((visibility ("default")))
#else
#define ZMQ_EXPORT
#endif
#endif

/*  Define integer types needed for event interface                          */
#define ZMQ_DEFINED_STDINT 1
#if defined ZMQ_HAVE_SOLARIS || defined ZMQ_HAVE_OPENVMS
#include <inttypes.h>
#elif defined _MSC_VER && _MSC_VER < 1600
#ifndef uint64_t
typedef unsigned __int64 uint64_t;
#endif
#ifndef int32_t
typedef __int32 int32_t;
#endif
#ifndef uint32_t
typedef unsigned __int32 uint32_t;
#endif
#ifndef uint16_t
typedef unsigned __int16 uint16_t;
#endif
#ifndef uint8_t
typedef unsigned __int8 uint8_t;
#endif
#else
#include <stdint.h>
#endif

//  32-bit AIX's pollfd struct members are called reqevents and rtnevents so it
//  defines compatibility macros for them. Need to include that header first to
//  stop build failures since zmq_pollset_t defines them as events and revents.
#ifdef ZMQ_HAVE_AIX
#include <poll.h>
#endif


/******************************************************************************/
/*  0MQ errors.                                                               */
/******************************************************************************/

/*  A number random enough not to collide with different errno ranges on      */
/*  different OSes. The assumption is that error_t is at least 32-bit type.   */
#define ZMQ_HAUSNUMERO 156384712

/*  On Windows platform some of the standard POSIX errnos are not defined.    */
#ifndef ENOTSUP
#define ENOTSUP (ZMQ_HAUSNUMERO + 1)
#endif
#ifndef EPROTONOSUPPORT
#define EPROTONOSUPPORT (ZMQ_HAUSNUMERO + 2)
#endif
#ifndef ENOBUFS
#define ENOBUFS (ZMQ_HAUSNUMERO + 3)
#endif
#ifndef ENETDOWN
#define ENETDOWN (ZMQ_HAUSNUMERO + 4)
#endif
#ifndef EADDRINUSE
#define EADDRINUSE (ZMQ_HAUSNUMERO + 5)
#endif
#ifndef EADDRNOTAVAIL
#define EADDRNOTAVAIL (ZMQ_HAUSNUMERO + 6)
#endif
#ifndef ECONNREFUSED
#define ECONNREFUSED (ZMQ_HAUSNUMERO + 7)
#endif
#ifndef EINPROGRESS
#define EINPROGRESS (ZMQ_HAUSNUMERO + 8)
#endif
#ifndef ENOTSOCK
#define ENOTSOCK (ZMQ_HAUSNUMERO + 9)
#endif
#ifndef EMSGSIZE
#define EMSGSIZE (ZMQ_HAUSNUMERO + 10)
#endif
#ifndef EAFNOSUPPORT
#define EAFNOSUPPORT (ZMQ_HAUSNUMERO + 11)
#endif
#ifndef ENETUNREACH
#define ENETUNREACH (ZMQ_HAUSNUMERO + 12)
#endif
#ifndef ECONNABORTED
#define ECONNABORTED (ZMQ_HAUSNUMERO + 13)
#endif
#ifndef ECONNRESET
#define ECONNRESET (ZMQ_HAUSNUMERO + 14)
#endif
#ifndef ENOTCONN
#define ENOTCONN (ZMQ_HAUSNUMERO + 15)
#endif
#ifndef ETIMEDOUT
#define ETIMEDOUT (ZMQ_HAUSNUMERO + 16)
#endif
#ifndef EHOSTUNREACH
#define EHOSTUNREACH (ZMQ_HAUSNUMERO + 17)
#endif
#ifndef ENETRESET
#define ENETRESET (ZMQ_HAUSNUMERO + 18)
#endif

/*  Native 0MQ error codes.                                                   */
#define EFSM (ZMQ_HAUSNUMERO + 51)
#define ENOCOMPATPROTO (ZMQ_HAUSNUMERO + 52)
#define ETERM (ZMQ_HAUSNUMERO + 53)
#define EMTHREAD (ZMQ_HAUSNUMERO + 54)

/*  This function retrieves the errno as it is known to 0MQ library. The goal */
/*  of this function is to make the code 100% portable, including where 0MQ   */
/*  compiled with certain CRT library (on Windows) is linked to an            */
/*  application that uses different CRT library.                              */
ZMQ_EXPORT int zmq_errno (void);

/*  Resolves system errors and 0MQ errors to human-readable string.           */
ZMQ_EXPORT const char *zmq_strerror (int errnum_);

/*  Run-time API version detection                                            */
ZMQ_EXPORT void zmq_version (int *major_, int *minor_, int *patch_);

/******************************************************************************/
/*  0MQ infrastructure (a.k.a. context) initialisation & termination.         */
/******************************************************************************/

/*  Context options                                                           */
#define ZMQ_IO_THREADS 1
#define ZMQ_MAX_SOCKETS 2
#define ZMQ_SOCKET_LIMIT 3
#define ZMQ_THREAD_PRIORITY 3
#define ZMQ_THREAD_SCHED_POLICY 4
#define ZMQ_MAX_MSGSZ 5
#define ZMQ_MSG_T_SIZE 6
#define ZMQ_THREAD_AFFINITY_CPU_ADD 7
#define ZMQ_THREAD_AFFINITY_CPU_REMOVE 8
#define ZMQ_THREAD_NAME_PREFIX 9

/*  Default for new contexts                                                  */
#define ZMQ_IO_THREADS_DFLT 1
#define ZMQ_MAX_SOCKETS_DFLT 1023
#define ZMQ_THREAD_PRIORITY_DFLT -1
#define ZMQ_THREAD_SCHED_POLICY_DFLT -1

ZMQ_EXPORT void *zmq_ctx_new (void);
ZMQ_EXPORT int zmq_ctx_term (void *context_);
ZMQ_EXPORT int zmq_ctx_shutdown (void *context_);
ZMQ_EXPORT int zmq_ctx_set (void *context_, int option_, int optval_);
ZMQ_EXPORT int zmq_ctx_get (void *context_, int option_);

/*  Old (legacy) API                                                          */
ZMQ_EXPORT void *zmq_init (int io_threads_);
ZMQ_EXPORT int zmq_term (void *context_);
ZMQ_EXPORT int zmq_ctx_destroy (void *context_);


/******************************************************************************/
/*  0MQ message definition.                                                   */
/******************************************************************************/

/* Some architectures, like sparc64 and some variants of aarch64, enforce pointer
 * alignment and raise sigbus on violations. Make sure applications allocate
 * zmq_msg_t on addresses aligned on a pointer-size boundary to avoid this issue.
 */
typedef struct zmq_msg_t
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    __declspec(align (8)) unsigned char _[64];
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_ARM_ARMV7VE))
    __declspec(align (4)) unsigned char _[64];
#elif defined(__GNUC__) || defined(__INTEL_COMPILER)                           \
  || (defined(__SUNPRO_C) && __SUNPRO_C >= 0x590)                              \
  || (defined(__SUNPRO_CC) && __SUNPRO_CC >= 0x590)
    unsigned char _[64] __attribute__ ((aligned (sizeof (void *))));
#else
    unsigned char _[64];
#endif
} zmq_msg_t;

typedef void(zmq_free_fn) (void *data_, void *hint_);

ZMQ_EXPORT int zmq_msg_init (zmq_msg_t *msg_);
ZMQ_EXPORT int zmq_msg_init_size (zmq_msg_t *msg_, size_t size_);
ZMQ_EXPORT int zmq_msg_init_data (
  zmq_msg_t *msg_, void *data_, size_t size_, zmq_free_fn *ffn_, void *hint_);
ZMQ_EXPORT int zmq_msg_send (zmq_msg_t *msg_, void *s_, int flags_);
ZMQ_EXPORT int zmq_msg_recv (zmq_msg_t *msg_, void *s_, int flags_);
ZMQ_EXPORT int zmq_msg_close (zmq_msg_t *msg_);
ZMQ_EXPORT int zmq_msg_move (zmq_msg_t *dest_, zmq_msg_t *src_);
ZMQ_EXPORT int zmq_msg_copy (zmq_msg_t *dest_, zmq_msg_t *src_);
ZMQ_EXPORT void *zmq_msg_data (zmq_msg_t *msg_);
ZMQ_EXPORT size_t zmq_msg_size (const zmq_msg_t *msg_);
ZMQ_EXPORT int zmq_msg_more (const zmq_msg_t *msg_);
ZMQ_EXPORT int zmq_msg_get (const zmq_msg_t *msg_, int property_);
ZMQ_EXPORT int zmq_msg_set (zmq_msg_t *msg_, int property_, int optval_);
ZMQ_EXPORT const char *zmq_msg_gets (const zmq_msg_t *msg_,
                                     const char *property_);

/******************************************************************************/
/*  0MQ socket definition.                                                    */
/******************************************************************************/

/*  Socket types.                                                             */
#define ZMQ_PAIR 0
#define ZMQ_PUB 1
#define ZMQ_SUB 2
#define ZMQ_REQ 3
#define ZMQ_REP 4
#define ZMQ_DEALER 5
#define ZMQ_ROUTER 6
#define ZMQ_PULL 7
#define ZMQ_PUSH 8
#define ZMQ_XPUB 9
#define ZMQ_XSUB 10
#define ZMQ_STREAM 11

/*  Deprecated aliases                                                        */
#define ZMQ_XREQ ZMQ_DEALER
#define ZMQ_XREP ZMQ_ROUTER

/*  Socket options.                                                           */
#define ZMQ_AFFINITY 4
#define ZMQ_ROUTING_ID 5
#define ZMQ_SUBSCRIBE 6
#define ZMQ_UNSUBSCRIBE 7
#define ZMQ_RATE 8
#define ZMQ_RECOVERY_IVL 9
#define ZMQ_SNDBUF 11
#define ZMQ_RCVBUF 12
#define ZMQ_RCVMORE 13
#define ZMQ_FD 14
#define ZMQ_EVENTS 15
#define ZMQ_TYPE 16
#define ZMQ_LINGER 17
#define ZMQ_RECONNECT_IVL 18
#define ZMQ_BACKLOG 19
#define ZMQ_RECONNECT_IVL_MAX 21
#define ZMQ_MAXMSGSIZE 22
#define ZMQ_SNDHWM 23
#define ZMQ_RCVHWM 24
#define ZMQ_MULTICAST_HOPS 25
#define ZMQ_RCVTIMEO 27
#define ZMQ_SNDTIMEO 28
#define ZMQ_LAST_ENDPOINT 32
#define ZMQ_ROUTER_MANDATORY 33
#define ZMQ_TCP_KEEPALIVE 34
#define ZMQ_TCP_KEEPALIVE_CNT 35
#define ZMQ_TCP_KEEPALIVE_IDLE 36
#define ZMQ_TCP_KEEPALIVE_INTVL 37
#define ZMQ_IMMEDIATE 39
#define ZMQ_XPUB_VERBOSE 40
#define ZMQ_ROUTER_RAW 41
#define ZMQ_IPV6 42
#define ZMQ_MECHANISM 43
#define ZMQ_PLAIN_SERVER 44
#define ZMQ_PLAIN_USERNAME 45
#define ZMQ_PLAIN_PASSWORD 46
#define ZMQ_CURVE_SERVER 47
#define ZMQ_CURVE_PUBLICKEY 48
#define ZMQ_CURVE_SECRETKEY 49
#define ZMQ_CURVE_SERVERKEY 50
#define ZMQ_PROBE_ROUTER 51
#define ZMQ_REQ_CORRELATE 52
#define ZMQ_REQ_RELAXED 53
#define ZMQ_CONFLATE 54
#define ZMQ_ZAP_DOMAIN 55
#define ZMQ_ROUTER_HANDOVER 56
#define ZMQ_TOS 57
#define ZMQ_CONNECT_ROUTING_ID 61
#define ZMQ_GSSAPI_SERVER 62
#define ZMQ_GSSAPI_PRINCIPAL 63
#define ZMQ_GSSAPI_SERVICE_PRINCIPAL 64
#define ZMQ_GSSAPI_PLAINTEXT 65
#define ZMQ_HANDSHAKE_IVL 66
#define ZMQ_SOCKS_PROXY 68
#define ZMQ_XPUB_NODROP 69
#define ZMQ_BLOCKY 70
#define ZMQ_XPUB_MANUAL 71
#define ZMQ_XPUB_WELCOME_MSG 72
#define ZMQ_STREAM_NOTIFY 73
#define ZMQ_INVERT_MATCHING 74
#define ZMQ_HEARTBEAT_IVL 75
#define ZMQ_HEARTBEAT_TTL 76
#define ZMQ_HEARTBEAT_TIMEOUT 77
#define ZMQ_XPUB_VERBOSER 78
#define ZMQ_CONNECT_TIMEOUT 79
#define ZMQ_TCP_MAXRT 80
#define ZMQ_THREAD_SAFE 81
#define ZMQ_MULTICAST_MAXTPDU 84
#define ZMQ_VMCI_BUFFER_SIZE 85
#define ZMQ_VMCI_BUFFER_MIN_SIZE 86
#define ZMQ_VMCI_BUFFER_MAX_SIZE 87
#define ZMQ_VMCI_CONNECT_TIMEOUT 88
#define ZMQ_USE_FD 89
#define ZMQ_GSSAPI_PRINCIPAL_NAMETYPE 90
#define ZMQ_GSSAPI_SERVICE_PRINCIPAL_NAMETYPE 91
#define ZMQ_BINDTODEVICE 92

/*  Message options                                                           */
#define ZMQ_MORE 1
#define ZMQ_SHARED 3

/*  Send/recv options.                                                        */
#define ZMQ_DONTWAIT 1
#define ZMQ_SNDMORE 2

/*  Security mechanisms                                                       */
#define ZMQ_NULL 0
#define ZMQ_PLAIN 1
#define ZMQ_CURVE 2
#define ZMQ_GSSAPI 3

/*  RADIO-DISH protocol                                                       */
#define ZMQ_GROUP_MAX_LENGTH 15

/*  Deprecated options and aliases                                            */
#define ZMQ_IDENTITY ZMQ_ROUTING_ID
#define ZMQ_CONNECT_RID ZMQ_CONNECT_ROUTING_ID
#define ZMQ_TCP_ACCEPT_FILTER 38
#define ZMQ_IPC_FILTER_PID 58
#define ZMQ_IPC_FILTER_UID 59
#define ZMQ_IPC_FILTER_GID 60
#define ZMQ_IPV4ONLY 31
#define ZMQ_DELAY_ATTACH_ON_CONNECT ZMQ_IMMEDIATE
#define ZMQ_NOBLOCK ZMQ_DONTWAIT
#define ZMQ_FAIL_UNROUTABLE ZMQ_ROUTER_MANDATORY
#define ZMQ_ROUTER_BEHAVIOR ZMQ_ROUTER_MANDATORY

/*  Deprecated Message options                                                */
#define ZMQ_SRCFD 2

/******************************************************************************/
/*  GSSAPI definitions                                                        */
/******************************************************************************/

/*  GSSAPI principal name types                                               */
#define ZMQ_GSSAPI_NT_HOSTBASED 0
#define ZMQ_GSSAPI_NT_USER_NAME 1
#define ZMQ_GSSAPI_NT_KRB5_PRINCIPAL 2

/******************************************************************************/
/*  0MQ socket events and monitoring                                          */
/******************************************************************************/

/*  Socket transport events (TCP, IPC and TIPC only)                          */

#define ZMQ_EVENT_CONNECTED 0x0001
#define ZMQ_EVENT_CONNECT_DELAYED 0x0002
#define ZMQ_EVENT_CONNECT_RETRIED 0x0004
#define ZMQ_EVENT_LISTENING 0x0008
#define ZMQ_EVENT_BIND_FAILED 0x0010
#define ZMQ_EVENT_ACCEPTED 0x0020
#define ZMQ_EVENT_ACCEPT_FAILED 0x0040
#define ZMQ_EVENT_CLOSED 0x0080
#define ZMQ_EVENT_CLOSE_FAILED 0x0100
#define ZMQ_EVENT_DISCONNECTED 0x0200
#define ZMQ_EVENT_MONITOR_STOPPED 0x0400
#define ZMQ_EVENT_ALL 0xFFFF
/*  Unspecified system errors during handshake. Event value is an errno.      */
#define ZMQ_EVENT_HANDSHAKE_FAILED_NO_DETAIL 0x0800
/*  Handshake complete successfully with successful authentication (if        *
 *  enabled). Event value is unused.                                          */
#define ZMQ_EVENT_HANDSHAKE_SUCCEEDED 0x1000
/*  Protocol errors between ZMTP peers or between server and ZAP handler.     *
 *  Event value is one of ZMQ_PROTOCOL_ERROR_*                                */
#define ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL 0x2000
/*  Failed authentication requests. Event value is the numeric ZAP status     *
 *  code, i.e. 300, 400 or 500.                                               */
#define ZMQ_EVENT_HANDSHAKE_FAILED_AUTH 0x4000
#define ZMQ_PROTOCOL_ERROR_ZMTP_UNSPECIFIED 0x10000000
#define ZMQ_PROTOCOL_ERROR_ZMTP_UNEXPECTED_COMMAND 0x10000001
#define ZMQ_PROTOCOL_ERROR_ZMTP_INVALID_SEQUENCE 0x10000002
#define ZMQ_PROTOCOL_ERROR_ZMTP_KEY_EXCHANGE 0x10000003
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED 0x10000011
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE 0x10000012
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_HELLO 0x10000013
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_INITIATE 0x10000014
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_ERROR 0x10000015
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_READY 0x10000016
#define ZMQ_PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_WELCOME 0x10000017
#define ZMQ_PROTOCOL_ERROR_ZMTP_INVALID_METADATA 0x10000018
// the following two may be due to erroneous configuration of a peer
#define ZMQ_PROTOCOL_ERROR_ZMTP_CRYPTOGRAPHIC 0x11000001
#define ZMQ_PROTOCOL_ERROR_ZMTP_MECHANISM_MISMATCH 0x11000002
#define ZMQ_PROTOCOL_ERROR_ZAP_UNSPECIFIED 0x20000000
#define ZMQ_PROTOCOL_ERROR_ZAP_MALFORMED_REPLY 0x20000001
#define ZMQ_PROTOCOL_ERROR_ZAP_BAD_REQUEST_ID 0x20000002
#define ZMQ_PROTOCOL_ERROR_ZAP_BAD_VERSION 0x20000003
#define ZMQ_PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE 0x20000004
#define ZMQ_PROTOCOL_ERROR_ZAP_INVALID_METADATA 0x20000005

ZMQ_EXPORT void *zmq_socket (void *, int type_);
ZMQ_EXPORT int zmq_close (void *s_);
ZMQ_EXPORT int
zmq_setsockopt (void *s_, int option_, const void *optval_, size_t optvallen_);
ZMQ_EXPORT int
zmq_getsockopt (void *s_, int option_, void *optval_, size_t *optvallen_);
ZMQ_EXPORT int zmq_bind (void *s_, const char *addr_);
ZMQ_EXPORT int zmq_connect (void *s_, const char *addr_);
ZMQ_EXPORT int zmq_unbind (void *s_, const char *addr_);
ZMQ_EXPORT int zmq_disconnect (void *s_, const char *addr_);
ZMQ_EXPORT int zmq_send (void *s_, const void *buf_, size_t len_, int flags_);
ZMQ_EXPORT int
zmq_send_const (void *s_, const void *buf_, size_t len_, int flags_);
ZMQ_EXPORT int zmq_recv (void *s_, void *buf_, size_t len_, int flags_);
ZMQ_EXPORT int zmq_socket_monitor (void *s_, const char *addr_, int events_);


/******************************************************************************/
/*  Deprecated I/O multiplexing. Prefer using zmq_poller API                  */
/******************************************************************************/

#define ZMQ_POLLIN 1
#define ZMQ_POLLOUT 2
#define ZMQ_POLLERR 4
#define ZMQ_POLLPRI 8

typedef struct zmq_pollitem_t
{
    void *socket;
#if defined _WIN32
    SOCKET fd;
#else
    int fd;
#endif
    short events;
    short revents;
} zmq_pollitem_t;

#define ZMQ_POLLITEMS_DFLT 16

ZMQ_EXPORT int zmq_poll (zmq_pollitem_t *items_, int nitems_, long timeout_);

/******************************************************************************/
/*  Message proxying                                                          */
/******************************************************************************/

ZMQ_EXPORT int zmq_proxy (void *frontend_, void *backend_, void *capture_);
ZMQ_EXPORT int zmq_proxy_steerable (void *frontend_,
                                    void *backend_,
                                    void *capture_,
                                    void *control_);

/******************************************************************************/
/*  Probe library capabilities                                                */
/******************************************************************************/

#define ZMQ_HAS_CAPABILITIES 1
ZMQ_EXPORT int zmq_has (const char *capability_);

/*  Deprecated aliases */
#define ZMQ_STREAMER 1
#define ZMQ_FORWARDER 2
#define ZMQ_QUEUE 3

/*  Deprecated methods */
ZMQ_EXPORT int zmq_device (int type_, void *frontend_, void *backend_);
ZMQ_EXPORT int zmq_sendmsg (void *s_, zmq_msg_t *msg_, int flags_);
ZMQ_EXPORT int zmq_recvmsg (void *s_, zmq_msg_t *msg_, int flags_);
struct iovec;
ZMQ_EXPORT int
zmq_sendiov (void *s_, struct iovec *iov_, size_t count_, int flags_);
ZMQ_EXPORT int
zmq_recviov (void *s_, struct iovec *iov_, size_t *count_, int flags_);

/******************************************************************************/
/*  Encryption functions                                                      */
/******************************************************************************/

/*  Encode data with Z85 encoding. Returns encoded data                       */
ZMQ_EXPORT char *
zmq_z85_encode (char *dest_, const uint8_t *data_, size_t size_);

/*  Decode data with Z85 encoding. Returns decoded data                       */
ZMQ_EXPORT uint8_t *zmq_z85_decode (uint8_t *dest_, const char *string_);

/*  Generate z85-encoded public and private keypair with tweetnacl/libsodium. */
/*  Returns 0 on success.                                                     */
ZMQ_EXPORT int zmq_curve_keypair (char *z85_public_key_, char *z85_secret_key_);

/*  Derive the z85-encoded public key from the z85-encoded secret key.        */
/*  Returns 0 on success.                                                     */
ZMQ_EXPORT int zmq_curve_public (char *z85_public_key_,
                                 const char *z85_secret_key_);

/******************************************************************************/
/*  Atomic utility methods                                                    */
/******************************************************************************/

ZMQ_EXPORT void *zmq_atomic_counter_new (void);
ZMQ_EXPORT void zmq_atomic_counter_set (void *counter_, int value_);
ZMQ_EXPORT int zmq_atomic_counter_inc (void *counter_);
ZMQ_EXPORT int zmq_atomic_counter_dec (void *counter_);
ZMQ_EXPORT int zmq_atomic_counter_value (void *counter_);
ZMQ_EXPORT void zmq_atomic_counter_destroy (void **counter_p_);

/******************************************************************************/
/*  Scheduling timers                                                         */
/******************************************************************************/

#define ZMQ_HAVE_TIMERS

typedef void(zmq_timer_fn) (int timer_id, void *arg);

ZMQ_EXPORT void *zmq_timers_new (void);
ZMQ_EXPORT int zmq_timers_destroy (void **timers_p);
ZMQ_EXPORT int
zmq_timers_add (void *timers, size_t interval, zmq_timer_fn handler, void *arg);
ZMQ_EXPORT int zmq_timers_cancel (void *timers, int timer_id);
ZMQ_EXPORT int
zmq_timers_set_interval (void *timers, int timer_id, size_t interval);
ZMQ_EXPORT int zmq_timers_reset (void *timers, int timer_id);
ZMQ_EXPORT long zmq_timers_timeout (void *timers);
ZMQ_EXPORT int zmq_timers_execute (void *timers);


/******************************************************************************/
/*  These functions are not documented by man pages -- use at your own risk.  */
/*  If you need these to be part of the formal ZMQ API, then (a) write a man  */
/*  page, and (b) write a test case in tests.                                 */
/******************************************************************************/

/*  Helper functions are used by perf tests so that they don't have to care   */
/*  about minutiae of time-related functions on different OS platforms.       */

/*  Starts the stopwatch. Returns the handle to the watch.                    */
ZMQ_EXPORT void *zmq_stopwatch_start (void);

/*  Returns the number of microseconds elapsed since the stopwatch was        */
/*  started, but does not stop or deallocate the stopwatch.                   */
ZMQ_EXPORT unsigned long zmq_stopwatch_intermediate (void *watch_);

/*  Stops the stopwatch. Returns the number of microseconds elapsed since     */
/*  the stopwatch was started, and deallocates that watch.                    */
ZMQ_EXPORT unsigned long zmq_stopwatch_stop (void *watch_);

/*  Sleeps for specified number of seconds.                                   */
ZMQ_EXPORT void zmq_sleep (int seconds_);

typedef void(zmq_thread_fn) (void *);

/* Start a thread. Returns a handle to the thread.                            */
ZMQ_EXPORT void *zmq_threadstart (zmq_thread_fn *func_, void *arg_);

/* Wait for thread to complete then free up resources.                        */
ZMQ_EXPORT void zmq_threadclose (void *thread_);


/******************************************************************************/
/*  These functions are DRAFT and disabled in stable releases, and subject to */
/*  change at ANY time until declared stable.                                 */
/******************************************************************************/

#ifdef ZMQ_BUILD_DRAFT_API

/*  DRAFT Socket types.                                                       */
#define ZMQ_SERVER 12
#define ZMQ_CLIENT 13
#define ZMQ_RADIO 14
#define ZMQ_DISH 15
#define ZMQ_GATHER 16
#define ZMQ_SCATTER 17
#define ZMQ_DGRAM 18

/*  DRAFT Socket options.                                                     */
#define ZMQ_ZAP_ENFORCE_DOMAIN 93
#define ZMQ_LOOPBACK_FASTPATH 94
#define ZMQ_METADATA 95
#define ZMQ_MULTICAST_LOOP 96
#define ZMQ_ROUTER_NOTIFY 97
#define ZMQ_XPUB_MANUAL_LAST_VALUE 98
#define ZMQ_SOCKS_USERNAME 99
#define ZMQ_SOCKS_PASSWORD 100
#define ZMQ_IN_BATCH_SIZE 101
#define ZMQ_OUT_BATCH_SIZE 102

/*  DRAFT Context options                                                     */
#define ZMQ_ZERO_COPY_RECV 10

/*  DRAFT Socket methods.                                                     */
ZMQ_EXPORT int zmq_join (void *s, const char *group);
ZMQ_EXPORT int zmq_leave (void *s, const char *group);

/*  DRAFT Msg methods.                                                        */
ZMQ_EXPORT int zmq_msg_set_routing_id (zmq_msg_t *msg, uint32_t routing_id);
ZMQ_EXPORT uint32_t zmq_msg_routing_id (zmq_msg_t *msg);
ZMQ_EXPORT int zmq_msg_set_group (zmq_msg_t *msg, const char *group);
ZMQ_EXPORT const char *zmq_msg_group (zmq_msg_t *msg);

/*  DRAFT Msg property names.                                                 */
#define ZMQ_MSG_PROPERTY_ROUTING_ID "Routing-Id"
#define ZMQ_MSG_PROPERTY_SOCKET_TYPE "Socket-Type"
#define ZMQ_MSG_PROPERTY_USER_ID "User-Id"
#define ZMQ_MSG_PROPERTY_PEER_ADDRESS "Peer-Address"

/*  Router notify options                                                     */
#define ZMQ_NOTIFY_CONNECT 1
#define ZMQ_NOTIFY_DISCONNECT 2

/******************************************************************************/
/*  Poller polling on sockets,fd and thread-safe sockets                      */
/******************************************************************************/

#define ZMQ_HAVE_POLLER

#if defined _WIN32
typedef SOCKET zmq_fd_t;
#else
typedef int zmq_fd_t;
#endif

typedef struct zmq_poller_event_t
{
    void *socket;
    zmq_fd_t fd;
    void *user_data;
    short events;
} zmq_poller_event_t;

ZMQ_EXPORT void *zmq_poller_new (void);
ZMQ_EXPORT int zmq_poller_destroy (void **poller_p);
ZMQ_EXPORT int
zmq_poller_add (void *poller, void *socket, void *user_data, short events);
ZMQ_EXPORT int zmq_poller_modify (void *poller, void *socket, short events);
ZMQ_EXPORT int zmq_poller_remove (void *poller, void *socket);
ZMQ_EXPORT int
zmq_poller_wait (void *poller, zmq_poller_event_t *event, long timeout);
ZMQ_EXPORT int zmq_poller_wait_all (void *poller,
                                    zmq_poller_event_t *events,
                                    int n_events,
                                    long timeout);
ZMQ_EXPORT int zmq_poller_fd (void *poller, zmq_fd_t *fd);

ZMQ_EXPORT int
zmq_poller_add_fd (void *poller, zmq_fd_t fd, void *user_data, short events);
ZMQ_EXPORT int zmq_poller_modify_fd (void *poller, zmq_fd_t fd, short events);
ZMQ_EXPORT int zmq_poller_remove_fd (void *poller, zmq_fd_t fd);

ZMQ_EXPORT int zmq_socket_get_peer_state (void *socket,
                                          const void *routing_id,
                                          size_t routing_id_size);

/*  DRAFT Socket monitoring events                                            */
#define ZMQ_EVENT_PIPES_STATS 0x10000

#define ZMQ_CURRENT_EVENT_VERSION 1
#define ZMQ_CURRENT_EVENT_VERSION_DRAFT 2

#define ZMQ_EVENT_ALL_V1 ZMQ_EVENT_ALL
#define ZMQ_EVENT_ALL_V2 ZMQ_EVENT_ALL_V1 | ZMQ_EVENT_PIPES_STATS

ZMQ_EXPORT int zmq_socket_monitor_versioned (
  void *s_, const char *addr_, uint64_t events_, int event_version_, int type_);
ZMQ_EXPORT int zmq_socket_monitor_pipes_stats (void *s);

#endif // ZMQ_BUILD_DRAFT_API


#undef ZMQ_EXPORT

#ifdef __cplusplus
}
#endif

#endif


//========= end of #include "zmq.h" ============


//========= begin of #include "zmq_draft.h" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DRAFT_H_INCLUDED__
#define __ZMQ_DRAFT_H_INCLUDED__

/******************************************************************************/
/*  These functions are DRAFT and disabled in stable releases, and subject to */
/*  change at ANY time until declared stable.                                 */
/******************************************************************************/
// ans ignore: #include "zmq.h"

#ifndef ZMQ_BUILD_DRAFT_API

/*  DRAFT Socket types.                                                       */
#define ZMQ_SERVER 12
#define ZMQ_CLIENT 13
#define ZMQ_RADIO 14
#define ZMQ_DISH 15
#define ZMQ_GATHER 16
#define ZMQ_SCATTER 17
#define ZMQ_DGRAM 18

/*  DRAFT Socket options.                                                     */
#define ZMQ_ZAP_ENFORCE_DOMAIN 93
#define ZMQ_LOOPBACK_FASTPATH 94
#define ZMQ_METADATA 95
#define ZMQ_MULTICAST_LOOP 96
#define ZMQ_ROUTER_NOTIFY 97
#define ZMQ_XPUB_MANUAL_LAST_VALUE 98
#define ZMQ_SOCKS_USERNAME 99
#define ZMQ_SOCKS_PASSWORD 100
#define ZMQ_IN_BATCH_SIZE 101
#define ZMQ_OUT_BATCH_SIZE 102

/*  DRAFT Context options                                                     */
#define ZMQ_ZERO_COPY_RECV 10

/*  DRAFT Socket methods.                                                     */
int zmq_join (void *s_, const char *group_);
int zmq_leave (void *s_, const char *group_);

/*  DRAFT Msg methods.                                                        */
int zmq_msg_set_routing_id (zmq_msg_t *msg_, uint32_t routing_id_);
uint32_t zmq_msg_routing_id (zmq_msg_t *msg_);
int zmq_msg_set_group (zmq_msg_t *msg_, const char *group_);
const char *zmq_msg_group (zmq_msg_t *msg_);

/*  DRAFT Msg property names.                                                 */
#define ZMQ_MSG_PROPERTY_ROUTING_ID "Routing-Id"
#define ZMQ_MSG_PROPERTY_SOCKET_TYPE "Socket-Type"
#define ZMQ_MSG_PROPERTY_USER_ID "User-Id"
#define ZMQ_MSG_PROPERTY_PEER_ADDRESS "Peer-Address"

/*  Router notify options                                                     */
#define ZMQ_NOTIFY_CONNECT 1
#define ZMQ_NOTIFY_DISCONNECT 2

/******************************************************************************/
/*  Poller polling on sockets,fd and thread-safe sockets                      */
/******************************************************************************/

#if defined _WIN32
typedef SOCKET zmq_fd_t;
#else
typedef int zmq_fd_t;
#endif

typedef struct zmq_poller_event_t
{
    void *socket;
    zmq_fd_t fd;
    void *user_data;
    short events;
} zmq_poller_event_t;

void *zmq_poller_new (void);
int zmq_poller_destroy (void **poller_p_);
int zmq_poller_add (void *poller_,
                    void *socket_,
                    void *user_data_,
                    short events_);
int zmq_poller_modify (void *poller_, void *socket_, short events_);
int zmq_poller_remove (void *poller_, void *socket_);
int zmq_poller_wait (void *poller_, zmq_poller_event_t *event_, long timeout_);
int zmq_poller_wait_all (void *poller_,
                         zmq_poller_event_t *events_,
                         int n_events_,
                         long timeout_);
zmq_fd_t zmq_poller_fd (void *poller_);

int zmq_poller_add_fd (void *poller_,
                       zmq_fd_t fd_,
                       void *user_data_,
                       short events_);
int zmq_poller_modify_fd (void *poller_, zmq_fd_t fd_, short events_);
int zmq_poller_remove_fd (void *poller_, zmq_fd_t fd_);

int zmq_socket_get_peer_state (void *socket_,
                               const void *routing_id_,
                               size_t routing_id_size_);

/*  DRAFT Socket monitoring events                                            */
#define ZMQ_EVENT_PIPES_STATS 0x10000

#define ZMQ_CURRENT_EVENT_VERSION 1
#define ZMQ_CURRENT_EVENT_VERSION_DRAFT 2

#define ZMQ_EVENT_ALL_V1 ZMQ_EVENT_ALL
#define ZMQ_EVENT_ALL_V2 ZMQ_EVENT_ALL_V1 | ZMQ_EVENT_PIPES_STATS

int zmq_socket_monitor_versioned (
  void *s_, const char *addr_, uint64_t events_, int event_version_, int type_);
int zmq_socket_monitor_pipes_stats (void *s_);

#endif // ZMQ_BUILD_DRAFT_API

#endif //ifndef __ZMQ_DRAFT_H_INCLUDED__


//========= end of #include "zmq_draft.h" ============


//========= begin of #include "likely.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_LIKELY_HPP_INCLUDED__
#define __ZMQ_LIKELY_HPP_INCLUDED__

#if defined __GNUC__
#define likely(x) __builtin_expect ((x), 1)
#define unlikely(x) __builtin_expect ((x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif


#endif


//========= end of #include "likely.hpp" ============


//========= begin of #include "err.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ERR_HPP_INCLUDED__
#define __ZMQ_ERR_HPP_INCLUDED__

#include <assert.h>
#if defined _WIN32_WCE
// ans ignore: #include "..\builds\msvc\errno.hpp"
#else
#include <errno.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef ZMQ_HAVE_WINDOWS
#include <netdb.h>
#endif

// ans ignore: #include "likely.hpp"

//  0MQ-specific error codes are defined in zmq.h

// EPROTO is not used by OpenBSD and maybe other platforms.
#ifndef EPROTO
#define EPROTO 0
#endif

namespace zmq
{
const char *errno_to_string (int errno_);
#if defined __clang__
#if __has_feature(attribute_analyzer_noreturn)
void zmq_abort (const char *errmsg_) __attribute__ ((analyzer_noreturn));
#endif
#elif defined __MSCVER__
__declspec(noreturn) void zmq_abort (const char *errmsg_);
#else
void zmq_abort (const char *errmsg_);
#endif
void print_backtrace ();
}

#ifdef ZMQ_HAVE_WINDOWS

namespace zmq
{
const char *wsa_error ();
const char *
wsa_error_no (int no_,
              const char *wsae_wouldblock_string_ = "Operation would block");
void win_error (char *buffer_, size_t buffer_size_);
int wsa_error_to_errno (int errcode_);
}

//  Provides convenient way to check WSA-style errors on Windows.
#define wsa_assert(x)                                                          \
    do {                                                                       \
        if (unlikely (!(x))) {                                                 \
            const char *errstr = zmq::wsa_error ();                            \
            if (errstr != NULL) {                                              \
                fprintf (stderr, "Assertion failed: %s [%i] (%s:%d)\n",        \
                         errstr, WSAGetLastError (), __FILE__, __LINE__);      \
                fflush (stderr);                                               \
                zmq::zmq_abort (errstr);                                       \
            }                                                                  \
        }                                                                      \
    } while (false)

//  Provides convenient way to assert on WSA-style errors on Windows.
#define wsa_assert_no(no)                                                      \
    do {                                                                       \
        const char *errstr = zmq::wsa_error_no (no);                           \
        if (errstr != NULL) {                                                  \
            fprintf (stderr, "Assertion failed: %s (%s:%d)\n", errstr,         \
                     __FILE__, __LINE__);                                      \
            fflush (stderr);                                                   \
            zmq::zmq_abort (errstr);                                           \
        }                                                                      \
    } while (false)

// Provides convenient way to check GetLastError-style errors on Windows.
#define win_assert(x)                                                          \
    do {                                                                       \
        if (unlikely (!(x))) {                                                 \
            char errstr[256];                                                  \
            zmq::win_error (errstr, 256);                                      \
            fprintf (stderr, "Assertion failed: %s (%s:%d)\n", errstr,         \
                     __FILE__, __LINE__);                                      \
            fflush (stderr);                                                   \
            zmq::zmq_abort (errstr);                                           \
        }                                                                      \
    } while (false)

#endif

//  This macro works in exactly the same way as the normal assert. It is used
//  in its stead because standard assert on Win32 in broken - it prints nothing
//  when used within the scope of JNI library.
#define zmq_assert(x)                                                          \
    do {                                                                       \
        if (unlikely (!(x))) {                                                 \
            fprintf (stderr, "Assertion failed: %s (%s:%d)\n", #x, __FILE__,   \
                     __LINE__);                                                \
            fflush (stderr);                                                   \
            zmq::zmq_abort (#x);                                               \
        }                                                                      \
    } while (false)

//  Provides convenient way to check for errno-style errors.
#define errno_assert(x)                                                        \
    do {                                                                       \
        if (unlikely (!(x))) {                                                 \
            const char *errstr = strerror (errno);                             \
            fprintf (stderr, "%s (%s:%d)\n", errstr, __FILE__, __LINE__);      \
            fflush (stderr);                                                   \
            zmq::zmq_abort (errstr);                                           \
        }                                                                      \
    } while (false)

//  Provides convenient way to check for POSIX errors.
#define posix_assert(x)                                                        \
    do {                                                                       \
        if (unlikely (x)) {                                                    \
            const char *errstr = strerror (x);                                 \
            fprintf (stderr, "%s (%s:%d)\n", errstr, __FILE__, __LINE__);      \
            fflush (stderr);                                                   \
            zmq::zmq_abort (errstr);                                           \
        }                                                                      \
    } while (false)

//  Provides convenient way to check for errors from getaddrinfo.
#define gai_assert(x)                                                          \
    do {                                                                       \
        if (unlikely (x)) {                                                    \
            const char *errstr = gai_strerror (x);                             \
            fprintf (stderr, "%s (%s:%d)\n", errstr, __FILE__, __LINE__);      \
            fflush (stderr);                                                   \
            zmq::zmq_abort (errstr);                                           \
        }                                                                      \
    } while (false)

//  Provides convenient way to check whether memory allocation have succeeded.
#define alloc_assert(x)                                                        \
    do {                                                                       \
        if (unlikely (!x)) {                                                   \
            fprintf (stderr, "FATAL ERROR: OUT OF MEMORY (%s:%d)\n", __FILE__, \
                     __LINE__);                                                \
            fflush (stderr);                                                   \
            zmq::zmq_abort ("FATAL ERROR: OUT OF MEMORY");                     \
        }                                                                      \
    } while (false)

#endif


//========= end of #include "err.hpp" ============


//========= begin of #include "config.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CONFIG_HPP_INCLUDED__
#define __ZMQ_CONFIG_HPP_INCLUDED__

namespace zmq
{
//  Compile-time settings.

enum
{
    //  Number of new messages in message pipe needed to trigger new memory
    //  allocation. Setting this parameter to 256 decreases the impact of
    //  memory allocation by approximately 99.6%
    message_pipe_granularity = 256,

    //  Commands in pipe per allocation event.
    command_pipe_granularity = 16,

    //  Determines how often does socket poll for new commands when it
    //  still has unprocessed messages to handle. Thus, if it is set to 100,
    //  socket will process 100 inbound messages before doing the poll.
    //  If there are no unprocessed messages available, poll is done
    //  immediately. Decreasing the value trades overall latency for more
    //  real-time behaviour (less latency peaks).
    inbound_poll_rate = 100,

    //  Maximal delta between high and low watermark.
    max_wm_delta = 1024,

    //  Maximum number of events the I/O thread can process in one go.
    max_io_events = 256,

    //  Maximal batch size of packets forwarded by a ZMQ proxy.
    //  Increasing this value improves throughput at the expense of
    //  latency and fairness.
    proxy_burst_size = 1000,

    //  Maximal delay to process command in API thread (in CPU ticks).
    //  3,000,000 ticks equals to 1 - 2 milliseconds on current CPUs.
    //  Note that delay is only applied when there is continuous stream of
    //  messages to process. If not so, commands are processed immediately.
    max_command_delay = 3000000,

    //  Low-precision clock precision in CPU ticks. 1ms. Value of 1000000
    //  should be OK for CPU frequencies above 1GHz. If should work
    //  reasonably well for CPU frequencies above 500MHz. For lower CPU
    //  frequencies you may consider lowering this value to get best
    //  possible latencies.
    clock_precision = 1000000,

    //  On some OSes the signaler has to be emulated using a TCP
    //  connection. In such cases following port is used.
    //  If 0, it lets the OS choose a free port without requiring use of a
    //  global mutex. The original implementation of a Windows signaler
    //  socket used port 5905 instead of letting the OS choose a free port.
    //  https://github.com/zeromq/libzmq/issues/1542
    signaler_port = 0
};
}

#endif


//========= end of #include "config.hpp" ============


//========= begin of #include "fd.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_FD_HPP_INCLUDED__
#define __ZMQ_FD_HPP_INCLUDED__

#if defined _WIN32
// ans ignore: #include "windows.hpp"
#endif

namespace zmq
{
#ifdef ZMQ_HAVE_WINDOWS
#if defined _MSC_VER && _MSC_VER <= 1400
///< \todo zmq.h uses SOCKET unconditionally, so probably VS versions before
/// VS2008 are unsupported anyway. Apart from that, this seems to depend on
/// the Windows SDK version rather than the VS version.
typedef UINT_PTR fd_t;
enum
{
    retired_fd = (fd_t) (~0)
};
#else
typedef SOCKET fd_t;
enum
#if _MSC_VER >= 1800
  : fd_t
#endif
{
    retired_fd = INVALID_SOCKET
};
#endif
#else
typedef int fd_t;
enum
{
    retired_fd = -1
};
#endif
}
#endif


//========= end of #include "fd.hpp" ============


//========= begin of #include "stdint.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_STDINT_HPP_INCLUDED__
#define __ZMQ_STDINT_HPP_INCLUDED__

#if defined ZMQ_HAVE_SOLARIS || defined ZMQ_HAVE_OPENVMS

#include <inttypes.h>

#elif defined _MSC_VER && _MSC_VER < 1600

#ifndef int8_t
typedef __int8 int8_t;
#endif
#ifndef int16_t
typedef __int16 int16_t;
#endif
#ifndef int32_t
typedef __int32 int32_t;
#endif
#ifndef int64_t
typedef __int64 int64_t;
#endif
#ifndef uint8_t
typedef unsigned __int8 uint8_t;
#endif
#ifndef uint16_t
typedef unsigned __int16 uint16_t;
#endif
#ifndef uint32_t
typedef unsigned __int32 uint32_t;
#endif
#ifndef uint64_t
typedef unsigned __int64 uint64_t;
#endif
#ifndef UINT16_MAX
#define UINT16_MAX _UI16_MAX
#endif
#ifndef UINT32_MAX
#define UINT32_MAX _UI32_MAX
#endif

#else

#include <stdint.h>

#endif

#ifndef UINT8_MAX
#define UINT8_MAX 0xFF
#endif

#endif


//========= end of #include "stdint.hpp" ============


//========= begin of #include "macros.hpp" ============


/******************************************************************************/
/*  0MQ Internal Use                                                          */
/******************************************************************************/

#define LIBZMQ_UNUSED(object) (void) object
#define LIBZMQ_DELETE(p_object)                                                \
    {                                                                          \
        delete p_object;                                                       \
        p_object = 0;                                                          \
    }

/******************************************************************************/

#if !defined ZMQ_NOEXCEPT
#if defined ZMQ_HAVE_NOEXCEPT
#define ZMQ_NOEXCEPT noexcept
#else
#define ZMQ_NOEXCEPT
#endif
#endif


//========= end of #include "macros.hpp" ============


//========= begin of #include "mutex.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MUTEX_HPP_INCLUDED__
#define __ZMQ_MUTEX_HPP_INCLUDED__

// ans ignore: #include "err.hpp"

//  Mutex class encapsulates OS mutex in a platform-independent way.

#ifdef ZMQ_HAVE_WINDOWS

// ans ignore: #include "windows.hpp"

namespace zmq
{
class mutex_t
{
  public:
    inline mutex_t () { InitializeCriticalSection (&_cs); }

    inline ~mutex_t () { DeleteCriticalSection (&_cs); }

    inline void lock () { EnterCriticalSection (&_cs); }

    inline bool try_lock ()
    {
        return (TryEnterCriticalSection (&_cs)) ? true : false;
    }

    inline void unlock () { LeaveCriticalSection (&_cs); }

    inline CRITICAL_SECTION *get_cs () { return &_cs; }

  private:
    CRITICAL_SECTION _cs;

    //  Disable copy construction and assignment.
    mutex_t (const mutex_t &);
    void operator= (const mutex_t &);
};
}

#elif defined ZMQ_HAVE_VXWORKS

#include <vxWorks.h>
#include <semLib.h>

namespace zmq
{
class mutex_t
{
  public:
    inline mutex_t ()
    {
        _semId =
          semMCreate (SEM_Q_PRIORITY | SEM_INVERSION_SAFE | SEM_DELETE_SAFE);
    }

    inline ~mutex_t () { semDelete (_semId); }

    inline void lock () { semTake (_semId, WAIT_FOREVER); }

    inline bool try_lock ()
    {
        if (semTake (_semId, NO_WAIT) == OK) {
            return true;
        }
        return false;
    }

    inline void unlock () { semGive (_semId); }

  private:
    SEM_ID _semId;

    // Disable copy construction and assignment.
    mutex_t (const mutex_t &);
    const mutex_t &operator= (const mutex_t &);
};
}

#else

#include <pthread.h>

namespace zmq
{
class mutex_t
{
  public:
    inline mutex_t ()
    {
        int rc = pthread_mutexattr_init (&_attr);
        posix_assert (rc);

        rc = pthread_mutexattr_settype (&_attr, PTHREAD_MUTEX_RECURSIVE);
        posix_assert (rc);

        rc = pthread_mutex_init (&_mutex, &_attr);
        posix_assert (rc);
    }

    inline ~mutex_t ()
    {
        int rc = pthread_mutex_destroy (&_mutex);
        posix_assert (rc);

        rc = pthread_mutexattr_destroy (&_attr);
        posix_assert (rc);
    }

    inline void lock ()
    {
        int rc = pthread_mutex_lock (&_mutex);
        posix_assert (rc);
    }

    inline bool try_lock ()
    {
        int rc = pthread_mutex_trylock (&_mutex);
        if (rc == EBUSY)
            return false;

        posix_assert (rc);
        return true;
    }

    inline void unlock ()
    {
        int rc = pthread_mutex_unlock (&_mutex);
        posix_assert (rc);
    }

    inline pthread_mutex_t *get_mutex () { return &_mutex; }

  private:
    pthread_mutex_t _mutex;
    pthread_mutexattr_t _attr;

    // Disable copy construction and assignment.
    mutex_t (const mutex_t &);
    const mutex_t &operator= (const mutex_t &);
};
}

#endif


namespace zmq
{
struct scoped_lock_t
{
    scoped_lock_t (mutex_t &mutex_) : _mutex (mutex_) { _mutex.lock (); }

    ~scoped_lock_t () { _mutex.unlock (); }

  private:
    mutex_t &_mutex;

    // Disable copy construction and assignment.
    scoped_lock_t (const scoped_lock_t &);
    const scoped_lock_t &operator= (const scoped_lock_t &);
};


struct scoped_optional_lock_t
{
    scoped_optional_lock_t (mutex_t *mutex_) : _mutex (mutex_)
    {
        if (_mutex != NULL)
            _mutex->lock ();
    }

    ~scoped_optional_lock_t ()
    {
        if (_mutex != NULL)
            _mutex->unlock ();
    }

  private:
    mutex_t *_mutex;

    // Disable copy construction and assignment.
    scoped_optional_lock_t (const scoped_lock_t &);
    const scoped_optional_lock_t &operator= (const scoped_lock_t &);
};
}

#endif


//========= end of #include "mutex.hpp" ============


//========= begin of #include "atomic_counter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ATOMIC_COUNTER_HPP_INCLUDED__
#define __ZMQ_ATOMIC_COUNTER_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"
// ans ignore: #include "macros.hpp"

#if defined ZMQ_FORCE_MUTEXES
#define ZMQ_ATOMIC_COUNTER_MUTEX
#elif (defined __cplusplus && __cplusplus >= 201103L)                          \
  || (defined _MSC_VER && _MSC_VER >= 1900)
#define ZMQ_ATOMIC_COUNTER_CXX11
#elif defined ZMQ_HAVE_ATOMIC_INTRINSICS
#define ZMQ_ATOMIC_COUNTER_INTRINSIC
#elif (defined __i386__ || defined __x86_64__) && defined __GNUC__
#define ZMQ_ATOMIC_COUNTER_X86
#elif defined __ARM_ARCH_7A__ && defined __GNUC__
#define ZMQ_ATOMIC_COUNTER_ARM
#elif defined ZMQ_HAVE_WINDOWS
#define ZMQ_ATOMIC_COUNTER_WINDOWS
#elif (defined ZMQ_HAVE_SOLARIS || defined ZMQ_HAVE_NETBSD                     \
       || defined ZMQ_HAVE_GNU)
#define ZMQ_ATOMIC_COUNTER_ATOMIC_H
#elif defined __tile__
#define ZMQ_ATOMIC_COUNTER_TILE
#else
#define ZMQ_ATOMIC_COUNTER_MUTEX
#endif

#if defined ZMQ_ATOMIC_COUNTER_MUTEX
// ans ignore: #include "mutex.hpp"
#elif defined ZMQ_ATOMIC_COUNTER_CXX11
#include <atomic>
#elif defined ZMQ_ATOMIC_COUNTER_WINDOWS
// ans ignore: #include "windows.hpp"
#elif defined ZMQ_ATOMIC_COUNTER_ATOMIC_H
#include <atomic.h>
#elif defined ZMQ_ATOMIC_COUNTER_TILE
#include <arch/atomic.h>
#endif

namespace zmq
{
//  This class represents an integer that can be incremented/decremented
//  in atomic fashion.
//
//  In zmq::shared_message_memory_allocator a buffer with an atomic_counter_t
//  at the start is allocated. If the class does not align to pointer size,
//  access to pointers in structures in the buffer will cause SIGBUS on
//  architectures that do not allow mis-aligned pointers (eg: SPARC).
//  Force the compiler to align to pointer size, which will cause the object
//  to grow from 4 bytes to 8 bytes on 64 bit architectures (when not using
//  mutexes).

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
class __declspec(align (8)) atomic_counter_t
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_ARM_ARMV7VE))
class __declspec(align (4)) atomic_counter_t
#else
class atomic_counter_t
#endif
{
  public:
    typedef uint32_t integer_t;

    inline atomic_counter_t (integer_t value_ = 0) ZMQ_NOEXCEPT
        : _value (value_)
    {
    }

    //  Set counter _value (not thread-safe).
    inline void set (integer_t value_) ZMQ_NOEXCEPT { _value = value_; }

    //  Atomic addition. Returns the old _value.
    inline integer_t add (integer_t increment_) ZMQ_NOEXCEPT
    {
        integer_t old_value;

#if defined ZMQ_ATOMIC_COUNTER_WINDOWS
        old_value = InterlockedExchangeAdd ((LONG *) &_value, increment_);
#elif defined ZMQ_ATOMIC_COUNTER_INTRINSIC
        old_value = __atomic_fetch_add (&_value, increment_, __ATOMIC_ACQ_REL);
#elif defined ZMQ_ATOMIC_COUNTER_CXX11
        old_value = _value.fetch_add (increment_, std::memory_order_acq_rel);
#elif defined ZMQ_ATOMIC_COUNTER_ATOMIC_H
        integer_t new_value = atomic_add_32_nv (&_value, increment_);
        old_value = new_value - increment_;
#elif defined ZMQ_ATOMIC_COUNTER_TILE
        old_value = arch_atomic_add (&_value, increment_);
#elif defined ZMQ_ATOMIC_COUNTER_X86
        __asm__ volatile("lock; xadd %0, %1 \n\t"
                         : "=r"(old_value), "=m"(_value)
                         : "0"(increment_), "m"(_value)
                         : "cc", "memory");
#elif defined ZMQ_ATOMIC_COUNTER_ARM
        integer_t flag, tmp;
        __asm__ volatile("       dmb     sy\n\t"
                         "1:     ldrex   %0, [%5]\n\t"
                         "       add     %2, %0, %4\n\t"
                         "       strex   %1, %2, [%5]\n\t"
                         "       teq     %1, #0\n\t"
                         "       bne     1b\n\t"
                         "       dmb     sy\n\t"
                         : "=&r"(old_value), "=&r"(flag), "=&r"(tmp),
                           "+Qo"(_value)
                         : "Ir"(increment_), "r"(&_value)
                         : "cc");
#elif defined ZMQ_ATOMIC_COUNTER_MUTEX
        sync.lock ();
        old_value = _value;
        _value += increment_;
        sync.unlock ();
#else
#error atomic_counter is not implemented for this platform
#endif
        return old_value;
    }

    //  Atomic subtraction. Returns false if the counter drops to zero.
    inline bool sub (integer_t decrement_) ZMQ_NOEXCEPT
    {
#if defined ZMQ_ATOMIC_COUNTER_WINDOWS
        LONG delta = -((LONG) decrement_);
        integer_t old = InterlockedExchangeAdd ((LONG *) &_value, delta);
        return old - decrement_ != 0;
#elif defined ZMQ_ATOMIC_COUNTER_INTRINSIC
        integer_t nv =
          __atomic_sub_fetch (&_value, decrement_, __ATOMIC_ACQ_REL);
        return nv != 0;
#elif defined ZMQ_ATOMIC_COUNTER_CXX11
        integer_t old =
          _value.fetch_sub (decrement_, std::memory_order_acq_rel);
        return old - decrement_ != 0;
#elif defined ZMQ_ATOMIC_COUNTER_ATOMIC_H
        int32_t delta = -((int32_t) decrement_);
        integer_t nv = atomic_add_32_nv (&_value, delta);
        return nv != 0;
#elif defined ZMQ_ATOMIC_COUNTER_TILE
        int32_t delta = -((int32_t) decrement_);
        integer_t nv = arch_atomic_add (&_value, delta);
        return nv != 0;
#elif defined ZMQ_ATOMIC_COUNTER_X86
        integer_t oldval = -decrement_;
        volatile integer_t *val = &_value;
        __asm__ volatile("lock; xaddl %0,%1"
                         : "=r"(oldval), "=m"(*val)
                         : "0"(oldval), "m"(*val)
                         : "cc", "memory");
        return oldval != decrement_;
#elif defined ZMQ_ATOMIC_COUNTER_ARM
        integer_t old_value, flag, tmp;
        __asm__ volatile("       dmb     sy\n\t"
                         "1:     ldrex   %0, [%5]\n\t"
                         "       sub     %2, %0, %4\n\t"
                         "       strex   %1, %2, [%5]\n\t"
                         "       teq     %1, #0\n\t"
                         "       bne     1b\n\t"
                         "       dmb     sy\n\t"
                         : "=&r"(old_value), "=&r"(flag), "=&r"(tmp),
                           "+Qo"(_value)
                         : "Ir"(decrement_), "r"(&_value)
                         : "cc");
        return old_value - decrement_ != 0;
#elif defined ZMQ_ATOMIC_COUNTER_MUTEX
        sync.lock ();
        _value -= decrement_;
        bool result = _value ? true : false;
        sync.unlock ();
        return result;
#else
#error atomic_counter is not implemented for this platform
#endif
    }

    inline integer_t get () const ZMQ_NOEXCEPT { return _value; }

  private:
#if defined ZMQ_ATOMIC_COUNTER_CXX11
    std::atomic<integer_t> _value;
#else
    volatile integer_t _value;
#endif

#if defined ZMQ_ATOMIC_COUNTER_MUTEX
    mutex_t sync;
#endif

#if !defined ZMQ_ATOMIC_COUNTER_CXX11
    atomic_counter_t (const atomic_counter_t &);
    const atomic_counter_t &operator= (const atomic_counter_t &);
#endif
#if defined(__GNUC__) || defined(__INTEL_COMPILER)                             \
  || (defined(__SUNPRO_C) && __SUNPRO_C >= 0x590)                              \
  || (defined(__SUNPRO_CC) && __SUNPRO_CC >= 0x590)
} __attribute__ ((aligned (sizeof (void *))));
#else
};
#endif
}

//  Remove macros local to this file.
#undef ZMQ_ATOMIC_COUNTER_MUTEX
#undef ZMQ_ATOMIC_COUNTER_INTRINSIC
#undef ZMQ_ATOMIC_COUNTER_CXX11
#undef ZMQ_ATOMIC_COUNTER_X86
#undef ZMQ_ATOMIC_COUNTER_ARM
#undef ZMQ_ATOMIC_COUNTER_WINDOWS
#undef ZMQ_ATOMIC_COUNTER_ATOMIC_H
#undef ZMQ_ATOMIC_COUNTER_TILE

#endif


//========= end of #include "atomic_counter.hpp" ============


//========= begin of #include "metadata.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_METADATA_HPP_INCLUDED__
#define __ZMQ_METADATA_HPP_INCLUDED__

#include <map>
#include <string>

// ans ignore: #include "atomic_counter.hpp"

namespace zmq
{
class metadata_t
{
  public:
    typedef std::map<std::string, std::string> dict_t;

    metadata_t (const dict_t &dict_);

    //  Returns pointer to property value or NULL if
    //  property is not found.
    const char *get (const std::string &property_) const;

    void add_ref ();

    //  Drop reference. Returns true iff the reference
    //  counter drops to zero.
    bool drop_ref ();

  private:
    metadata_t (const metadata_t &);
    metadata_t &operator= (const metadata_t &);

    //  Reference counter.
    atomic_counter_t _ref_cnt;

    //  Dictionary holding metadata.
    const dict_t _dict;
};
}

#endif


//========= end of #include "metadata.hpp" ============


//========= begin of #include "msg.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MSG_HPP_INCLUDE__
#define __ZMQ_MSG_HPP_INCLUDE__

#include <stddef.h>
#include <stdio.h>

// ans ignore: #include "config.hpp"
// ans ignore: #include "err.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "atomic_counter.hpp"
// ans ignore: #include "metadata.hpp"

//  bits 2-5
#define CMD_TYPE_MASK 0x1c

//  Signature for free function to deallocate the message content.
//  Note that it has to be declared as "C" so that it is the same as
//  zmq_free_fn defined in zmq.h.
extern "C" {
typedef void(msg_free_fn) (void *data_, void *hint_);
}

namespace zmq
{
//  Note that this structure needs to be explicitly constructed
//  (init functions) and destructed (close function).

class msg_t
{
  public:
    //  Shared message buffer. Message data are either allocated in one
    //  continuous block along with this structure - thus avoiding one
    //  malloc/free pair or they are stored in user-supplied memory.
    //  In the latter case, ffn member stores pointer to the function to be
    //  used to deallocate the data. If the buffer is actually shared (there
    //  are at least 2 references to it) refcount member contains number of
    //  references.
    struct content_t
    {
        void *data;
        size_t size;
        msg_free_fn *ffn;
        void *hint;
        zmq::atomic_counter_t refcnt;
    };

    //  Message flags.
    enum
    {
        more = 1,    //  Followed by more parts
        command = 2, //  Command frame (see ZMTP spec)
        //  Command types, use only bits 2-5 and compare with ==, not bitwise,
        //  a command can never be of more that one type at the same time
        ping = 4,
        pong = 8,
        subscribe = 12,
        cancel = 16,
        credential = 32,
        routing_id = 64,
        shared = 128
    };

    bool check () const;
    int init ();

    int init (void *data_,
              size_t size_,
              msg_free_fn *ffn_,
              void *hint_,
              content_t *content_ = NULL);

    int init_size (size_t size_);
    int init_data (void *data_, size_t size_, msg_free_fn *ffn_, void *hint_);
    int init_external_storage (content_t *content_,
                               void *data_,
                               size_t size_,
                               msg_free_fn *ffn_,
                               void *hint_);
    int init_delimiter ();
    int init_join ();
    int init_leave ();
    int close ();
    int move (msg_t &src_);
    int copy (msg_t &src_);
    void *data ();
    size_t size () const;
    unsigned char flags () const;
    void set_flags (unsigned char flags_);
    void reset_flags (unsigned char flags_);
    metadata_t *metadata () const;
    void set_metadata (metadata_t *metadata_);
    void reset_metadata ();
    bool is_routing_id () const;
    bool is_credential () const;
    bool is_delimiter () const;
    bool is_join () const;
    bool is_leave () const;
    bool is_ping () const;
    bool is_pong () const;

    //  These are called on each message received by the session_base class,
    //  so get them inlined to avoid the overhead of 2 function calls per msg
    inline bool is_subscribe () const
    {
        return (_u.base.flags & CMD_TYPE_MASK) == subscribe;
    }
    inline bool is_cancel () const
    {
        return (_u.base.flags & CMD_TYPE_MASK) == cancel;
    }

    size_t command_body_size () const;
    void *command_body ();
    bool is_vsm () const;
    bool is_cmsg () const;
    bool is_lmsg () const;
    bool is_zcmsg () const;
    uint32_t get_routing_id ();
    int set_routing_id (uint32_t routing_id_);
    int reset_routing_id ();
    const char *group ();
    int set_group (const char *group_);
    int set_group (const char *, size_t length_);

    //  After calling this function you can copy the message in POD-style
    //  refs_ times. No need to call copy.
    void add_refs (int refs_);

    //  Removes references previously added by add_refs. If the number of
    //  references drops to 0, the message is closed and false is returned.
    bool rm_refs (int refs_);

    //  Size in bytes of the largest message that is still copied around
    //  rather than being reference-counted.
    enum
    {
        msg_t_size = 64
    };
    enum
    {
        max_vsm_size =
          msg_t_size - (sizeof (metadata_t *) + 3 + 16 + sizeof (uint32_t))
    };
    enum
    {
        ping_cmd_name_size = 5,   // 4PING
        cancel_cmd_name_size = 7, // 6CANCEL
        sub_cmd_name_size = 10    // 9SUBSCRIBE
    };

  private:
    zmq::atomic_counter_t *refcnt ();

    //  Different message types.
    enum type_t
    {
        type_min = 101,
        //  VSM messages store the content in the message itself
        type_vsm = 101,
        //  LMSG messages store the content in malloc-ed memory
        type_lmsg = 102,
        //  Delimiter messages are used in envelopes
        type_delimiter = 103,
        //  CMSG messages point to constant data
        type_cmsg = 104,

        // zero-copy LMSG message for v2_decoder
        type_zclmsg = 105,

        //  Join message for radio_dish
        type_join = 106,

        //  Leave message for radio_dish
        type_leave = 107,

        type_max = 107
    };

    //  Note that fields shared between different message types are not
    //  moved to the parent class (msg_t). This way we get tighter packing
    //  of the data. Shared fields can be accessed via 'base' member of
    //  the union.
    union
    {
        struct
        {
            metadata_t *metadata;
            unsigned char
              unused[msg_t_size
                     - (sizeof (metadata_t *) + 2 + 16 + sizeof (uint32_t))];
            unsigned char type;
            unsigned char flags;
            char group[16];
            uint32_t routing_id;
        } base;
        struct
        {
            metadata_t *metadata;
            unsigned char data[max_vsm_size];
            unsigned char size;
            unsigned char type;
            unsigned char flags;
            char group[16];
            uint32_t routing_id;
        } vsm;
        struct
        {
            metadata_t *metadata;
            content_t *content;
            unsigned char unused[msg_t_size
                                 - (sizeof (metadata_t *) + sizeof (content_t *)
                                    + 2 + 16 + sizeof (uint32_t))];
            unsigned char type;
            unsigned char flags;
            char group[16];
            uint32_t routing_id;
        } lmsg;
        struct
        {
            metadata_t *metadata;
            content_t *content;
            unsigned char unused[msg_t_size
                                 - (sizeof (metadata_t *) + sizeof (content_t *)
                                    + 2 + 16 + sizeof (uint32_t))];
            unsigned char type;
            unsigned char flags;
            char group[16];
            uint32_t routing_id;
        } zclmsg;
        struct
        {
            metadata_t *metadata;
            void *data;
            size_t size;
            unsigned char
              unused[msg_t_size
                     - (sizeof (metadata_t *) + sizeof (void *)
                        + sizeof (size_t) + 2 + 16 + sizeof (uint32_t))];
            unsigned char type;
            unsigned char flags;
            char group[16];
            uint32_t routing_id;
        } cmsg;
        struct
        {
            metadata_t *metadata;
            unsigned char
              unused[msg_t_size
                     - (sizeof (metadata_t *) + 2 + 16 + sizeof (uint32_t))];
            unsigned char type;
            unsigned char flags;
            char group[16];
            uint32_t routing_id;
        } delimiter;
    } _u;
};

inline int close_and_return (zmq::msg_t *msg_, int echo_)
{
    // Since we abort on close failure we preserve errno for success case.
    int err = errno;
    const int rc = msg_->close ();
    errno_assert (rc == 0);
    errno = err;
    return echo_;
}

inline int close_and_return (zmq::msg_t msg_[], int count_, int echo_)
{
    for (int i = 0; i < count_; i++)
        close_and_return (&msg_[i], 0);
    return echo_;
}
}

#endif


//========= end of #include "msg.hpp" ============


//========= begin of #include "atomic_ptr.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ATOMIC_PTR_HPP_INCLUDED__
#define __ZMQ_ATOMIC_PTR_HPP_INCLUDED__

// ans ignore: #include "macros.hpp"

#if defined ZMQ_FORCE_MUTEXES
#define ZMQ_ATOMIC_PTR_MUTEX
#elif (defined __cplusplus && __cplusplus >= 201103L)                          \
  || (defined _MSC_VER && _MSC_VER >= 1900)
#define ZMQ_ATOMIC_PTR_CXX11
#elif defined ZMQ_HAVE_ATOMIC_INTRINSICS
#define ZMQ_ATOMIC_PTR_INTRINSIC
#elif (defined __i386__ || defined __x86_64__) && defined __GNUC__
#define ZMQ_ATOMIC_PTR_X86
#elif defined __ARM_ARCH_7A__ && defined __GNUC__
#define ZMQ_ATOMIC_PTR_ARM
#elif defined __tile__
#define ZMQ_ATOMIC_PTR_TILE
#elif defined ZMQ_HAVE_WINDOWS
#define ZMQ_ATOMIC_PTR_WINDOWS
#elif (defined ZMQ_HAVE_SOLARIS || defined ZMQ_HAVE_NETBSD                     \
       || defined ZMQ_HAVE_GNU)
#define ZMQ_ATOMIC_PTR_ATOMIC_H
#else
#define ZMQ_ATOMIC_PTR_MUTEX
#endif

#if defined ZMQ_ATOMIC_PTR_MUTEX
// ans ignore: #include "mutex.hpp"
#elif defined ZMQ_ATOMIC_PTR_CXX11
#include <atomic>
#elif defined ZMQ_ATOMIC_PTR_WINDOWS
// ans ignore: #include "windows.hpp"
#elif defined ZMQ_ATOMIC_PTR_ATOMIC_H
#include <atomic.h>
#elif defined ZMQ_ATOMIC_PTR_TILE
#include <arch/atomic.h>
#endif

namespace zmq
{
#if !defined ZMQ_ATOMIC_PTR_CXX11
inline void *atomic_xchg_ptr (void **ptr_,
                              void *const val_
#if defined ZMQ_ATOMIC_PTR_MUTEX
                              ,
                              mutex_t &_sync
#endif
                              ) ZMQ_NOEXCEPT
{
#if defined ZMQ_ATOMIC_PTR_WINDOWS
    return InterlockedExchangePointer ((PVOID *) ptr_, val_);
#elif defined ZMQ_ATOMIC_PTR_INTRINSIC
    return __atomic_exchange_n (ptr_, val_, __ATOMIC_ACQ_REL);
#elif defined ZMQ_ATOMIC_PTR_ATOMIC_H
    return atomic_swap_ptr (ptr_, val_);
#elif defined ZMQ_ATOMIC_PTR_TILE
    return arch_atomic_exchange (ptr_, val_);
#elif defined ZMQ_ATOMIC_PTR_X86
    void *old;
    __asm__ volatile("lock; xchg %0, %2"
                     : "=r"(old), "=m"(*ptr_)
                     : "m"(*ptr_), "0"(val_));
    return old;
#elif defined ZMQ_ATOMIC_PTR_ARM
    void *old;
    unsigned int flag;
    __asm__ volatile("       dmb     sy\n\t"
                     "1:     ldrex   %1, [%3]\n\t"
                     "       strex   %0, %4, [%3]\n\t"
                     "       teq     %0, #0\n\t"
                     "       bne     1b\n\t"
                     "       dmb     sy\n\t"
                     : "=&r"(flag), "=&r"(old), "+Qo"(*ptr_)
                     : "r"(ptr_), "r"(val_)
                     : "cc");
    return old;
#elif defined ZMQ_ATOMIC_PTR_MUTEX
    _sync.lock ();
    void *old = *ptr_;
    *ptr_ = val_;
    _sync.unlock ();
    return old;
#else
#error atomic_ptr is not implemented for this platform
#endif
}

inline void *atomic_cas (void *volatile *ptr_,
                         void *cmp_,
                         void *val_
#if defined ZMQ_ATOMIC_PTR_MUTEX
                         ,
                         mutex_t &_sync
#endif
                         ) ZMQ_NOEXCEPT
{
#if defined ZMQ_ATOMIC_PTR_WINDOWS
    return InterlockedCompareExchangePointer ((volatile PVOID *) ptr_, val_,
                                              cmp_);
#elif defined ZMQ_ATOMIC_PTR_INTRINSIC
    void *old = cmp_;
    __atomic_compare_exchange_n (ptr_, &old, val_, false, __ATOMIC_RELEASE,
                                 __ATOMIC_ACQUIRE);
    return old;
#elif defined ZMQ_ATOMIC_PTR_ATOMIC_H
    return atomic_cas_ptr (ptr_, cmp_, val_);
#elif defined ZMQ_ATOMIC_PTR_TILE
    return arch_atomic_val_compare_and_exchange (ptr_, cmp_, val_);
#elif defined ZMQ_ATOMIC_PTR_X86
    void *old;
    __asm__ volatile("lock; cmpxchg %2, %3"
                     : "=a"(old), "=m"(*ptr_)
                     : "r"(val_), "m"(*ptr_), "0"(cmp_)
                     : "cc");
    return old;
#elif defined ZMQ_ATOMIC_PTR_ARM
    void *old;
    unsigned int flag;
    __asm__ volatile("       dmb     sy\n\t"
                     "1:     ldrex   %1, [%3]\n\t"
                     "       mov     %0, #0\n\t"
                     "       teq     %1, %4\n\t"
                     "       it      eq\n\t"
                     "       strexeq %0, %5, [%3]\n\t"
                     "       teq     %0, #0\n\t"
                     "       bne     1b\n\t"
                     "       dmb     sy\n\t"
                     : "=&r"(flag), "=&r"(old), "+Qo"(*ptr_)
                     : "r"(ptr_), "r"(cmp_), "r"(val_)
                     : "cc");
    return old;
#elif defined ZMQ_ATOMIC_PTR_MUTEX
    _sync.lock ();
    void *old = *ptr_;
    if (*ptr_ == cmp_)
        *ptr_ = val_;
    _sync.unlock ();
    return old;
#else
#error atomic_ptr is not implemented for this platform
#endif
}
#endif

//  This class encapsulates several atomic operations on pointers.

template <typename T> class atomic_ptr_t
{
  public:
    //  Initialise atomic pointer
    inline atomic_ptr_t () ZMQ_NOEXCEPT { _ptr = NULL; }

    //  Set value of atomic pointer in a non-threadsafe way
    //  Use this function only when you are sure that at most one
    //  thread is accessing the pointer at the moment.
    inline void set (T *ptr_) ZMQ_NOEXCEPT { _ptr = ptr_; }

    //  Perform atomic 'exchange pointers' operation. Pointer is set
    //  to the 'val_' value. Old value is returned.
    inline T *xchg (T *val_) ZMQ_NOEXCEPT
    {
#if defined ZMQ_ATOMIC_PTR_CXX11
        return _ptr.exchange (val_, std::memory_order_acq_rel);
#else
        return (T *) atomic_xchg_ptr ((void **) &_ptr, val_
#if defined ZMQ_ATOMIC_PTR_MUTEX
                                      ,
                                      _sync
#endif
        );
#endif
    }

    //  Perform atomic 'compare and swap' operation on the pointer.
    //  The pointer is compared to 'cmp' argument and if they are
    //  equal, its value is set to 'val_'. Old value of the pointer
    //  is returned.
    inline T *cas (T *cmp_, T *val_) ZMQ_NOEXCEPT
    {
#if defined ZMQ_ATOMIC_PTR_CXX11
        _ptr.compare_exchange_strong (cmp_, val_, std::memory_order_acq_rel);
        return cmp_;
#else
        return (T *) atomic_cas ((void **) &_ptr, cmp_, val_
#if defined ZMQ_ATOMIC_PTR_MUTEX
                                 ,
                                 _sync
#endif
        );
#endif
    }

  private:
#if defined ZMQ_ATOMIC_PTR_CXX11
    std::atomic<T *> _ptr;
#else
    volatile T *_ptr;
#endif

#if defined ZMQ_ATOMIC_PTR_MUTEX
    mutex_t _sync;
#endif

#if !defined ZMQ_ATOMIC_PTR_CXX11
    atomic_ptr_t (const atomic_ptr_t &);
    const atomic_ptr_t &operator= (const atomic_ptr_t &);
#endif
};

struct atomic_value_t
{
    atomic_value_t (const int value_) ZMQ_NOEXCEPT : _value (value_) {}

    atomic_value_t (const atomic_value_t &src_) ZMQ_NOEXCEPT
        : _value (src_.load ())
    {
    }

    void store (const int value_) ZMQ_NOEXCEPT
    {
#if defined ZMQ_ATOMIC_PTR_CXX11
        _value.store (value_, std::memory_order_release);
#else
        atomic_xchg_ptr ((void **) &_value, (void *) (ptrdiff_t) value_
#if defined ZMQ_ATOMIC_PTR_MUTEX
                         ,
                         _sync
#endif
        );
#endif
    }

    int load () const ZMQ_NOEXCEPT
    {
#if defined ZMQ_ATOMIC_PTR_CXX11
        return _value.load (std::memory_order_acquire);
#else
        return (int) (ptrdiff_t) atomic_cas ((void **) &_value, 0, 0
#if defined ZMQ_ATOMIC_PTR_MUTEX
                                             ,
#if defined __SUNPRO_CC
                                             const_cast<mutex_t &> (_sync)
#else
                                             _sync
#endif
#endif
        );
#endif
    }

  private:
#if defined ZMQ_ATOMIC_PTR_CXX11
    std::atomic<int> _value;
#else
    volatile ptrdiff_t _value;
#endif

#if defined ZMQ_ATOMIC_PTR_MUTEX
    mutable mutex_t _sync;
#endif

  private:
    atomic_value_t &operator= (const atomic_value_t &src_);
};
}

//  Remove macros local to this file.
#undef ZMQ_ATOMIC_PTR_MUTEX
#undef ZMQ_ATOMIC_PTR_INTRINSIC
#undef ZMQ_ATOMIC_PTR_CXX11
#undef ZMQ_ATOMIC_PTR_X86
#undef ZMQ_ATOMIC_PTR_ARM
#undef ZMQ_ATOMIC_PTR_TILE
#undef ZMQ_ATOMIC_PTR_WINDOWS
#undef ZMQ_ATOMIC_PTR_ATOMIC_H

#endif


//========= end of #include "atomic_ptr.hpp" ============


//========= begin of #include "address.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ADDRESS_HPP_INCLUDED__
#define __ZMQ_ADDRESS_HPP_INCLUDED__

// ans ignore: #include "fd.hpp"

#include <string>

#ifndef ZMQ_HAVE_WINDOWS
#include <sys/socket.h>
#else
#include <ws2tcpip.h>
#endif

namespace zmq
{
class ctx_t;
class tcp_address_t;
class udp_address_t;
#if !defined ZMQ_HAVE_WINDOWS && !defined ZMQ_HAVE_OPENVMS
class ipc_address_t;
#endif
#if defined ZMQ_HAVE_LINUX || defined ZMQ_HAVE_VXWORKS
class tipc_address_t;
#endif
#if defined ZMQ_HAVE_VMCI
class vmci_address_t;
#endif

namespace protocol_name
{
static const char inproc[] = "inproc";
static const char tcp[] = "tcp";
static const char udp[] = "udp";
#if !defined ZMQ_HAVE_WINDOWS && !defined ZMQ_HAVE_OPENVMS                     \
  && !defined ZMQ_HAVE_VXWORKS
static const char ipc[] = "ipc";
#endif
#if defined ZMQ_HAVE_TIPC
static const char tipc[] = "tipc";
#endif
#if defined ZMQ_HAVE_VMCI
static const char vmci[] = "vmci";
#endif
}

struct address_t
{
    address_t (const std::string &protocol_,
               const std::string &address_,
               ctx_t *parent_);

    ~address_t ();

    const std::string protocol;
    const std::string address;
    ctx_t *const parent;

    //  Protocol specific resolved address
    //  All members must be pointers to allow for consistent initialization
    union
    {
        void *dummy;
        tcp_address_t *tcp_addr;
        udp_address_t *udp_addr;
#if !defined ZMQ_HAVE_WINDOWS && !defined ZMQ_HAVE_OPENVMS                     \
  && !defined ZMQ_HAVE_VXWORKS
        ipc_address_t *ipc_addr;
#endif
#if defined ZMQ_HAVE_LINUX || defined ZMQ_HAVE_VXWORKS
        tipc_address_t *tipc_addr;
#endif
#if defined ZMQ_HAVE_VMCI
        vmci_address_t *vmci_addr;
#endif
    } resolved;

    int to_string (std::string &addr_) const;
};

#if defined(ZMQ_HAVE_HPUX) || defined(ZMQ_HAVE_VXWORKS)                        \
  || defined(ZMQ_HAVE_WINDOWS)
typedef int zmq_socklen_t;
#else
typedef socklen_t zmq_socklen_t;
#endif

enum socket_end_t
{
    socket_end_local,
    socket_end_remote
};

zmq_socklen_t
get_socket_address (fd_t fd_, socket_end_t socket_end_, sockaddr_storage *ss_);

template <typename T>
std::string get_socket_name (fd_t fd_, socket_end_t socket_end_)
{
    struct sockaddr_storage ss;
    const zmq_socklen_t sl = get_socket_address (fd_, socket_end_, &ss);
    if (sl == 0) {
        return std::string ();
    }

    const T addr (reinterpret_cast<struct sockaddr *> (&ss), sl);
    std::string address_string;
    addr.to_string (address_string);
    return address_string;
}
}

#endif


//========= end of #include "address.hpp" ============


//========= begin of #include "ip_resolver.hpp" ============

/*
    Copyright (c) 2007-2018 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_IP_RESOLVER_HPP_INCLUDED__
#define __ZMQ_IP_RESOLVER_HPP_INCLUDED__

#if !defined ZMQ_HAVE_WINDOWS
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#endif

// ans ignore: #include "address.hpp"

namespace zmq
{
union ip_addr_t
{
    sockaddr generic;
    sockaddr_in ipv4;
    sockaddr_in6 ipv6;

    int family () const;
    bool is_multicast () const;
    uint16_t port () const;

    const struct sockaddr *as_sockaddr () const;
    zmq_socklen_t sockaddr_len () const;

    void set_port (uint16_t);

    static ip_addr_t any (int family_);
};

class ip_resolver_options_t
{
  public:
    ip_resolver_options_t ();

    ip_resolver_options_t &bindable (bool bindable_);
    ip_resolver_options_t &allow_nic_name (bool allow_);
    ip_resolver_options_t &ipv6 (bool ipv6_);
    ip_resolver_options_t &expect_port (bool expect_);
    ip_resolver_options_t &allow_dns (bool allow_);

    bool bindable ();
    bool allow_nic_name ();
    bool ipv6 ();
    bool expect_port ();
    bool allow_dns ();

  private:
    bool _bindable_wanted;
    bool _nic_name_allowed;
    bool _ipv6_wanted;
    bool _port_expected;
    bool _dns_allowed;
};

class ip_resolver_t
{
  public:
    ip_resolver_t (ip_resolver_options_t opts_);

    int resolve (ip_addr_t *ip_addr_, const char *name_);

  protected:
    //  Virtual functions that are overridden in tests
    virtual int do_getaddrinfo (const char *node_,
                                const char *service_,
                                const struct addrinfo *hints_,
                                struct addrinfo **res_);

    virtual void do_freeaddrinfo (struct addrinfo *res_);

    virtual unsigned int do_if_nametoindex (const char *ifname_);

  private:
    ip_resolver_options_t _options;

    int resolve_nic_name (ip_addr_t *ip_addr_, const char *nic_);
    int resolve_getaddrinfo (ip_addr_t *ip_addr_, const char *addr_);

#if defined ZMQ_HAVE_WINDOWS
    int get_interface_name (unsigned long index_, char **dest_) const;
    int wchar_to_utf8 (const WCHAR *src_, char **dest_) const;
#endif
};
}

#endif


//========= end of #include "ip_resolver.hpp" ============


//========= begin of #include "tcp_address.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TCP_ADDRESS_HPP_INCLUDED__
#define __ZMQ_TCP_ADDRESS_HPP_INCLUDED__

#if !defined ZMQ_HAVE_WINDOWS
#include <sys/socket.h>
#include <netinet/in.h>
#endif

// ans ignore: #include "ip_resolver.hpp"

namespace zmq
{
class tcp_address_t
{
  public:
    tcp_address_t ();
    tcp_address_t (const sockaddr *sa_, socklen_t sa_len_);

    //  This function translates textual TCP address into an address
    //  structure. If 'local' is true, names are resolved as local interface
    //  names. If it is false, names are resolved as remote hostnames.
    //  If 'ipv6' is true, the name may resolve to IPv6 address.
    int resolve (const char *name_, bool local_, bool ipv6_);

    //  The opposite to resolve()
    int to_string (std::string &addr_) const;

#if defined ZMQ_HAVE_WINDOWS
    unsigned short family () const;
#else
    sa_family_t family () const;
#endif
    const sockaddr *addr () const;
    socklen_t addrlen () const;

    const sockaddr *src_addr () const;
    socklen_t src_addrlen () const;
    bool has_src_addr () const;

  private:
    ip_addr_t _address;
    ip_addr_t _source_address;
    bool _has_src_addr;
};

class tcp_address_mask_t
{
  public:
    tcp_address_mask_t ();

    // This function enhances tcp_address_t::resolve() with ability to parse
    // additional cidr-like(/xx) mask value at the end of the name string.
    // Works only with remote hostnames.
    int resolve (const char *name_, bool ipv6_);

    // The opposite to resolve()
    int to_string (std::string &addr_) const;

    int mask () const;

    bool match_address (const struct sockaddr *ss_,
                        const socklen_t ss_len_) const;

  private:
    ip_addr_t _network_address;
    int _address_mask;
};
}

#endif


//========= end of #include "tcp_address.hpp" ============


//========= begin of #include "options.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_OPTIONS_HPP_INCLUDED__
#define __ZMQ_OPTIONS_HPP_INCLUDED__

#include <string>
#include <vector>
#include <map>

// ans ignore: #include "atomic_ptr.hpp"
// ans ignore: #include "stddef.h"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "tcp_address.hpp"

#if defined ZMQ_HAVE_SO_PEERCRED || defined ZMQ_HAVE_LOCAL_PEERCRED
#include <set>
#include <sys/types.h>
#endif
#ifdef ZMQ_HAVE_LOCAL_PEERCRED
#include <sys/ucred.h>
#endif

#if __cplusplus >= 201103L
#include <type_traits>
#endif

//  Normal base 256 key is 32 bytes
#define CURVE_KEYSIZE 32
//  Key encoded using Z85 is 40 bytes
#define CURVE_KEYSIZE_Z85 40

namespace zmq
{
struct options_t
{
    options_t ();

    int set_curve_key (uint8_t *destination_,
                       const void *optval_,
                       size_t optvallen_);

    int setsockopt (int option_, const void *optval_, size_t optvallen_);
    int getsockopt (int option_, void *optval_, size_t *optvallen_) const;

    //  High-water marks for message pipes.
    int sndhwm;
    int rcvhwm;

    //  I/O thread affinity.
    uint64_t affinity;

    //  Socket routing id.
    unsigned char routing_id_size;
    unsigned char routing_id[256];

    //  Maximum transfer rate [kb/s]. Default 100kb/s.
    int rate;

    //  Reliability time interval [ms]. Default 10 seconds.
    int recovery_ivl;

    // Sets the time-to-live field in every multicast packet sent.
    int multicast_hops;

    // Sets the maximum transport data unit size in every multicast
    // packet sent.
    int multicast_maxtpdu;

    // SO_SNDBUF and SO_RCVBUF to be passed to underlying transport sockets.
    int sndbuf;
    int rcvbuf;

    // Type of service (containing DSCP and ECN socket options)
    int tos;

    //  Socket type.
    int8_t type;

    //  Linger time, in milliseconds.
    atomic_value_t linger;

    //  Maximum interval in milliseconds beyond which userspace will
    //  timeout connect().
    //  Default 0 (unused)
    int connect_timeout;

    //  Maximum interval in milliseconds beyond which TCP will timeout
    //  retransmitted packets.
    //  Default 0 (unused)
    int tcp_maxrt;

    //  Minimum interval between attempts to reconnect, in milliseconds.
    //  Default 100ms
    int reconnect_ivl;

    //  Maximum interval between attempts to reconnect, in milliseconds.
    //  Default 0 (unused)
    int reconnect_ivl_max;

    //  Maximum backlog for pending connections.
    int backlog;

    //  Maximal size of message to handle.
    int64_t maxmsgsize;

    // The timeout for send/recv operations for this socket, in milliseconds.
    int rcvtimeo;
    int sndtimeo;

    //  If true, IPv6 is enabled (as well as IPv4)
    bool ipv6;

    //  If 1, connecting pipes are not attached immediately, meaning a send()
    //  on a socket with only connecting pipes would block
    int immediate;

    //  If 1, (X)SUB socket should filter the messages. If 0, it should not.
    bool filter;

    //  If true, the subscription matching on (X)PUB and (X)SUB sockets
    //  is reversed. Messages are sent to and received by non-matching
    //  sockets.
    bool invert_matching;

    //  If true, the routing id message is forwarded to the socket.
    bool recv_routing_id;

    // if true, router socket accepts non-zmq tcp connections
    bool raw_socket;
    bool raw_notify; //  Provide connect notifications

    //  Address of SOCKS proxy
    std::string socks_proxy_address;

    // Credentials for SOCKS proxy.
    // Conneciton method will be basic auth if username
    // is not empty, no auth otherwise.
    std::string socks_proxy_username;
    std::string socks_proxy_password;

    //  TCP keep-alive settings.
    //  Defaults to -1 = do not change socket options
    int tcp_keepalive;
    int tcp_keepalive_cnt;
    int tcp_keepalive_idle;
    int tcp_keepalive_intvl;

    // TCP accept() filters
    typedef std::vector<tcp_address_mask_t> tcp_accept_filters_t;
    tcp_accept_filters_t tcp_accept_filters;

    // IPC accept() filters
#if defined ZMQ_HAVE_SO_PEERCRED || defined ZMQ_HAVE_LOCAL_PEERCRED
    typedef std::set<uid_t> ipc_uid_accept_filters_t;
    ipc_uid_accept_filters_t ipc_uid_accept_filters;
    typedef std::set<gid_t> ipc_gid_accept_filters_t;
    ipc_gid_accept_filters_t ipc_gid_accept_filters;
#endif
#if defined ZMQ_HAVE_SO_PEERCRED
    typedef std::set<pid_t> ipc_pid_accept_filters_t;
    ipc_pid_accept_filters_t ipc_pid_accept_filters;
#endif

    //  Security mechanism for all connections on this socket
    int mechanism;

    //  If peer is acting as server for PLAIN or CURVE mechanisms
    int as_server;

    //  ZAP authentication domain
    std::string zap_domain;

    //  Security credentials for PLAIN mechanism
    std::string plain_username;
    std::string plain_password;

    //  Security credentials for CURVE mechanism
    uint8_t curve_public_key[CURVE_KEYSIZE];
    uint8_t curve_secret_key[CURVE_KEYSIZE];
    uint8_t curve_server_key[CURVE_KEYSIZE];

    //  Principals for GSSAPI mechanism
    std::string gss_principal;
    std::string gss_service_principal;

    //  Name types GSSAPI principals
    int gss_principal_nt;
    int gss_service_principal_nt;

    //  If true, gss encryption will be disabled
    bool gss_plaintext;

    //  ID of the socket.
    int socket_id;

    //  If true, socket conflates outgoing/incoming messages.
    //  Applicable to dealer, push/pull, pub/sub socket types.
    //  Cannot receive multi-part messages.
    //  Ignores hwm
    bool conflate;

    //  If connection handshake is not done after this many milliseconds,
    //  close socket.  Default is 30 secs.  0 means no handshake timeout.
    int handshake_ivl;

    bool connected;
    //  If remote peer receives a PING message and doesn't receive another
    //  message within the ttl value, it should close the connection
    //  (measured in tenths of a second)
    uint16_t heartbeat_ttl;
    //  Time in milliseconds between sending heartbeat PING messages.
    int heartbeat_interval;
    //  Time in milliseconds to wait for a PING response before disconnecting
    int heartbeat_timeout;

#if defined ZMQ_HAVE_VMCI
    uint64_t vmci_buffer_size;
    uint64_t vmci_buffer_min_size;
    uint64_t vmci_buffer_max_size;
    int vmci_connect_timeout;
#endif

    //  When creating a new ZMQ socket, if this option is set the value
    //  will be used as the File Descriptor instead of allocating a new
    //  one via the socket () system call.
    int use_fd;

    // Device to bind the underlying socket to, eg. VRF or interface
    std::string bound_device;

    //  Enforce a non-empty ZAP domain requirement for PLAIN auth
    bool zap_enforce_domain;

    // Use of loopback fastpath.
    bool loopback_fastpath;

    //  Loop sent multicast packets to local sockets
    bool multicast_loop;

    //  Maximal batching size for engines with receiving functionality.
    //  So, if there are 10 messages that fit into the batch size, all of
    //  them may be read by a single 'recv' system call, thus avoiding
    //  unnecessary network stack traversals.
    int in_batch_size;
    //  Maximal batching size for engines with sending functionality.
    //  So, if there are 10 messages that fit into the batch size, all of
    //  them may be written by a single 'send' system call, thus avoiding
    //  unnecessary network stack traversals.
    int out_batch_size;

    // Use zero copy strategy for storing message content when decoding.
    bool zero_copy;

    // Router socket ZMQ_NOTIFY_CONNECT/ZMQ_NOTIFY_DISCONNECT notifications
    int router_notify;

    // Application metadata
    std::map<std::string, std::string> app_metadata;

    // Version of monitor events to emit
    int monitor_event_version;
};

inline bool get_effective_conflate_option (const options_t &options)
{
    // conflate is only effective for some socket types
    return options.conflate
           && (options.type == ZMQ_DEALER || options.type == ZMQ_PULL
               || options.type == ZMQ_PUSH || options.type == ZMQ_PUB
               || options.type == ZMQ_SUB);
}

int do_getsockopt (void *const optval_,
                   size_t *const optvallen_,
                   const void *value_,
                   const size_t value_len_);

template <typename T>
int do_getsockopt (void *const optval_, size_t *const optvallen_, T value_)
{
#if __cplusplus >= 201103L && (!defined(__GNUC__) || __GNUC__ > 5)
    static_assert (std::is_trivially_copyable<T>::value,
                   "invalid use of do_getsockopt");
#endif
    return do_getsockopt (optval_, optvallen_, &value_, sizeof (T));
}

int do_getsockopt (void *const optval_,
                   size_t *const optvallen_,
                   const std::string &value_);

int do_setsockopt_int_as_bool_strict (const void *const optval_,
                                      const size_t optvallen_,
                                      bool *const out_value_);

int do_setsockopt_int_as_bool_relaxed (const void *const optval_,
                                       const size_t optvallen_,
                                       bool *const out_value_);
}

#endif


//========= end of #include "options.hpp" ============


//========= begin of #include "blob.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_BLOB_HPP_INCLUDED__
#define __ZMQ_BLOB_HPP_INCLUDED__

// ans ignore: #include "macros.hpp"
// ans ignore: #include "err.hpp"

#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <ios>

#if __cplusplus >= 201103L || defined(_MSC_VER) && _MSC_VER > 1700
#define ZMQ_HAS_MOVE_SEMANTICS
#define ZMQ_MAP_INSERT_OR_EMPLACE(k, v) emplace (k, v)
#define ZMQ_PUSH_OR_EMPLACE_BACK emplace_back
#define ZMQ_MOVE(x) std::move (x)
#else
#if defined __SUNPRO_CC
template <typename K, typename V>
std::pair<const K, V> make_pair_fix_const (const K &k, const V &v)
{
    return std::pair<const K, V> (k, v);
}

#define ZMQ_MAP_INSERT_OR_EMPLACE(k, v) insert (make_pair_fix_const (k, v))
#else
#define ZMQ_MAP_INSERT_OR_EMPLACE(k, v) insert (std::make_pair (k, v))
#endif

#define ZMQ_PUSH_OR_EMPLACE_BACK push_back
#define ZMQ_MOVE(x) (x)
#endif

namespace zmq
{
struct reference_tag_t
{
};

//  Object to hold dynamically allocated opaque binary data.
//  On modern compilers, it will be movable but not copyable. Copies
//  must be explicitly created by set_deep_copy.
//  On older compilers, it is copyable for syntactical reasons.
struct blob_t
{
    //  Creates an empty blob_t.
    blob_t () : _data (0), _size (0), _owned (true) {}

    //  Creates a blob_t of a given size, with uninitialized content.
    explicit blob_t (const size_t size_) :
        _data (static_cast<unsigned char *> (malloc (size_))),
        _size (size_),
        _owned (true)
    {
        alloc_assert (_data);
    }

    //  Creates a blob_t of a given size, an initializes content by copying
    // from another buffer.
    blob_t (const unsigned char *const data_, const size_t size_) :
        _data (static_cast<unsigned char *> (malloc (size_))),
        _size (size_),
        _owned (true)
    {
        alloc_assert (_data);
        memcpy (_data, data_, size_);
    }

    //  Creates a blob_t for temporary use that only references a
    //  pre-allocated block of data.
    //  Use with caution and ensure that the blob_t will not outlive
    //  the referenced data.
    blob_t (unsigned char *const data_, const size_t size_, reference_tag_t) :
        _data (data_),
        _size (size_),
        _owned (false)
    {
    }

    //  Returns the size of the blob_t.
    size_t size () const { return _size; }

    //  Returns a pointer to the data of the blob_t.
    const unsigned char *data () const { return _data; }

    //  Returns a pointer to the data of the blob_t.
    unsigned char *data () { return _data; }

    //  Defines an order relationship on blob_t.
    bool operator< (blob_t const &other_) const
    {
        const int cmpres =
          memcmp (_data, other_._data, std::min (_size, other_._size));
        return cmpres < 0 || (cmpres == 0 && _size < other_._size);
    }

    //  Sets a blob_t to a deep copy of another blob_t.
    void set_deep_copy (blob_t const &other_)
    {
        clear ();
        _data = static_cast<unsigned char *> (malloc (other_._size));
        alloc_assert (_data);
        _size = other_._size;
        _owned = true;
        memcpy (_data, other_._data, _size);
    }

    //  Sets a blob_t to a copy of a given buffer.
    void set (const unsigned char *const data_, const size_t size_)
    {
        clear ();
        _data = static_cast<unsigned char *> (malloc (size_));
        alloc_assert (_data);
        _size = size_;
        _owned = true;
        memcpy (_data, data_, size_);
    }

    //  Empties a blob_t.
    void clear ()
    {
        if (_owned) {
            free (_data);
        }
        _data = 0;
        _size = 0;
    }

    ~blob_t ()
    {
        if (_owned) {
            free (_data);
        }
    }

#ifdef ZMQ_HAS_MOVE_SEMANTICS
    blob_t (const blob_t &) = delete;
    blob_t &operator= (const blob_t &) = delete;

    blob_t (blob_t &&other_) ZMQ_NOEXCEPT : _data (other_._data),
                                            _size (other_._size),
                                            _owned (other_._owned)
    {
        other_._owned = false;
    }
    blob_t &operator= (blob_t &&other_) ZMQ_NOEXCEPT
    {
        if (this != &other_) {
            clear ();
            _data = other_._data;
            _size = other_._size;
            _owned = other_._owned;
            other_._owned = false;
        }
        return *this;
    }
#else
    blob_t (const blob_t &other) : _owned (false) { set_deep_copy (other); }
    blob_t &operator= (const blob_t &other)
    {
        if (this != &other) {
            clear ();
            set_deep_copy (other);
        }
        return *this;
    }
#endif

  private:
    unsigned char *_data;
    size_t _size;
    bool _owned;
};
}

#endif


//========= end of #include "blob.hpp" ============


//========= begin of #include "mechanism.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MECHANISM_HPP_INCLUDED__
#define __ZMQ_MECHANISM_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "blob.hpp"
// ans ignore: #include "metadata.hpp"

namespace zmq
{
class msg_t;
class session_base_t;

//  Abstract class representing security mechanism.
//  Different mechanism extends this class.

class mechanism_t
{
  public:
    enum status_t
    {
        handshaking,
        ready,
        error
    };

    mechanism_t (const options_t &options_);

    virtual ~mechanism_t ();

    //  Prepare next handshake command that is to be sent to the peer.
    virtual int next_handshake_command (msg_t *msg_) = 0;

    //  Process the handshake command received from the peer.
    virtual int process_handshake_command (msg_t *msg_) = 0;

    virtual int encode (msg_t *) { return 0; }

    virtual int decode (msg_t *) { return 0; }

    //  Notifies mechanism about availability of ZAP message.
    virtual int zap_msg_available () { return 0; }

    //  Returns the status of this mechanism.
    virtual status_t status () const = 0;

    void set_peer_routing_id (const void *id_ptr_, size_t id_size_);

    void peer_routing_id (msg_t *msg_);

    void set_user_id (const void *user_id_, size_t size_);

    const blob_t &get_user_id () const;

    const metadata_t::dict_t &get_zmtp_properties ()
    {
        return _zmtp_properties;
    }

    const metadata_t::dict_t &get_zap_properties () { return _zap_properties; }

  protected:
    //  Only used to identify the socket for the Socket-Type
    //  property in the wire protocol.
    const char *socket_type_string (int socket_type_) const;

    static size_t add_property (unsigned char *ptr_,
                                size_t ptr_capacity_,
                                const char *name_,
                                const void *value_,
                                size_t value_len_);
    static size_t property_len (const char *name_, size_t value_len_);

    size_t add_basic_properties (unsigned char *ptr_,
                                 size_t ptr_capacity_) const;
    size_t basic_properties_len () const;

    void make_command_with_basic_properties (msg_t *msg_,
                                             const char *prefix_,
                                             size_t prefix_len_) const;

    //  Parses a metadata.
    //  Metadata consists of a list of properties consisting of
    //  name and value as size-specified strings.
    //  Returns 0 on success and -1 on error, in which case errno is set.
    int parse_metadata (const unsigned char *ptr_,
                        size_t length_,
                        bool zap_flag_ = false);

    //  This is called by parse_property method whenever it
    //  parses a new property. The function should return 0
    //  on success and -1 on error, in which case it should
    //  set errno. Signaling error prevents parser from
    //  parsing remaining data.
    //  Derived classes are supposed to override this
    //  method to handle custom processing.
    virtual int
    property (const std::string &name_, const void *value_, size_t length_);

    const options_t options;

  private:
    //  Properties received from ZMTP peer.
    metadata_t::dict_t _zmtp_properties;

    //  Properties received from ZAP server.
    metadata_t::dict_t _zap_properties;

    blob_t _routing_id;

    blob_t _user_id;

    //  Returns true iff socket associated with the mechanism
    //  is compatible with a given socket type 'type_'.
    bool check_socket_type (const char *type_, size_t len_) const;
};
}

#endif


//========= end of #include "mechanism.hpp" ============


//========= begin of #include "endpoint.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ENDPOINT_HPP_INCLUDED__
#define __ZMQ_ENDPOINT_HPP_INCLUDED__

#include <string>

namespace zmq
{
enum endpoint_type_t
{
    endpoint_type_none,   // a connection-less endpoint
    endpoint_type_bind,   // a connection-oriented bind endpoint
    endpoint_type_connect // a connection-oriented connect endpoint
};

struct endpoint_uri_pair_t
{
    endpoint_uri_pair_t () : local_type (endpoint_type_none) {}
    endpoint_uri_pair_t (const std::string &local,
                         const std::string &remote,
                         endpoint_type_t local_type) :
        local (local),
        remote (remote),
        local_type (local_type)
    {
    }

    const std::string &identifier () const
    {
        return local_type == endpoint_type_bind ? local : remote;
    }

    std::string local, remote;
    endpoint_type_t local_type;
};

endpoint_uri_pair_t
make_unconnected_connect_endpoint_pair (const std::string &endpoint_);

endpoint_uri_pair_t
make_unconnected_bind_endpoint_pair (const std::string &endpoint_);
}

#endif


//========= end of #include "endpoint.hpp" ============


//========= begin of #include "object.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_OBJECT_HPP_INCLUDED__
#define __ZMQ_OBJECT_HPP_INCLUDED__

#include <string>
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "endpoint.hpp"

namespace zmq
{
struct i_engine;
struct endpoint_t;
struct pending_connection_t;
struct command_t;
class ctx_t;
class pipe_t;
class socket_base_t;
class session_base_t;
class io_thread_t;
class own_t;

//  Base class for all objects that participate in inter-thread
//  communication.

class object_t
{
  public:
    object_t (zmq::ctx_t *ctx_, uint32_t tid_);
    object_t (object_t *parent_);
    virtual ~object_t ();

    uint32_t get_tid ();
    void set_tid (uint32_t id_);
    ctx_t *get_ctx ();
    void process_command (zmq::command_t &cmd_);
    void send_inproc_connected (zmq::socket_base_t *socket_);
    void send_bind (zmq::own_t *destination_,
                    zmq::pipe_t *pipe_,
                    bool inc_seqnum_ = true);

  protected:
    //  Using following function, socket is able to access global
    //  repository of inproc endpoints.
    int register_endpoint (const char *addr_, const zmq::endpoint_t &endpoint_);
    int unregister_endpoint (const std::string &addr_, socket_base_t *socket_);
    void unregister_endpoints (zmq::socket_base_t *socket_);
    zmq::endpoint_t find_endpoint (const char *addr_);
    void pend_connection (const std::string &addr_,
                          const endpoint_t &endpoint_,
                          pipe_t **pipes_);
    void connect_pending (const char *addr_, zmq::socket_base_t *bind_socket_);

    void destroy_socket (zmq::socket_base_t *socket_);

    //  Logs an message.
    void log (const char *format_, ...);

    //  Chooses least loaded I/O thread.
    zmq::io_thread_t *choose_io_thread (uint64_t affinity_);

    //  Derived object can use these functions to send commands
    //  to other objects.
    void send_stop ();
    void send_plug (zmq::own_t *destination_, bool inc_seqnum_ = true);
    void send_own (zmq::own_t *destination_, zmq::own_t *object_);
    void send_attach (zmq::session_base_t *destination_,
                      zmq::i_engine *engine_,
                      bool inc_seqnum_ = true);
    void send_activate_read (zmq::pipe_t *destination_);
    void send_activate_write (zmq::pipe_t *destination_, uint64_t msgs_read_);
    void send_hiccup (zmq::pipe_t *destination_, void *pipe_);
    void send_pipe_peer_stats (zmq::pipe_t *destination_,
                               uint64_t queue_count_,
                               zmq::own_t *socket_base,
                               endpoint_uri_pair_t *endpoint_pair_);
    void send_pipe_stats_publish (zmq::own_t *destination_,
                                  uint64_t outbound_queue_count_,
                                  uint64_t inbound_queue_count_,
                                  endpoint_uri_pair_t *endpoint_pair_);
    void send_pipe_term (zmq::pipe_t *destination_);
    void send_pipe_term_ack (zmq::pipe_t *destination_);
    void send_pipe_hwm (zmq::pipe_t *destination_, int inhwm_, int outhwm_);
    void send_term_req (zmq::own_t *destination_, zmq::own_t *object_);
    void send_term (zmq::own_t *destination_, int linger_);
    void send_term_ack (zmq::own_t *destination_);
    void send_term_endpoint (own_t *destination_, std::string *endpoint_);
    void send_reap (zmq::socket_base_t *socket_);
    void send_reaped ();
    void send_done ();

    //  These handlers can be overridden by the derived objects. They are
    //  called when command arrives from another thread.
    virtual void process_stop ();
    virtual void process_plug ();
    virtual void process_own (zmq::own_t *object_);
    virtual void process_attach (zmq::i_engine *engine_);
    virtual void process_bind (zmq::pipe_t *pipe_);
    virtual void process_activate_read ();
    virtual void process_activate_write (uint64_t msgs_read_);
    virtual void process_hiccup (void *pipe_);
    virtual void process_pipe_peer_stats (uint64_t queue_count_,
                                          zmq::own_t *socket_base_,
                                          endpoint_uri_pair_t *endpoint_pair_);
    virtual void
    process_pipe_stats_publish (uint64_t outbound_queue_count_,
                                uint64_t inbound_queue_count_,
                                endpoint_uri_pair_t *endpoint_pair_);
    virtual void process_pipe_term ();
    virtual void process_pipe_term_ack ();
    virtual void process_pipe_hwm (int inhwm_, int outhwm_);
    virtual void process_term_req (zmq::own_t *object_);
    virtual void process_term (int linger_);
    virtual void process_term_ack ();
    virtual void process_term_endpoint (std::string *endpoint_);
    virtual void process_reap (zmq::socket_base_t *socket_);
    virtual void process_reaped ();

    //  Special handler called after a command that requires a seqnum
    //  was processed. The implementation should catch up with its counter
    //  of processed commands here.
    virtual void process_seqnum ();

  private:
    //  Context provides access to the global state.
    zmq::ctx_t *const _ctx;

    //  Thread ID of the thread the object belongs to.
    uint32_t _tid;

    void send_command (command_t &cmd_);

    object_t (const object_t &);
    const object_t &operator= (const object_t &);
};
}

#endif


//========= end of #include "object.hpp" ============


//========= begin of #include "own.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_OWN_HPP_INCLUDED__
#define __ZMQ_OWN_HPP_INCLUDED__

#include <set>

// ans ignore: #include "object.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "atomic_counter.hpp"
// ans ignore: #include "stdint.hpp"

namespace zmq
{
class ctx_t;
class io_thread_t;

//  Base class for objects forming a part of ownership hierarchy.
//  It handles initialisation and destruction of such objects.

class own_t : public object_t
{
  public:
    //  Note that the owner is unspecified in the constructor.
    //  It'll be supplied later on when the object is plugged in.

    //  The object is not living within an I/O thread. It has it's own
    //  thread outside of 0MQ infrastructure.
    own_t (zmq::ctx_t *parent_, uint32_t tid_);

    //  The object is living within I/O thread.
    own_t (zmq::io_thread_t *io_thread_, const options_t &options_);

    //  When another owned object wants to send command to this object
    //  it calls this function to let it know it should not shut down
    //  before the command is delivered.
    void inc_seqnum ();

    //  Use following two functions to wait for arbitrary events before
    //  terminating. Just add number of events to wait for using
    //  register_tem_acks functions. When event occurs, call
    //  remove_term_ack. When number of pending acks reaches zero
    //  object will be deallocated.
    void register_term_acks (int count_);
    void unregister_term_ack ();

  protected:
    //  Launch the supplied object and become its owner.
    void launch_child (own_t *object_);

    //  Terminate owned object
    void term_child (own_t *object_);

    //  Ask owner object to terminate this object. It may take a while
    //  while actual termination is started. This function should not be
    //  called more than once.
    void terminate ();

    //  Returns true if the object is in process of termination.
    bool is_terminating ();

    //  Derived object destroys own_t. There's no point in allowing
    //  others to invoke the destructor. At the same time, it has to be
    //  virtual so that generic own_t deallocation mechanism destroys
    //  specific type of the owned object correctly.
    virtual ~own_t ();

    //  Term handler is protected rather than private so that it can
    //  be intercepted by the derived class. This is useful to add custom
    //  steps to the beginning of the termination process.
    void process_term (int linger_);

    //  A place to hook in when physical destruction of the object
    //  is to be delayed.
    virtual void process_destroy ();

    //  Socket options associated with this object.
    options_t options;

  private:
    //  Set owner of the object
    void set_owner (own_t *owner_);

    //  Handlers for incoming commands.
    void process_own (own_t *object_);
    void process_term_req (own_t *object_);
    void process_term_ack ();
    void process_seqnum ();

    //  Check whether all the pending term acks were delivered.
    //  If so, deallocate this object.
    void check_term_acks ();

    //  True if termination was already initiated. If so, we can destroy
    //  the object if there are no more child objects or pending term acks.
    bool _terminating;

    //  Sequence number of the last command sent to this object.
    atomic_counter_t _sent_seqnum;

    //  Sequence number of the last command processed by this object.
    uint64_t _processed_seqnum;

    //  Socket owning this object. It's responsible for shutting down
    //  this object.
    own_t *_owner;

    //  List of all objects owned by this socket. We are responsible
    //  for deallocating them before we quit.
    typedef std::set<own_t *> owned_t;
    owned_t _owned;

    //  Number of events we have to get before we can destroy the object.
    int _term_acks;

    own_t (const own_t &);
    const own_t &operator= (const own_t &);
};
}

#endif


//========= end of #include "own.hpp" ============


//========= begin of #include "signaler.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SIGNALER_HPP_INCLUDED__
#define __ZMQ_SIGNALER_HPP_INCLUDED__

#ifdef HAVE_FORK
#include <unistd.h>
#endif

// ans ignore: #include "fd.hpp"

namespace zmq
{
//  This is a cross-platform equivalent to signal_fd. However, as opposed
//  to signal_fd there can be at most one signal in the signaler at any
//  given moment. Attempt to send a signal before receiving the previous
//  one will result in undefined behaviour.

class signaler_t
{
  public:
    signaler_t ();
    ~signaler_t ();

    // Returns the socket/file descriptor
    // May return retired_fd if the signaler could not be initialized.
    fd_t get_fd () const;
    void send ();
    int wait (int timeout_);
    void recv ();
    int recv_failable ();

    bool valid () const;

#ifdef HAVE_FORK
    // close the file descriptors in a forked child process so that they
    // do not interfere with the context in the parent process.
    void forked ();
#endif

  private:
    //  Underlying write & read file descriptor
    //  Will be -1 if an error occurred during initialization, e.g. we
    //  exceeded the number of available handles
    fd_t _w;
    fd_t _r;

    //  Disable copying of signaler_t object.
    signaler_t (const signaler_t &);
    const signaler_t &operator= (const signaler_t &);

#ifdef HAVE_FORK
    // the process that created this context. Used to detect forking.
    pid_t pid;
    // idempotent close of file descriptors that is safe to use by destructor
    // and forked().
    void close_internal ();
#endif
};
}

#endif


//========= end of #include "signaler.hpp" ============


//========= begin of #include "command.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_COMMAND_HPP_INCLUDED__
#define __ZMQ_COMMAND_HPP_INCLUDED__

#include <string>
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "endpoint.hpp"

namespace zmq
{
class object_t;
class own_t;
struct i_engine;
class pipe_t;
class socket_base_t;

//  This structure defines the commands that can be sent between threads.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324) // C4324: alignment padding warnings
__declspec(align (64))
#endif
  struct command_t
{
    //  Object to process the command.
    zmq::object_t *destination;

    enum type_t
    {
        stop,
        plug,
        own,
        attach,
        bind,
        activate_read,
        activate_write,
        hiccup,
        pipe_term,
        pipe_term_ack,
        pipe_hwm,
        term_req,
        term,
        term_ack,
        term_endpoint,
        reap,
        reaped,
        inproc_connected,
        pipe_peer_stats,
        pipe_stats_publish,
        done
    } type;

    union args_t
    {
        //  Sent to I/O thread to let it know that it should
        //  terminate itself.
        struct
        {
        } stop;

        //  Sent to I/O object to make it register with its I/O thread.
        struct
        {
        } plug;

        //  Sent to socket to let it know about the newly created object.
        struct
        {
            zmq::own_t *object;
        } own;

        //  Attach the engine to the session. If engine is NULL, it informs
        //  session that the connection have failed.
        struct
        {
            struct i_engine *engine;
        } attach;

        //  Sent from session to socket to establish pipe(s) between them.
        //  Caller have used inc_seqnum beforehand sending the command.
        struct
        {
            zmq::pipe_t *pipe;
        } bind;

        //  Sent by pipe writer to inform dormant pipe reader that there
        //  are messages in the pipe.
        struct
        {
        } activate_read;

        //  Sent by pipe reader to inform pipe writer about how many
        //  messages it has read so far.
        struct
        {
            uint64_t msgs_read;
        } activate_write;

        //  Sent by pipe reader to writer after creating a new inpipe.
        //  The parameter is actually of type pipe_t::upipe_t, however,
        //  its definition is private so we'll have to do with void*.
        struct
        {
            void *pipe;
        } hiccup;

        //  Sent by pipe reader to pipe writer to ask it to terminate
        //  its end of the pipe.
        struct
        {
        } pipe_term;

        //  Pipe writer acknowledges pipe_term command.
        struct
        {
        } pipe_term_ack;

        //  Sent by one of pipe to another part for modify hwm
        struct
        {
            int inhwm;
            int outhwm;
        } pipe_hwm;

        //  Sent by I/O object ot the socket to request the shutdown of
        //  the I/O object.
        struct
        {
            zmq::own_t *object;
        } term_req;

        //  Sent by socket to I/O object to start its shutdown.
        struct
        {
            int linger;
        } term;

        //  Sent by I/O object to the socket to acknowledge it has
        //  shut down.
        struct
        {
        } term_ack;

        //  Sent by session_base (I/O thread) to socket (application thread)
        //  to ask to disconnect the endpoint.
        struct
        {
            std::string *endpoint;
        } term_endpoint;

        //  Transfers the ownership of the closed socket
        //  to the reaper thread.
        struct
        {
            zmq::socket_base_t *socket;
        } reap;

        //  Closed socket notifies the reaper that it's already deallocated.
        struct
        {
        } reaped;

        //  Send application-side pipe count and ask to send monitor event
        struct
        {
            uint64_t queue_count;
            zmq::own_t *socket_base;
            endpoint_uri_pair_t *endpoint_pair;
        } pipe_peer_stats;

        //  Collate application thread and I/O thread pipe counts and endpoints
        //  and send as event
        struct
        {
            uint64_t outbound_queue_count;
            uint64_t inbound_queue_count;
            endpoint_uri_pair_t *endpoint_pair;
        } pipe_stats_publish;

        //  Sent by reaper thread to the term thread when all the sockets
        //  are successfully deallocated.
        struct
        {
        } done;

    } args;
#ifdef _MSC_VER
};
#pragma warning(pop)
#else
} __attribute__ ((aligned (64)));
#endif
}

#endif


//========= end of #include "command.hpp" ============


//========= begin of #include "yqueue.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_YQUEUE_HPP_INCLUDED__
#define __ZMQ_YQUEUE_HPP_INCLUDED__

#include <stdlib.h>
#include <stddef.h>

// ans ignore: #include "err.hpp"
// ans ignore: #include "atomic_ptr.hpp"

namespace zmq
{
//  yqueue is an efficient queue implementation. The main goal is
//  to minimise number of allocations/deallocations needed. Thus yqueue
//  allocates/deallocates elements in batches of N.
//
//  yqueue allows one thread to use push/back function and another one
//  to use pop/front functions. However, user must ensure that there's no
//  pop on the empty queue and that both threads don't access the same
//  element in unsynchronised manner.
//
//  T is the type of the object in the queue.
//  N is granularity of the queue (how many pushes have to be done till
//  actual memory allocation is required).
#ifdef HAVE_POSIX_MEMALIGN
// ALIGN is the memory alignment size to use in the case where we have
// posix_memalign available. Default value is 64, this alignment will
// prevent two queue chunks from occupying the same CPU cache line on
// architectures where cache lines are <= 64 bytes (e.g. most things
// except POWER). It is detected at build time to try to account for other
// platforms like POWER and s390x.
template <typename T, int N, size_t ALIGN = ZMQ_CACHELINE_SIZE> class yqueue_t
#else
template <typename T, int N> class yqueue_t
#endif
{
  public:
    //  Create the queue.
    inline yqueue_t ()
    {
        _begin_chunk = allocate_chunk ();
        alloc_assert (_begin_chunk);
        _begin_pos = 0;
        _back_chunk = NULL;
        _back_pos = 0;
        _end_chunk = _begin_chunk;
        _end_pos = 0;
    }

    //  Destroy the queue.
    inline ~yqueue_t ()
    {
        while (true) {
            if (_begin_chunk == _end_chunk) {
                free (_begin_chunk);
                break;
            }
            chunk_t *o = _begin_chunk;
            _begin_chunk = _begin_chunk->next;
            free (o);
        }

        chunk_t *sc = _spare_chunk.xchg (NULL);
        free (sc);
    }

    //  Returns reference to the front element of the queue.
    //  If the queue is empty, behaviour is undefined.
    inline T &front () { return _begin_chunk->values[_begin_pos]; }

    //  Returns reference to the back element of the queue.
    //  If the queue is empty, behaviour is undefined.
    inline T &back () { return _back_chunk->values[_back_pos]; }

    //  Adds an element to the back end of the queue.
    inline void push ()
    {
        _back_chunk = _end_chunk;
        _back_pos = _end_pos;

        if (++_end_pos != N)
            return;

        chunk_t *sc = _spare_chunk.xchg (NULL);
        if (sc) {
            _end_chunk->next = sc;
            sc->prev = _end_chunk;
        } else {
            _end_chunk->next = allocate_chunk ();
            alloc_assert (_end_chunk->next);
            _end_chunk->next->prev = _end_chunk;
        }
        _end_chunk = _end_chunk->next;
        _end_pos = 0;
    }

    //  Removes element from the back end of the queue. In other words
    //  it rollbacks last push to the queue. Take care: Caller is
    //  responsible for destroying the object being unpushed.
    //  The caller must also guarantee that the queue isn't empty when
    //  unpush is called. It cannot be done automatically as the read
    //  side of the queue can be managed by different, completely
    //  unsynchronised thread.
    inline void unpush ()
    {
        //  First, move 'back' one position backwards.
        if (_back_pos)
            --_back_pos;
        else {
            _back_pos = N - 1;
            _back_chunk = _back_chunk->prev;
        }

        //  Now, move 'end' position backwards. Note that obsolete end chunk
        //  is not used as a spare chunk. The analysis shows that doing so
        //  would require free and atomic operation per chunk deallocated
        //  instead of a simple free.
        if (_end_pos)
            --_end_pos;
        else {
            _end_pos = N - 1;
            _end_chunk = _end_chunk->prev;
            free (_end_chunk->next);
            _end_chunk->next = NULL;
        }
    }

    //  Removes an element from the front end of the queue.
    inline void pop ()
    {
        if (++_begin_pos == N) {
            chunk_t *o = _begin_chunk;
            _begin_chunk = _begin_chunk->next;
            _begin_chunk->prev = NULL;
            _begin_pos = 0;

            //  'o' has been more recently used than _spare_chunk,
            //  so for cache reasons we'll get rid of the spare and
            //  use 'o' as the spare.
            chunk_t *cs = _spare_chunk.xchg (o);
            free (cs);
        }
    }

  private:
    //  Individual memory chunk to hold N elements.
    struct chunk_t
    {
        T values[N];
        chunk_t *prev;
        chunk_t *next;
    };

    inline chunk_t *allocate_chunk ()
    {
#ifdef HAVE_POSIX_MEMALIGN
        void *pv;
        if (posix_memalign (&pv, ALIGN, sizeof (chunk_t)) == 0)
            return (chunk_t *) pv;
        return NULL;
#else
        return (chunk_t *) malloc (sizeof (chunk_t));
#endif
    }

    //  Back position may point to invalid memory if the queue is empty,
    //  while begin & end positions are always valid. Begin position is
    //  accessed exclusively be queue reader (front/pop), while back and
    //  end positions are accessed exclusively by queue writer (back/push).
    chunk_t *_begin_chunk;
    int _begin_pos;
    chunk_t *_back_chunk;
    int _back_pos;
    chunk_t *_end_chunk;
    int _end_pos;

    //  People are likely to produce and consume at similar rates.  In
    //  this scenario holding onto the most recently freed chunk saves
    //  us from having to call malloc/free.
    atomic_ptr_t<chunk_t> _spare_chunk;

    //  Disable copying of yqueue.
    yqueue_t (const yqueue_t &);
    const yqueue_t &operator= (const yqueue_t &);
};
}

#endif


//========= end of #include "yqueue.hpp" ============


//========= begin of #include "ypipe_base.hpp" ============


/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_YPIPE_BASE_HPP_INCLUDED__
#define __ZMQ_YPIPE_BASE_HPP_INCLUDED__


namespace zmq
{
// ypipe_base abstracts ypipe and ypipe_conflate specific
// classes, one is selected according to a the conflate
// socket option

template <typename T> class ypipe_base_t
{
  public:
    virtual ~ypipe_base_t () {}
    virtual void write (const T &value_, bool incomplete_) = 0;
    virtual bool unwrite (T *value_) = 0;
    virtual bool flush () = 0;
    virtual bool check_read () = 0;
    virtual bool read (T *value_) = 0;
    virtual bool probe (bool (*fn_) (const T &)) = 0;
};
}

#endif


//========= end of #include "ypipe_base.hpp" ============


//========= begin of #include "ypipe.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_YPIPE_HPP_INCLUDED__
#define __ZMQ_YPIPE_HPP_INCLUDED__

// ans ignore: #include "atomic_ptr.hpp"
// ans ignore: #include "yqueue.hpp"
// ans ignore: #include "ypipe_base.hpp"

namespace zmq
{
//  Lock-free queue implementation.
//  Only a single thread can read from the pipe at any specific moment.
//  Only a single thread can write to the pipe at any specific moment.
//  T is the type of the object in the queue.
//  N is granularity of the pipe, i.e. how many items are needed to
//  perform next memory allocation.

template <typename T, int N> class ypipe_t : public ypipe_base_t<T>
{
  public:
    //  Initialises the pipe.
    inline ypipe_t ()
    {
        //  Insert terminator element into the queue.
        _queue.push ();

        //  Let all the pointers to point to the terminator.
        //  (unless pipe is dead, in which case c is set to NULL).
        _r = _w = _f = &_queue.back ();
        _c.set (&_queue.back ());
    }

    //  The destructor doesn't have to be virtual. It is made virtual
    //  just to keep ICC and code checking tools from complaining.
    inline virtual ~ypipe_t () {}

    //  Following function (write) deliberately copies uninitialised data
    //  when used with zmq_msg. Initialising the VSM body for
    //  non-VSM messages won't be good for performance.

#ifdef ZMQ_HAVE_OPENVMS
#pragma message save
#pragma message disable(UNINIT)
#endif

    //  Write an item to the pipe.  Don't flush it yet. If incomplete is
    //  set to true the item is assumed to be continued by items
    //  subsequently written to the pipe. Incomplete items are never
    //  flushed down the stream.
    inline void write (const T &value_, bool incomplete_)
    {
        //  Place the value to the queue, add new terminator element.
        _queue.back () = value_;
        _queue.push ();

        //  Move the "flush up to here" poiter.
        if (!incomplete_)
            _f = &_queue.back ();
    }

#ifdef ZMQ_HAVE_OPENVMS
#pragma message restore
#endif

    //  Pop an incomplete item from the pipe. Returns true if such
    //  item exists, false otherwise.
    inline bool unwrite (T *value_)
    {
        if (_f == &_queue.back ())
            return false;
        _queue.unpush ();
        *value_ = _queue.back ();
        return true;
    }

    //  Flush all the completed items into the pipe. Returns false if
    //  the reader thread is sleeping. In that case, caller is obliged to
    //  wake the reader up before using the pipe again.
    inline bool flush ()
    {
        //  If there are no un-flushed items, do nothing.
        if (_w == _f)
            return true;

        //  Try to set 'c' to 'f'.
        if (_c.cas (_w, _f) != _w) {
            //  Compare-and-swap was unseccessful because 'c' is NULL.
            //  This means that the reader is asleep. Therefore we don't
            //  care about thread-safeness and update c in non-atomic
            //  manner. We'll return false to let the caller know
            //  that reader is sleeping.
            _c.set (_f);
            _w = _f;
            return false;
        }

        //  Reader is alive. Nothing special to do now. Just move
        //  the 'first un-flushed item' pointer to 'f'.
        _w = _f;
        return true;
    }

    //  Check whether item is available for reading.
    inline bool check_read ()
    {
        //  Was the value prefetched already? If so, return.
        if (&_queue.front () != _r && _r)
            return true;

        //  There's no prefetched value, so let us prefetch more values.
        //  Prefetching is to simply retrieve the
        //  pointer from c in atomic fashion. If there are no
        //  items to prefetch, set c to NULL (using compare-and-swap).
        _r = _c.cas (&_queue.front (), NULL);

        //  If there are no elements prefetched, exit.
        //  During pipe's lifetime r should never be NULL, however,
        //  it can happen during pipe shutdown when items
        //  are being deallocated.
        if (&_queue.front () == _r || !_r)
            return false;

        //  There was at least one value prefetched.
        return true;
    }

    //  Reads an item from the pipe. Returns false if there is no value.
    //  available.
    inline bool read (T *value_)
    {
        //  Try to prefetch a value.
        if (!check_read ())
            return false;

        //  There was at least one value prefetched.
        //  Return it to the caller.
        *value_ = _queue.front ();
        _queue.pop ();
        return true;
    }

    //  Applies the function fn to the first elemenent in the pipe
    //  and returns the value returned by the fn.
    //  The pipe mustn't be empty or the function crashes.
    inline bool probe (bool (*fn_) (const T &))
    {
        bool rc = check_read ();
        zmq_assert (rc);

        return (*fn_) (_queue.front ());
    }

  protected:
    //  Allocation-efficient queue to store pipe items.
    //  Front of the queue points to the first prefetched item, back of
    //  the pipe points to last un-flushed item. Front is used only by
    //  reader thread, while back is used only by writer thread.
    yqueue_t<T, N> _queue;

    //  Points to the first un-flushed item. This variable is used
    //  exclusively by writer thread.
    T *_w;

    //  Points to the first un-prefetched item. This variable is used
    //  exclusively by reader thread.
    T *_r;

    //  Points to the first item to be flushed in the future.
    T *_f;

    //  The single point of contention between writer and reader thread.
    //  Points past the last flushed item. If it is NULL,
    //  reader is asleep. This pointer should be always accessed using
    //  atomic operations.
    atomic_ptr_t<T> _c;

    //  Disable copying of ypipe object.
    ypipe_t (const ypipe_t &);
    const ypipe_t &operator= (const ypipe_t &);
};
}

#endif


//========= end of #include "ypipe.hpp" ============


//========= begin of #include "i_mailbox.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_I_MAILBOX_HPP_INCLUDED__
#define __ZMQ_I_MAILBOX_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"

namespace zmq
{
//  Interface to be implemented by mailbox.

class i_mailbox
{
  public:
    virtual ~i_mailbox () {}

    virtual void send (const command_t &cmd_) = 0;
    virtual int recv (command_t *cmd_, int timeout_) = 0;


#ifdef HAVE_FORK
    // close the file descriptors in the signaller. This is used in a forked
    // child process to close the file descriptors so that they do not interfere
    // with the context in the parent process.
    virtual void forked () = 0;
#endif
};
}

#endif


//========= end of #include "i_mailbox.hpp" ============


//========= begin of #include "mailbox.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MAILBOX_HPP_INCLUDED__
#define __ZMQ_MAILBOX_HPP_INCLUDED__

#include <stddef.h>

// ans ignore: #include "signaler.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "config.hpp"
// ans ignore: #include "command.hpp"
// ans ignore: #include "ypipe.hpp"
// ans ignore: #include "mutex.hpp"
// ans ignore: #include "i_mailbox.hpp"

namespace zmq
{
class mailbox_t : public i_mailbox
{
  public:
    mailbox_t ();
    ~mailbox_t ();

    fd_t get_fd () const;
    void send (const command_t &cmd_);
    int recv (command_t *cmd_, int timeout_);

    bool valid () const;

#ifdef HAVE_FORK
    // close the file descriptors in the signaller. This is used in a forked
    // child process to close the file descriptors so that they do not interfere
    // with the context in the parent process.
    void forked () { _signaler.forked (); }
#endif

  private:
    //  The pipe to store actual commands.
    typedef ypipe_t<command_t, command_pipe_granularity> cpipe_t;
    cpipe_t _cpipe;

    //  Signaler to pass signals from writer thread to reader thread.
    signaler_t _signaler;

    //  There's only one thread receiving from the mailbox, but there
    //  is arbitrary number of threads sending. Given that ypipe requires
    //  synchronised access on both of its endpoints, we have to synchronise
    //  the sending side.
    mutex_t _sync;

    //  True if the underlying pipe is active, ie. when we are allowed to
    //  read commands from it.
    bool _active;

    //  Disable copying of mailbox_t object.
    mailbox_t (const mailbox_t &);
    const mailbox_t &operator= (const mailbox_t &);
};
}

#endif


//========= end of #include "mailbox.hpp" ============


//========= begin of #include "array.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ARRAY_INCLUDED__
#define __ZMQ_ARRAY_INCLUDED__

#include <vector>
#include <algorithm>

namespace zmq
{
//  Implementation of fast arrays with O(1) access, insertion and
//  removal. The array stores pointers rather than objects.
//  O(1) is achieved by making items inheriting from
//  array_item_t<ID> class which internally stores the position
//  in the array.
//  The ID template argument is used to differentiate among arrays
//  and thus let an object be stored in different arrays.

//  Base class for objects stored in the array. If you want to store
//  same object in multiple arrays, each of those arrays has to have
//  different ID. The item itself has to be derived from instantiations of
//  array_item_t template for all relevant IDs.

template <int ID = 0> class array_item_t
{
  public:
    inline array_item_t () : _array_index (-1) {}

    //  The destructor doesn't have to be virtual. It is made virtual
    //  just to keep ICC and code checking tools from complaining.
    inline virtual ~array_item_t () {}

    inline void set_array_index (int index_) { _array_index = index_; }

    inline int get_array_index () { return _array_index; }

  private:
    int _array_index;

    array_item_t (const array_item_t &);
    const array_item_t &operator= (const array_item_t &);
};


template <typename T, int ID = 0> class array_t
{
  private:
    typedef array_item_t<ID> item_t;

  public:
    typedef typename std::vector<T *>::size_type size_type;

    inline array_t () {}

    inline ~array_t () {}

    inline size_type size () { return _items.size (); }

    inline bool empty () { return _items.empty (); }

    inline T *&operator[] (size_type index_) { return _items[index_]; }

    inline void push_back (T *item_)
    {
        if (item_)
            ((item_t *) item_)->set_array_index ((int) _items.size ());
        _items.push_back (item_);
    }

    inline void erase (T *item_)
    {
        erase (((item_t *) item_)->get_array_index ());
    }

    inline void erase (size_type index_)
    {
        if (_items.back ())
            ((item_t *) _items.back ())->set_array_index ((int) index_);
        _items[index_] = _items.back ();
        _items.pop_back ();
    }

    inline void swap (size_type index1_, size_type index2_)
    {
        if (_items[index1_])
            ((item_t *) _items[index1_])->set_array_index ((int) index2_);
        if (_items[index2_])
            ((item_t *) _items[index2_])->set_array_index ((int) index1_);
        std::swap (_items[index1_], _items[index2_]);
    }

    inline void clear () { _items.clear (); }

    inline size_type index (T *item_)
    {
        return (size_type) ((item_t *) item_)->get_array_index ();
    }

  private:
    typedef std::vector<T *> items_t;
    items_t _items;

    array_t (const array_t &);
    const array_t &operator= (const array_t &);
};
}

#endif


//========= end of #include "array.hpp" ============


//========= begin of #include "thread.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_THREAD_HPP_INCLUDED__
#define __ZMQ_THREAD_HPP_INCLUDED__

#if defined ZMQ_HAVE_VXWORKS
#include <vxWorks.h>
#include <taskLib.h>
#elif !defined ZMQ_HAVE_WINDOWS
#include <pthread.h>
#endif
#include <set>
#include <cstring>

namespace zmq
{
typedef void(thread_fn) (void *);

//  Class encapsulating OS thread. Thread initiation/termination is done
//  using special functions rather than in constructor/destructor so that
//  thread isn't created during object construction by accident, causing
//  newly created thread to access half-initialised object. Same applies
//  to the destruction process: Thread should be terminated before object
//  destruction begins, otherwise it can access half-destructed object.

class thread_t
{
  public:
    inline thread_t () :
        _tfn (NULL),
        _arg (NULL),
        _started (false),
        _thread_priority (ZMQ_THREAD_PRIORITY_DFLT),
        _thread_sched_policy (ZMQ_THREAD_SCHED_POLICY_DFLT)
    {
        memset (_name, 0, sizeof (_name));
    }

#ifdef ZMQ_HAVE_VXWORKS
    ~thread_t ()
    {
        if (descriptor != NULL || descriptor > 0) {
            taskDelete (descriptor);
        }
    }
#endif

    //  Creates OS thread. 'tfn' is main thread function. It'll be passed
    //  'arg' as an argument.
    //  Name is 16 characters max including terminating NUL. Thread naming is
    //  implemented only for pthread, and windows when a debugger is attached.
    void start (thread_fn *tfn_, void *arg_, const char *name_);

    //  Returns whether the thread was started, i.e. start was called.
    bool get_started () const;

    //  Returns whether the executing thread is the thread represented by the
    //  thread object.
    bool is_current_thread () const;

    //  Waits for thread termination.
    void stop ();

    // Sets the thread scheduling parameters. Only implemented for
    // pthread. Has no effect on other platforms.
    void setSchedulingParameters (int priority_,
                                  int scheduling_policy_,
                                  const std::set<int> &affinity_cpus_);

    //  These are internal members. They should be private, however then
    //  they would not be accessible from the main C routine of the thread.
    void applySchedulingParameters ();
    void applyThreadName ();
    thread_fn *_tfn;
    void *_arg;
    char _name[16];

  private:
    bool _started;

#ifdef ZMQ_HAVE_WINDOWS
    HANDLE _descriptor;
#elif defined ZMQ_HAVE_VXWORKS
    int _descriptor;
    enum
    {
        DEFAULT_PRIORITY = 100,
        DEFAULT_OPTIONS = 0,
        DEFAULT_STACK_SIZE = 4000
    };
#else
    pthread_t _descriptor;
#endif

    //  Thread scheduling parameters.
    int _thread_priority;
    int _thread_sched_policy;
    std::set<int> _thread_affinity_cpus;

    thread_t (const thread_t &);
    const thread_t &operator= (const thread_t &);
};
}

#endif


//========= end of #include "thread.hpp" ============


//========= begin of #include "ctx.hpp" ============

/*
    Copyright (c) 2007-2017 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CTX_HPP_INCLUDED__
#define __ZMQ_CTX_HPP_INCLUDED__

#include <map>
#include <vector>
#include <string>
#include <stdarg.h>

// ans ignore: #include "mailbox.hpp"
// ans ignore: #include "array.hpp"
// ans ignore: #include "config.hpp"
// ans ignore: #include "mutex.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "atomic_counter.hpp"
// ans ignore: #include "thread.hpp"

namespace zmq
{
class object_t;
class io_thread_t;
class socket_base_t;
class reaper_t;
class pipe_t;

//  Information associated with inproc endpoint. Note that endpoint options
//  are registered as well so that the peer can access them without a need
//  for synchronisation, handshaking or similar.
struct endpoint_t
{
    socket_base_t *socket;
    options_t options;
};

class thread_ctx_t
{
  public:
    thread_ctx_t ();

    //  Start a new thread with proper scheduling parameters.
    void start_thread (thread_t &thread_,
                       thread_fn *tfn_,
                       void *arg_,
                       const char *name_ = NULL) const;

    int set (int option_, int optval_);
    int get (int option_);

  protected:
    //  Synchronisation of access to context options.
    mutex_t _opt_sync;

  private:
    //  Thread parameters.
    int _thread_priority;
    int _thread_sched_policy;
    std::set<int> _thread_affinity_cpus;
    std::string _thread_name_prefix;
};

//  Context object encapsulates all the global state associated with
//  the library.

class ctx_t : public thread_ctx_t
{
  public:
    //  Create the context object.
    ctx_t ();

    //  Returns false if object is not a context.
    bool check_tag ();

    //  This function is called when user invokes zmq_ctx_term. If there are
    //  no more sockets open it'll cause all the infrastructure to be shut
    //  down. If there are open sockets still, the deallocation happens
    //  after the last one is closed.
    int terminate ();

    // This function starts the terminate process by unblocking any blocking
    // operations currently in progress and stopping any more socket activity
    // (except zmq_close).
    // This function is non-blocking.
    // terminate must still be called afterwards.
    // This function is optional, terminate will unblock any current
    // operations as well.
    int shutdown ();

    //  Set and get context properties.
    int set (int option_, int optval_);
    int get (int option_);

    //  Create and destroy a socket.
    zmq::socket_base_t *create_socket (int type_);
    void destroy_socket (zmq::socket_base_t *socket_);

    //  Send command to the destination thread.
    void send_command (uint32_t tid_, const command_t &command_);

    //  Returns the I/O thread that is the least busy at the moment.
    //  Affinity specifies which I/O threads are eligible (0 = all).
    //  Returns NULL if no I/O thread is available.
    zmq::io_thread_t *choose_io_thread (uint64_t affinity_);

    //  Returns reaper thread object.
    zmq::object_t *get_reaper ();

    //  Management of inproc endpoints.
    int register_endpoint (const char *addr_, const endpoint_t &endpoint_);
    int unregister_endpoint (const std::string &addr_, socket_base_t *socket_);
    void unregister_endpoints (zmq::socket_base_t *socket_);
    endpoint_t find_endpoint (const char *addr_);
    void pend_connection (const std::string &addr_,
                          const endpoint_t &endpoint_,
                          pipe_t **pipes_);
    void connect_pending (const char *addr_, zmq::socket_base_t *bind_socket_);

#ifdef ZMQ_HAVE_VMCI
    // Return family for the VMCI socket or -1 if it's not available.
    int get_vmci_socket_family ();
#endif

    enum
    {
        term_tid = 0,
        reaper_tid = 1
    };

    ~ctx_t ();

    bool valid () const;

  private:
    bool start ();

    struct pending_connection_t
    {
        endpoint_t endpoint;
        pipe_t *connect_pipe;
        pipe_t *bind_pipe;
    };

    //  Used to check whether the object is a context.
    uint32_t _tag;

    //  Sockets belonging to this context. We need the list so that
    //  we can notify the sockets when zmq_ctx_term() is called.
    //  The sockets will return ETERM then.
    typedef array_t<socket_base_t> sockets_t;
    sockets_t _sockets;

    //  List of unused thread slots.
    typedef std::vector<uint32_t> empty_slots_t;
    empty_slots_t _empty_slots;

    //  If true, zmq_init has been called but no socket has been created
    //  yet. Launching of I/O threads is delayed.
    bool _starting;

    //  If true, zmq_ctx_term was already called.
    bool _terminating;

    //  Synchronisation of accesses to global slot-related data:
    //  sockets, empty_slots, terminating. It also synchronises
    //  access to zombie sockets as such (as opposed to slots) and provides
    //  a memory barrier to ensure that all CPU cores see the same data.
    mutex_t _slot_sync;

    //  The reaper thread.
    zmq::reaper_t *_reaper;

    //  I/O threads.
    typedef std::vector<zmq::io_thread_t *> io_threads_t;
    io_threads_t _io_threads;

    //  Array of pointers to mailboxes for both application and I/O threads.
    std::vector<i_mailbox *> _slots;

    //  Mailbox for zmq_ctx_term thread.
    mailbox_t _term_mailbox;

    //  List of inproc endpoints within this context.
    typedef std::map<std::string, endpoint_t> endpoints_t;
    endpoints_t _endpoints;

    // List of inproc connection endpoints pending a bind
    typedef std::multimap<std::string, pending_connection_t>
      pending_connections_t;
    pending_connections_t _pending_connections;

    //  Synchronisation of access to the list of inproc endpoints.
    mutex_t _endpoints_sync;

    //  Maximum socket ID.
    static atomic_counter_t max_socket_id;

    //  Maximum number of sockets that can be opened at the same time.
    int _max_sockets;

    //  Maximum allowed message size
    int _max_msgsz;

    //  Number of I/O threads to launch.
    int _io_thread_count;

    //  Does context wait (possibly forever) on termination?
    bool _blocky;

    //  Is IPv6 enabled on this context?
    bool _ipv6;

    // Should we use zero copy message decoding in this context?
    bool _zero_copy;

    ctx_t (const ctx_t &);
    const ctx_t &operator= (const ctx_t &);

#ifdef HAVE_FORK
    // the process that created this context. Used to detect forking.
    pid_t _pid;
#endif
    enum side
    {
        connect_side,
        bind_side
    };
    void
    connect_inproc_sockets (zmq::socket_base_t *bind_socket_,
                            options_t &bind_options_,
                            const pending_connection_t &pending_connection_,
                            side side_);

#ifdef ZMQ_HAVE_VMCI
    int _vmci_fd;
    int _vmci_family;
    mutex_t _vmci_sync;
#endif
};
}

#endif


//========= end of #include "ctx.hpp" ============


//========= begin of #include "clock.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CLOCK_HPP_INCLUDED__
#define __ZMQ_CLOCK_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"

#if defined ZMQ_HAVE_OSX
// TODO this is not required in this file, but condition_variable.hpp includes
// clock.hpp to get these definitions
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef HAVE_CLOCK_GETTIME
#define HAVE_CLOCK_GETTIME
#endif

#include <mach/clock.h>
#include <mach/mach.h>
#include <time.h>
#include <sys/time.h>
#endif

namespace zmq
{
class clock_t
{
  public:
    clock_t ();

    //  CPU's timestamp counter. Returns 0 if it's not available.
    static uint64_t rdtsc ();

    //  High precision timestamp.
    static uint64_t now_us ();

    //  Low precision timestamp. In tight loops generating it can be
    //  10 to 100 times faster than the high precision timestamp.
    uint64_t now_ms ();

  private:
    //  TSC timestamp of when last time measurement was made.
    uint64_t _last_tsc;

    //  Physical time corresponding to the TSC above (in milliseconds).
    uint64_t _last_time;

    clock_t (const clock_t &);
    const clock_t &operator= (const clock_t &);
};
}

#endif


//========= end of #include "clock.hpp" ============


//========= begin of #include "poller_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_POLLER_BASE_HPP_INCLUDED__
#define __ZMQ_POLLER_BASE_HPP_INCLUDED__

#include <map>

// ans ignore: #include "clock.hpp"
// ans ignore: #include "atomic_counter.hpp"
// ans ignore: #include "ctx.hpp"

namespace zmq
{
struct i_poll_events;

// A build of libzmq must provide an implementation of the poller_t concept. By
// convention, this is done via a typedef.
//
// At the time of writing, the following implementations of the poller_t
// concept exist: zmq::devpoll_t, zmq::epoll_t, zmq::kqueue_t, zmq::poll_t,
// zmq::pollset_t, zmq::select_t
//
// An implementation of the poller_t concept must provide the following public
// methods:
//   Returns load of the poller.
// int get_load() const;
//
//   Add a timeout to expire in timeout_ milliseconds. After the
//   expiration, timer_event on sink_ object will be called with
//   argument set to id_.
// void add_timer(int timeout_, zmq::i_poll_events *sink_, int id_);
//
//   Cancel the timer created by sink_ object with ID equal to id_.
// void cancel_timer(zmq::i_poll_events *sink_, int id_);
//
//   Adds a fd to the poller. Initially, no events are activated. These must
//   be activated by the set_* methods using the returned handle_.
// handle_t add_fd(fd_t fd_, zmq::i_poll_events *events_);
//
//   Deactivates any events that may be active for the given handle_, and
//   removes the fd associated with the given handle_.
// void rm_fd(handle_t handle_);
//
//   The set_* and reset_* methods activate resp. deactivate polling for
//   input/output readiness on the respective handle_, such that the
//   in_event/out_event methods on the associated zmq::i_poll_events object
//   will be called.
//   Note: while handle_t and fd_t may be the same type, and may even have the
//   same values for some implementation, this may not be assumed in general.
//   The methods may only be called with the handle returned by add_fd.
// void set_pollin(handle_t handle_);
// void reset_pollin(handle_t handle_);
// void set_pollout(handle_t handle_);//
// void reset_pollout(handle_t handle_);
//
//   Starts operation of the poller. See below for details.
// void start();
//
//   Request termination of the poller.
//   TODO: might be removed in the future, as it has no effect.
// void stop();
//
//   Returns the maximum number of fds that can be added to an instance of the
//   poller at the same time, or -1 if there is no such fixed limit.
// static int max_fds();
//
// Most of the methods may only be called from a zmq::i_poll_events callback
// function when invoked by the poller (and, therefore, typically from the
// poller's worker thread), with the following exceptions:
// - get_load may be called from outside
// - add_fd and add_timer may be called from outside before start
// - start may be called from outside once
//
// After a poller is started, it waits for the registered events (input/output
// readiness, timeout) to happen, and calls the respective functions on the
// zmq::i_poll_events object. It terminates when no further registrations (fds
// or timers) exist.
//
// Before start, add_fd must have been called at least once. Behavior may be
// undefined otherwise.
//
// If the poller is implemented by a single worker thread (the
// worker_poller_base_t base  class may be used to implement such a poller),
// no synchronization is required for the data structures modified by
// add_fd, rm_fd, add_timer, cancel_timer, (re)set_poll(in|out). However,
// reentrancy must be considered, e.g. when one of the functions modifies
// a container that is being iterated by the poller.


// A class that can be used as a base class for implementations of the poller
// concept.
//
// For documentation of the public methods, see the description of the poller_t
// concept.
class poller_base_t
{
  public:
    poller_base_t ();
    virtual ~poller_base_t ();

    // Methods from the poller concept.
    int get_load () const;
    void add_timer (int timeout_, zmq::i_poll_events *sink_, int id_);
    void cancel_timer (zmq::i_poll_events *sink_, int id_);

  protected:
    //  Called by individual poller implementations to manage the load.
    void adjust_load (int amount_);

    //  Executes any timers that are due. Returns number of milliseconds
    //  to wait to match the next timer or 0 meaning "no timers".
    uint64_t execute_timers ();

  private:
    //  Clock instance private to this I/O thread.
    clock_t _clock;

    //  List of active timers.
    struct timer_info_t
    {
        zmq::i_poll_events *sink;
        int id;
    };
    typedef std::multimap<uint64_t, timer_info_t> timers_t;
    timers_t _timers;

    //  Load of the poller. Currently the number of file descriptors
    //  registered.
    atomic_counter_t _load;

    poller_base_t (const poller_base_t &);
    const poller_base_t &operator= (const poller_base_t &);
};

//  Base class for a poller with a single worker thread.
class worker_poller_base_t : public poller_base_t
{
  public:
    worker_poller_base_t (const thread_ctx_t &ctx_);

    // Methods from the poller concept.
    void start (const char *name = NULL);

  protected:
    //  Checks whether the currently executing thread is the worker thread
    //  via an assertion.
    //  Should be called by the add_fd, removed_fd, set_*, reset_* functions
    //  to ensure correct usage.
    void check_thread ();

    //  Stops the worker thread. Should be called from the destructor of the
    //  leaf class.
    void stop_worker ();

  private:
    //  Main worker thread routine.
    static void worker_routine (void *arg_);

    virtual void loop () = 0;

    // Reference to ZMQ context.
    const thread_ctx_t &_ctx;

    //  Handle of the physical thread doing the I/O work.
    thread_t _worker;
};
}

#endif


//========= end of #include "poller_base.hpp" ============


//========= begin of #include "kqueue.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_KQUEUE_HPP_INCLUDED__
#define __ZMQ_KQUEUE_HPP_INCLUDED__

//  poller.hpp decides which polling mechanism to use.
// ans ignore: #include "poller.hpp"
#if defined ZMQ_IOTHREAD_POLLER_USE_KQUEUE

#include <vector>
#include <unistd.h>

// ans ignore: #include "ctx.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "thread.hpp"
// ans ignore: #include "poller_base.hpp"

namespace zmq
{
struct i_poll_events;

//  Implements socket polling mechanism using the BSD-specific
//  kqueue interface.

class kqueue_t : public worker_poller_base_t
{
  public:
    typedef void *handle_t;

    kqueue_t (const thread_ctx_t &ctx_);
    ~kqueue_t ();

    //  "poller" concept.
    handle_t add_fd (fd_t fd_, zmq::i_poll_events *events_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void stop ();

    static int max_fds ();

  private:
    //  Main event loop.
    void loop ();

    //  File descriptor referring to the kernel event queue.
    fd_t kqueue_fd;

    //  Adds the event to the kqueue.
    void kevent_add (fd_t fd_, short filter_, void *udata_);

    //  Deletes the event from the kqueue.
    void kevent_delete (fd_t fd_, short filter_);

    struct poll_entry_t
    {
        fd_t fd;
        bool flag_pollin;
        bool flag_pollout;
        zmq::i_poll_events *reactor;
    };

    //  List of retired event sources.
    typedef std::vector<poll_entry_t *> retired_t;
    retired_t retired;

    kqueue_t (const kqueue_t &);
    const kqueue_t &operator= (const kqueue_t &);

#ifdef HAVE_FORK
    // the process that created this context. Used to detect forking.
    pid_t pid;
#endif
};

typedef kqueue_t poller_t;
}

#endif

#endif


//========= end of #include "kqueue.hpp" ============


//========= begin of #include "epoll.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_EPOLL_HPP_INCLUDED__
#define __ZMQ_EPOLL_HPP_INCLUDED__

//  poller.hpp decides which polling mechanism to use.
// ans ignore: #include "poller.hpp"
#if defined ZMQ_IOTHREAD_POLLER_USE_EPOLL

#include <vector>

#if defined ZMQ_HAVE_WINDOWS
// ans ignore: #include "../external/wepoll/wepoll.h"
#else
#include <sys/epoll.h>
#endif

// ans ignore: #include "ctx.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "thread.hpp"
// ans ignore: #include "poller_base.hpp"
// ans ignore: #include "mutex.hpp"

namespace zmq
{
struct i_poll_events;

//  This class implements socket polling mechanism using the Linux-specific
//  epoll mechanism.

class epoll_t : public worker_poller_base_t
{
  public:
    typedef void *handle_t;

    epoll_t (const thread_ctx_t &ctx_);
    ~epoll_t ();

    //  "poller" concept.
    handle_t add_fd (fd_t fd_, zmq::i_poll_events *events_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void stop ();

    static int max_fds ();

  private:
#if defined ZMQ_HAVE_WINDOWS
    typedef HANDLE epoll_fd_t;
    static const epoll_fd_t epoll_retired_fd;
#else
    typedef fd_t epoll_fd_t;
    enum
    {
        epoll_retired_fd = retired_fd
    };
#endif

    //  Main event loop.
    void loop ();

    //  Main epoll file descriptor
    epoll_fd_t _epoll_fd;

    struct poll_entry_t
    {
        fd_t fd;
        epoll_event ev;
        zmq::i_poll_events *events;
    };

    //  List of retired event sources.
    typedef std::vector<poll_entry_t *> retired_t;
    retired_t _retired;

    epoll_t (const epoll_t &);
    const epoll_t &operator= (const epoll_t &);
};

typedef epoll_t poller_t;
}

#endif

#endif


//========= end of #include "epoll.hpp" ============


//========= begin of #include "devpoll.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DEVPOLL_HPP_INCLUDED__
#define __ZMQ_DEVPOLL_HPP_INCLUDED__

//  poller.hpp decides which polling mechanism to use.
// ans ignore: #include "poller.hpp"
#if defined ZMQ_IOTHREAD_POLLER_USE_DEVPOLL

#include <vector>

// ans ignore: #include "ctx.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "thread.hpp"
// ans ignore: #include "poller_base.hpp"

namespace zmq
{
struct i_poll_events;

//  Implements socket polling mechanism using the "/dev/poll" interface.

class devpoll_t : public worker_poller_base_t
{
  public:
    typedef fd_t handle_t;

    devpoll_t (const thread_ctx_t &ctx_);
    ~devpoll_t ();

    //  "poller" concept.
    handle_t add_fd (fd_t fd_, zmq::i_poll_events *events_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void stop ();

    static int max_fds ();

  private:
    //  Main event loop.
    void loop ();

    //  File descriptor referring to "/dev/poll" pseudo-device.
    fd_t devpoll_fd;

    struct fd_entry_t
    {
        short events;
        zmq::i_poll_events *reactor;
        bool valid;
        bool accepted;
    };

    typedef std::vector<fd_entry_t> fd_table_t;
    fd_table_t fd_table;

    typedef std::vector<fd_t> pending_list_t;
    pending_list_t pending_list;

    //  Pollset manipulation function.
    void devpoll_ctl (fd_t fd_, short events_);

    devpoll_t (const devpoll_t &);
    const devpoll_t &operator= (const devpoll_t &);
};

typedef devpoll_t poller_t;
}

#endif

#endif


//========= end of #include "devpoll.hpp" ============


//========= begin of #include "pollset.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_POLLSET_HPP_INCLUDED__
#define __ZMQ_POLLSET_HPP_INCLUDED__

//  poller.hpp decides which polling mechanism to use.
// ans ignore: #include "poller.hpp"
#if defined ZMQ_IOTHREAD_POLLER_USE_POLLSET

#include <sys/poll.h>
#include <sys/pollset.h>
#include <vector>

// ans ignore: #include "ctx.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "thread.hpp"
// ans ignore: #include "poller_base.hpp"

namespace zmq
{
struct i_poll_events;

//  This class implements socket polling mechanism using the AIX-specific
//  pollset mechanism.

class pollset_t : public poller_base_t
{
  public:
    typedef void *handle_t;

    pollset_t (const thread_ctx_t &ctx_);
    ~pollset_t ();

    //  "poller" concept.
    handle_t add_fd (fd_t fd_, zmq::i_poll_events *events_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void start ();
    void stop ();

    static int max_fds ();

  private:
    //  Main worker thread routine.
    static void worker_routine (void *arg_);

    //  Main event loop.
    void loop ();

    // Reference to ZMQ context.
    const thread_ctx_t &ctx;

    //  Main pollset file descriptor
    ::pollset_t pollset_fd;

    struct poll_entry_t
    {
        fd_t fd;
        bool flag_pollin;
        bool flag_pollout;
        zmq::i_poll_events *events;
    };

    //  List of retired event sources.
    typedef std::vector<poll_entry_t *> retired_t;
    retired_t retired;

    //  This table stores data for registered descriptors.
    typedef std::vector<poll_entry_t *> fd_table_t;
    fd_table_t fd_table;

    //  If true, thread is in the process of shutting down.
    bool stopping;

    //  Handle of the physical thread doing the I/O work.
    thread_t worker;

    pollset_t (const pollset_t &);
    const pollset_t &operator= (const pollset_t &);
};

typedef pollset_t poller_t;
}

#endif

#endif


//========= end of #include "pollset.hpp" ============


//========= begin of #include "poll.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_POLL_HPP_INCLUDED__
#define __ZMQ_POLL_HPP_INCLUDED__

//  poller.hpp decides which polling mechanism to use.
// ans ignore: #include "poller.hpp"
#if defined ZMQ_IOTHREAD_POLLER_USE_POLL

#if defined ZMQ_HAVE_WINDOWS
#error                                                                         \
  "poll is broken on Windows for the purpose of the I/O thread poller, use select instead; "\
  "see https://github.com/zeromq/libzmq/issues/3107"
#endif

#include <poll.h>
#include <stddef.h>
#include <vector>

// ans ignore: #include "ctx.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "thread.hpp"
// ans ignore: #include "poller_base.hpp"

namespace zmq
{
struct i_poll_events;

//  Implements socket polling mechanism using the POSIX.1-2001
//  poll() system call.

class poll_t : public worker_poller_base_t
{
  public:
    typedef fd_t handle_t;

    poll_t (const thread_ctx_t &ctx_);
    ~poll_t ();

    //  "poller" concept.
    //  These methods may only be called from an event callback; add_fd may also be called before start.
    handle_t add_fd (fd_t fd_, zmq::i_poll_events *events_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void stop ();

    static int max_fds ();

  private:
    //  Main event loop.
    virtual void loop ();

    void cleanup_retired ();

    struct fd_entry_t
    {
        fd_t index;
        zmq::i_poll_events *events;
    };

    //  This table stores data for registered descriptors.
    typedef std::vector<fd_entry_t> fd_table_t;
    fd_table_t fd_table;

    //  Pollset to pass to the poll function.
    typedef std::vector<pollfd> pollset_t;
    pollset_t pollset;

    //  If true, there's at least one retired event source.
    bool retired;

    poll_t (const poll_t &);
    const poll_t &operator= (const poll_t &);
};

typedef poll_t poller_t;
}

#endif

#endif


//========= end of #include "poll.hpp" ============


//========= begin of #include "select.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SELECT_HPP_INCLUDED__
#define __ZMQ_SELECT_HPP_INCLUDED__

//  poller.hpp decides which polling mechanism to use.
// ans ignore: #include "poller.hpp"
#if defined ZMQ_IOTHREAD_POLLER_USE_SELECT

#include <stddef.h>
#include <vector>
#include <map>

#if defined ZMQ_HAVE_WINDOWS
#elif defined ZMQ_HAVE_OPENVMS
#include <sys/types.h>
#include <sys/time.h>
#else
#include <sys/select.h>
#endif

// ans ignore: #include "ctx.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "poller_base.hpp"

namespace zmq
{
struct i_poll_events;

//  Implements socket polling mechanism using POSIX.1-2001 select()
//  function.

class select_t : public worker_poller_base_t
{
  public:
    typedef fd_t handle_t;

    select_t (const thread_ctx_t &ctx_);
    ~select_t ();

    //  "poller" concept.
    handle_t add_fd (fd_t fd_, zmq::i_poll_events *events_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void stop ();

    static int max_fds ();

  private:
    //  Main event loop.
    void loop ();

    //  Internal state.
    struct fds_set_t
    {
        fds_set_t ();
        fds_set_t (const fds_set_t &other_);
        fds_set_t &operator= (const fds_set_t &other_);
        //  Convenience method to descriptor from all sets.
        void remove_fd (const fd_t &fd_);

        fd_set read;
        fd_set write;
        fd_set error;
    };

    struct fd_entry_t
    {
        fd_t fd;
        zmq::i_poll_events *events;
    };
    typedef std::vector<fd_entry_t> fd_entries_t;

    void trigger_events (const fd_entries_t &fd_entries_,
                         const fds_set_t &local_fds_set_,
                         int event_count_);

    struct family_entry_t
    {
        family_entry_t ();

        fd_entries_t fd_entries;
        fds_set_t fds_set;
        bool has_retired;
    };

    void select_family_entry (family_entry_t &family_entry_,
                              int max_fd_,
                              bool use_timeout_,
                              struct timeval &tv_);

#if defined ZMQ_HAVE_WINDOWS
    typedef std::map<u_short, family_entry_t> family_entries_t;

    struct wsa_events_t
    {
        wsa_events_t ();
        ~wsa_events_t ();

        //  read, write, error and readwrite
        WSAEVENT events[4];
    };

    family_entries_t _family_entries;
    // See loop for details.
    family_entries_t::iterator _current_family_entry_it;

    int try_retire_fd_entry (family_entries_t::iterator family_entry_it_,
                             zmq::fd_t &handle_);

    static const size_t fd_family_cache_size = 8;
    std::pair<fd_t, u_short> _fd_family_cache[fd_family_cache_size];

    u_short get_fd_family (fd_t fd_);

    //  Socket's family or AF_UNSPEC on error.
    static u_short determine_fd_family (fd_t fd_);
#else
    //  on non-Windows, we can treat all fds as one family
    family_entry_t _family_entry;
    fd_t _max_fd;
#endif

    void cleanup_retired ();
    bool cleanup_retired (family_entry_t &family_entry_);

    //  Checks if an fd_entry_t is retired.
    static bool is_retired_fd (const fd_entry_t &entry_);

    static fd_entries_t::iterator
    find_fd_entry_by_handle (fd_entries_t &fd_entries_, handle_t handle_);

    select_t (const select_t &);
    const select_t &operator= (const select_t &);
};

typedef select_t poller_t;
}

#endif

#endif


//========= end of #include "select.hpp" ============


//========= begin of #include "poller.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_POLLER_HPP_INCLUDED__
#define __ZMQ_POLLER_HPP_INCLUDED__

#if defined ZMQ_IOTHREAD_POLLER_USE_KQUEUE                                     \
    + defined ZMQ_IOTHREAD_POLLER_USE_EPOLL                                    \
    + defined ZMQ_IOTHREAD_POLLER_USE_DEVPOLL                                  \
    + defined ZMQ_IOTHREAD_POLLER_USE_POLLSET                                  \
    + defined ZMQ_IOTHREAD_POLLER_POLL                                         \
    + defined ZMQ_IOTHREAD_POLLER_USE_SELECT                                   \
  > 1
#error More than one of the ZMQ_IOTHREAD_POLLER_USE_* macros defined
#endif

#if defined ZMQ_IOTHREAD_POLLER_USE_KQUEUE
// ans ignore: #include "kqueue.hpp"
#elif defined ZMQ_IOTHREAD_POLLER_USE_EPOLL
// ans ignore: #include "epoll.hpp"
#elif defined ZMQ_IOTHREAD_POLLER_USE_DEVPOLL
// ans ignore: #include "devpoll.hpp"
#elif defined ZMQ_IOTHREAD_POLLER_USE_POLLSET
// ans ignore: #include "pollset.hpp"
#elif defined ZMQ_IOTHREAD_POLLER_USE_POLL
// ans ignore: #include "poll.hpp"
#elif defined ZMQ_IOTHREAD_POLLER_USE_SELECT
// ans ignore: #include "select.hpp"
#elif defined ZMQ_HAVE_GNU
#define ZMQ_IOTHREAD_POLLER_USE_POLL
// ans ignore: #include "poll.hpp"
#else
#error None of the ZMQ_IOTHREAD_POLLER_USE_* macros defined
#endif

#if (defined ZMQ_POLL_BASED_ON_SELECT + defined ZMQ_POLL_BASED_ON_POLL) > 1
#error More than one of the ZMQ_POLL_BASED_ON_* macros defined
#elif (defined ZMQ_POLL_BASED_ON_SELECT + defined ZMQ_POLL_BASED_ON_POLL) == 0
#error None of the ZMQ_POLL_BASED_ON_* macros defined
#endif

#endif


//========= end of #include "poller.hpp" ============


//========= begin of #include "i_poll_events.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_I_POLL_EVENTS_HPP_INCLUDED__
#define __ZMQ_I_POLL_EVENTS_HPP_INCLUDED__

namespace zmq
{
// Virtual interface to be exposed by object that want to be notified
// about events on file descriptors.

struct i_poll_events
{
    virtual ~i_poll_events () {}

    // Called by I/O thread when file descriptor is ready for reading.
    virtual void in_event () = 0;

    // Called by I/O thread when file descriptor is ready for writing.
    virtual void out_event () = 0;

    // Called when timer expires.
    virtual void timer_event (int id_) = 0;
};
}

#endif


//========= end of #include "i_poll_events.hpp" ============


//========= begin of #include "io_object.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_IO_OBJECT_HPP_INCLUDED__
#define __ZMQ_IO_OBJECT_HPP_INCLUDED__

#include <stddef.h>

// ans ignore: #include "stdint.hpp"
// ans ignore: #include "poller.hpp"
// ans ignore: #include "i_poll_events.hpp"

namespace zmq
{
class io_thread_t;

//  Simple base class for objects that live in I/O threads.
//  It makes communication with the poller object easier and
//  makes defining unneeded event handlers unnecessary.

class io_object_t : public i_poll_events
{
  public:
    io_object_t (zmq::io_thread_t *io_thread_ = NULL);
    ~io_object_t ();

    //  When migrating an object from one I/O thread to another, first
    //  unplug it, then migrate it, then plug it to the new thread.
    void plug (zmq::io_thread_t *io_thread_);
    void unplug ();

  protected:
    typedef poller_t::handle_t handle_t;

    //  Methods to access underlying poller object.
    handle_t add_fd (fd_t fd_);
    void rm_fd (handle_t handle_);
    void set_pollin (handle_t handle_);
    void reset_pollin (handle_t handle_);
    void set_pollout (handle_t handle_);
    void reset_pollout (handle_t handle_);
    void add_timer (int timeout_, int id_);
    void cancel_timer (int id_);

    //  i_poll_events interface implementation.
    void in_event ();
    void out_event ();
    void timer_event (int id_);

  private:
    poller_t *_poller;

    io_object_t (const io_object_t &);
    const io_object_t &operator= (const io_object_t &);
};
}

#endif


//========= end of #include "io_object.hpp" ============


//========= begin of #include "pipe.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PIPE_HPP_INCLUDED__
#define __ZMQ_PIPE_HPP_INCLUDED__

// ans ignore: #include "ypipe_base.hpp"
// ans ignore: #include "config.hpp"
// ans ignore: #include "object.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "array.hpp"
// ans ignore: #include "blob.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "endpoint.hpp"

namespace zmq
{
class msg_t;
class pipe_t;

//  Create a pipepair for bi-directional transfer of messages.
//  First HWM is for messages passed from first pipe to the second pipe.
//  Second HWM is for messages passed from second pipe to the first pipe.
//  Delay specifies how the pipe behaves when the peer terminates. If true
//  pipe receives all the pending messages before terminating, otherwise it
//  terminates straight away.
//  If conflate is true, only the most recently arrived message could be
//  read (older messages are discarded)
int pipepair (zmq::object_t *parents_[2],
              zmq::pipe_t *pipes_[2],
              int hwms_[2],
              bool conflate_[2]);

struct i_pipe_events
{
    virtual ~i_pipe_events () {}

    virtual void read_activated (zmq::pipe_t *pipe_) = 0;
    virtual void write_activated (zmq::pipe_t *pipe_) = 0;
    virtual void hiccuped (zmq::pipe_t *pipe_) = 0;
    virtual void pipe_terminated (zmq::pipe_t *pipe_) = 0;
};

//  Note that pipe can be stored in three different arrays.
//  The array of inbound pipes (1), the array of outbound pipes (2) and
//  the generic array of pipes to be deallocated (3).

class pipe_t : public object_t,
               public array_item_t<1>,
               public array_item_t<2>,
               public array_item_t<3>
{
    //  This allows pipepair to create pipe objects.
    friend int pipepair (zmq::object_t *parents_[2],
                         zmq::pipe_t *pipes_[2],
                         int hwms_[2],
                         bool conflate_[2]);

  public:
    //  Specifies the object to send events to.
    void set_event_sink (i_pipe_events *sink_);

    //  Pipe endpoint can store an routing ID to be used by its clients.
    void set_server_socket_routing_id (uint32_t server_socket_routing_id_);
    uint32_t get_server_socket_routing_id () const;

    //  Pipe endpoint can store an opaque ID to be used by its clients.
    void set_router_socket_routing_id (const blob_t &router_socket_routing_id_);
    const blob_t &get_routing_id () const;

    //  Returns true if there is at least one message to read in the pipe.
    bool check_read ();

    //  Reads a message to the underlying pipe.
    bool read (msg_t *msg_);

    //  Checks whether messages can be written to the pipe. If the pipe is
    //  closed or if writing the message would cause high watermark the
    //  function returns false.
    bool check_write ();

    //  Writes a message to the underlying pipe. Returns false if the
    //  message does not pass check_write. If false, the message object
    //  retains ownership of its message buffer.
    bool write (msg_t *msg_);

    //  Remove unfinished parts of the outbound message from the pipe.
    void rollback () const;

    //  Flush the messages downstream.
    void flush ();

    //  Temporarily disconnects the inbound message stream and drops
    //  all the messages on the fly. Causes 'hiccuped' event to be generated
    //  in the peer.
    void hiccup ();

    //  Ensure the pipe won't block on receiving pipe_term.
    void set_nodelay ();

    //  Ask pipe to terminate. The termination will happen asynchronously
    //  and user will be notified about actual deallocation by 'terminated'
    //  event. If delay is true, the pending messages will be processed
    //  before actual shutdown.
    void terminate (bool delay_);

    //  Set the high water marks.
    void set_hwms (int inhwm_, int outhwm_);

    //  Set the boost to high water marks, used by inproc sockets so total hwm are sum of connect and bind sockets watermarks
    void set_hwms_boost (int inhwmboost_, int outhwmboost_);

    // send command to peer for notify the change of hwm
    void send_hwms_to_peer (int inhwm_, int outhwm_);

    //  Returns true if HWM is not reached
    bool check_hwm () const;

    void set_endpoint_pair (endpoint_uri_pair_t endpoint_pair_);
    const endpoint_uri_pair_t &get_endpoint_pair () const;

    void send_stats_to_peer (own_t *socket_base_);

  private:
    //  Type of the underlying lock-free pipe.
    typedef ypipe_base_t<msg_t> upipe_t;

    //  Command handlers.
    void process_activate_read ();
    void process_activate_write (uint64_t msgs_read_);
    void process_hiccup (void *pipe_);
    void process_pipe_peer_stats (uint64_t queue_count_,
                                  own_t *socket_base_,
                                  endpoint_uri_pair_t *endpoint_pair_);
    void process_pipe_term ();
    void process_pipe_term_ack ();
    void process_pipe_hwm (int inhwm_, int outhwm_);

    //  Handler for delimiter read from the pipe.
    void process_delimiter ();

    //  Constructor is private. Pipe can only be created using
    //  pipepair function.
    pipe_t (object_t *parent_,
            upipe_t *inpipe_,
            upipe_t *outpipe_,
            int inhwm_,
            int outhwm_,
            bool conflate_);

    //  Pipepair uses this function to let us know about
    //  the peer pipe object.
    void set_peer (pipe_t *peer_);

    //  Destructor is private. Pipe objects destroy themselves.
    ~pipe_t ();

    //  Underlying pipes for both directions.
    upipe_t *_in_pipe;
    upipe_t *_out_pipe;

    //  Can the pipe be read from / written to?
    bool _in_active;
    bool _out_active;

    //  High watermark for the outbound pipe.
    int _hwm;

    //  Low watermark for the inbound pipe.
    int _lwm;

    // boosts for high and low watermarks, used with inproc sockets so hwm are sum of send and recv hmws on each side of pipe
    int _in_hwm_boost;
    int _out_hwm_boost;

    //  Number of messages read and written so far.
    uint64_t _msgs_read;
    uint64_t _msgs_written;

    //  Last received peer's msgs_read. The actual number in the peer
    //  can be higher at the moment.
    uint64_t _peers_msgs_read;

    //  The pipe object on the other side of the pipepair.
    pipe_t *_peer;

    //  Sink to send events to.
    i_pipe_events *_sink;

    //  States of the pipe endpoint:
    //  active: common state before any termination begins,
    //  delimiter_received: delimiter was read from pipe before
    //      term command was received,
    //  waiting_for_delimiter: term command was already received
    //      from the peer but there are still pending messages to read,
    //  term_ack_sent: all pending messages were already read and
    //      all we are waiting for is ack from the peer,
    //  term_req_sent1: 'terminate' was explicitly called by the user,
    //  term_req_sent2: user called 'terminate' and then we've got
    //      term command from the peer as well.
    enum
    {
        active,
        delimiter_received,
        waiting_for_delimiter,
        term_ack_sent,
        term_req_sent1,
        term_req_sent2
    } _state;

    //  If true, we receive all the pending inbound messages before
    //  terminating. If false, we terminate immediately when the peer
    //  asks us to.
    bool _delay;

    //  Routing id of the writer. Used uniquely by the reader side.
    blob_t _router_socket_routing_id;

    //  Routing id of the writer. Used uniquely by the reader side.
    int _server_socket_routing_id;

    //  Returns true if the message is delimiter; false otherwise.
    static bool is_delimiter (const msg_t &msg_);

    //  Computes appropriate low watermark from the given high watermark.
    static int compute_lwm (int hwm_);

    const bool _conflate;

    // The endpoints of this pipe.
    endpoint_uri_pair_t _endpoint_pair;

    //  Disable copying.
    pipe_t (const pipe_t &);
    const pipe_t &operator= (const pipe_t &);
};

void send_routing_id (pipe_t *pipe_, const options_t &options_);
}

#endif


//========= end of #include "pipe.hpp" ============


//========= begin of #include "socket_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SOCKET_BASE_HPP_INCLUDED__
#define __ZMQ_SOCKET_BASE_HPP_INCLUDED__

#include <string>
#include <map>
#include <stdarg.h>

// ans ignore: #include "own.hpp"
// ans ignore: #include "array.hpp"
// ans ignore: #include "blob.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "poller.hpp"
// ans ignore: #include "i_poll_events.hpp"
// ans ignore: #include "i_mailbox.hpp"
// ans ignore: #include "clock.hpp"
// ans ignore: #include "pipe.hpp"
// ans ignore: #include "endpoint.hpp"

extern "C" {
void zmq_free_event (void *data_, void *hint_);
}

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;

class socket_base_t : public own_t,
                      public array_item_t<>,
                      public i_poll_events,
                      public i_pipe_events
{
    friend class reaper_t;

  public:
    //  Returns false if object is not a socket.
    bool check_tag () const;

    //  Returns whether the socket is thread-safe.
    bool is_thread_safe () const;

    //  Create a socket of a specified type.
    static socket_base_t *
    create (int type_, zmq::ctx_t *parent_, uint32_t tid_, int sid_);

    //  Returns the mailbox associated with this socket.
    i_mailbox *get_mailbox () const;

    //  Interrupt blocking call if the socket is stuck in one.
    //  This function can be called from a different thread!
    void stop ();

    //  Interface for communication with the API layer.
    int setsockopt (int option_, const void *optval_, size_t optvallen_);
    int getsockopt (int option_, void *optval_, size_t *optvallen_);
    int bind (const char *endpoint_uri_);
    int connect (const char *endpoint_uri_);
    int term_endpoint (const char *endpoint_uri_);
    int send (zmq::msg_t *msg_, int flags_);
    int recv (zmq::msg_t *msg_, int flags_);
    void add_signaler (signaler_t *s_);
    void remove_signaler (signaler_t *s_);
    int close ();

    //  These functions are used by the polling mechanism to determine
    //  which events are to be reported from this socket.
    bool has_in ();
    bool has_out ();

    //  Joining and leaving groups
    int join (const char *group_);
    int leave (const char *group_);

    //  Using this function reaper thread ask the socket to register with
    //  its poller.
    void start_reaping (poller_t *poller_);

    //  i_poll_events implementation. This interface is used when socket
    //  is handled by the poller in the reaper thread.
    void in_event ();
    void out_event ();
    void timer_event (int id_);

    //  i_pipe_events interface implementation.
    void read_activated (pipe_t *pipe_);
    void write_activated (pipe_t *pipe_);
    void hiccuped (pipe_t *pipe_);
    void pipe_terminated (pipe_t *pipe_);
    void lock ();
    void unlock ();

    int monitor (const char *endpoint_,
                 uint64_t events_,
                 int event_version_,
                 int type_);

    void event_connected (const endpoint_uri_pair_t &endpoint_uri_pair_,
                          zmq::fd_t fd_);
    void event_connect_delayed (const endpoint_uri_pair_t &endpoint_uri_pair_,
                                int err_);
    void event_connect_retried (const endpoint_uri_pair_t &endpoint_uri_pair_,
                                int interval_);
    void event_listening (const endpoint_uri_pair_t &endpoint_uri_pair_,
                          zmq::fd_t fd_);
    void event_bind_failed (const endpoint_uri_pair_t &endpoint_uri_pair_,
                            int err_);
    void event_accepted (const endpoint_uri_pair_t &endpoint_uri_pair_,
                         zmq::fd_t fd_);
    void event_accept_failed (const endpoint_uri_pair_t &endpoint_uri_pair_,
                              int err_);
    void event_closed (const endpoint_uri_pair_t &endpoint_uri_pair_,
                       zmq::fd_t fd_);
    void event_close_failed (const endpoint_uri_pair_t &endpoint_uri_pair_,
                             int err_);
    void event_disconnected (const endpoint_uri_pair_t &endpoint_uri_pair_,
                             zmq::fd_t fd_);
    void event_handshake_failed_no_detail (
      const endpoint_uri_pair_t &endpoint_uri_pair_, int err_);
    void event_handshake_failed_protocol (
      const endpoint_uri_pair_t &endpoint_uri_pair_, int err_);
    void
    event_handshake_failed_auth (const endpoint_uri_pair_t &endpoint_uri_pair_,
                                 int err_);
    void
    event_handshake_succeeded (const endpoint_uri_pair_t &endpoint_uri_pair_,
                               int err_);

    //  Query the state of a specific peer. The default implementation
    //  always returns an ENOTSUP error.
    virtual int get_peer_state (const void *routing_id_,
                                size_t routing_id_size_) const;

    //  Request for pipes statistics - will generate a ZMQ_EVENT_PIPES_STATS
    //  after gathering the data asynchronously. Requires event monitoring to
    //  be enabled.
    int query_pipes_stats ();

  protected:
    socket_base_t (zmq::ctx_t *parent_,
                   uint32_t tid_,
                   int sid_,
                   bool thread_safe_ = false);
    virtual ~socket_base_t ();

    //  Concrete algorithms for the x- methods are to be defined by
    //  individual socket types.
    virtual void xattach_pipe (zmq::pipe_t *pipe_,
                               bool subscribe_to_all_ = false,
                               bool locally_initiated_ = false) = 0;

    //  The default implementation assumes there are no specific socket
    //  options for the particular socket type. If not so, override this
    //  method.
    virtual int
    xsetsockopt (int option_, const void *optval_, size_t optvallen_);

    //  The default implementation assumes that send is not supported.
    virtual bool xhas_out ();
    virtual int xsend (zmq::msg_t *msg_);

    //  The default implementation assumes that recv in not supported.
    virtual bool xhas_in ();
    virtual int xrecv (zmq::msg_t *msg_);

    //  i_pipe_events will be forwarded to these functions.
    virtual void xread_activated (pipe_t *pipe_);
    virtual void xwrite_activated (pipe_t *pipe_);
    virtual void xhiccuped (pipe_t *pipe_);
    virtual void xpipe_terminated (pipe_t *pipe_) = 0;

    //  the default implementation assumes that joub and leave are not supported.
    virtual int xjoin (const char *group_);
    virtual int xleave (const char *group_);

    //  Delay actual destruction of the socket.
    void process_destroy ();

  private:
    // test if event should be sent and then dispatch it
    void event (const endpoint_uri_pair_t &endpoint_uri_pair_,
                uint64_t values_[],
                uint64_t values_count_,
                uint64_t type_);

    // Socket event data dispatch
    void monitor_event (uint64_t event_,
                        uint64_t values_[],
                        uint64_t values_count_,
                        const endpoint_uri_pair_t &endpoint_uri_pair_) const;

    // Monitor socket cleanup
    void stop_monitor (bool send_monitor_stopped_event_ = true);

    //  Creates new endpoint ID and adds the endpoint to the map.
    void add_endpoint (const endpoint_uri_pair_t &endpoint_pair_,
                       own_t *endpoint_,
                       pipe_t *pipe_);

    //  Map of open endpoints.
    typedef std::pair<own_t *, pipe_t *> endpoint_pipe_t;
    typedef std::multimap<std::string, endpoint_pipe_t> endpoints_t;
    endpoints_t _endpoints;

    //  Map of open inproc endpoints.
    class inprocs_t
    {
      public:
        void emplace (const char *endpoint_uri_, pipe_t *pipe_);
        int erase_pipes (const std::string &endpoint_uri_str_);
        void erase_pipe (pipe_t *pipe_);

      private:
        typedef std::multimap<std::string, pipe_t *> map_t;
        map_t _inprocs;
    };
    inprocs_t _inprocs;

    //  To be called after processing commands or invoking any command
    //  handlers explicitly. If required, it will deallocate the socket.
    void check_destroy ();

    //  Moves the flags from the message to local variables,
    //  to be later retrieved by getsockopt.
    void extract_flags (msg_t *msg_);

    //  Used to check whether the object is a socket.
    uint32_t _tag;

    //  If true, associated context was already terminated.
    bool _ctx_terminated;

    //  If true, object should have been already destroyed. However,
    //  destruction is delayed while we unwind the stack to the point
    //  where it doesn't intersect the object being destroyed.
    bool _destroyed;

    //  Parse URI string.
    static int
    parse_uri (const char *uri_, std::string &protocol_, std::string &path_);

    //  Check whether transport protocol, as specified in connect or
    //  bind, is available and compatible with the socket type.
    int check_protocol (const std::string &protocol_) const;

    //  Register the pipe with this socket.
    void attach_pipe (zmq::pipe_t *pipe_,
                      bool subscribe_to_all_ = false,
                      bool locally_initiated_ = false);

    //  Processes commands sent to this socket (if any). If timeout is -1,
    //  returns only after at least one command was processed.
    //  If throttle argument is true, commands are processed at most once
    //  in a predefined time period.
    int process_commands (int timeout_, bool throttle_);

    //  Handlers for incoming commands.
    void process_stop ();
    void process_bind (zmq::pipe_t *pipe_);
    void process_pipe_stats_publish (uint64_t outbound_queue_count_,
                                     uint64_t inbound_queue_count_,
                                     endpoint_uri_pair_t *endpoint_pair_);
    void process_term (int linger_);
    void process_term_endpoint (std::string *endpoint_);

    void update_pipe_options (int option_);

    std::string resolve_tcp_addr (std::string endpoint_uri_,
                                  const char *tcp_address_);

    //  Socket's mailbox object.
    i_mailbox *_mailbox;

    //  List of attached pipes.
    typedef array_t<pipe_t, 3> pipes_t;
    pipes_t _pipes;

    //  Reaper's poller and handle of this socket within it.
    poller_t *_poller;
    poller_t::handle_t _handle;

    //  Timestamp of when commands were processed the last time.
    uint64_t _last_tsc;

    //  Number of messages received since last command processing.
    int _ticks;

    //  True if the last message received had MORE flag set.
    bool _rcvmore;

    //  Improves efficiency of time measurement.
    clock_t _clock;

    // Monitor socket;
    void *_monitor_socket;

    // Bitmask of events being monitored
    int64_t _monitor_events;

    // Last socket endpoint resolved URI
    std::string _last_endpoint;

    // Indicate if the socket is thread safe
    const bool _thread_safe;

    // Signaler to be used in the reaping stage
    signaler_t *_reaper_signaler;

    // Mutex for synchronize access to the socket in thread safe mode
    mutex_t _sync;

    // Mutex to synchronize access to the monitor Pair socket
    mutex_t _monitor_sync;

    socket_base_t (const socket_base_t &);
    const socket_base_t &operator= (const socket_base_t &);
};

class routing_socket_base_t : public socket_base_t
{
  protected:
    routing_socket_base_t (class ctx_t *parent_, uint32_t tid_, int sid_);
    ~routing_socket_base_t ();

    // methods from socket_base_t
    virtual int
    xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    virtual void xwrite_activated (pipe_t *pipe_);

    // own methods
    std::string extract_connect_routing_id ();
    bool connect_routing_id_is_set () const;

    struct out_pipe_t
    {
        pipe_t *pipe;
        bool active;
    };

    void add_out_pipe (blob_t routing_id_, pipe_t *pipe_);
    bool has_out_pipe (const blob_t &routing_id_) const;
    out_pipe_t *lookup_out_pipe (const blob_t &routing_id_);
    const out_pipe_t *lookup_out_pipe (const blob_t &routing_id_) const;
    void erase_out_pipe (pipe_t *pipe_);
    out_pipe_t try_erase_out_pipe (const blob_t &routing_id_);
    template <typename Func> bool any_of_out_pipes (Func func_)
    {
        bool res = false;
        for (out_pipes_t::iterator it = _out_pipes.begin ();
             it != _out_pipes.end () && !res; ++it) {
            res |= func_ (*it->second.pipe);
        }

        return res;
    }

  private:
    //  Outbound pipes indexed by the peer IDs.
    typedef std::map<blob_t, out_pipe_t> out_pipes_t;
    out_pipes_t _out_pipes;

    // Next assigned name on a zmq_connect() call used by ROUTER and STREAM socket types
    std::string _connect_routing_id;
};
}

#endif


//========= end of #include "socket_base.hpp" ============


//========= begin of #include "i_engine.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_I_ENGINE_HPP_INCLUDED__
#define __ZMQ_I_ENGINE_HPP_INCLUDED__

// ans ignore: #include "endpoint.hpp"

namespace zmq
{
class io_thread_t;

//  Abstract interface to be implemented by various engines.

struct i_engine
{
    virtual ~i_engine () {}

    //  Plug the engine to the session.
    virtual void plug (zmq::io_thread_t *io_thread_,
                       class session_base_t *session_) = 0;

    //  Terminate and deallocate the engine. Note that 'detached'
    //  events are not fired on termination.
    virtual void terminate () = 0;

    //  This method is called by the session to signalise that more
    //  messages can be written to the pipe.
    //  Returns false if the engine was deleted due to an error.
    //  TODO it is probably better to change the design such that the engine
    //  does not delete itself
    virtual bool restart_input () = 0;

    //  This method is called by the session to signalise that there
    //  are messages to send available.
    virtual void restart_output () = 0;

    virtual void zap_msg_available () = 0;

    virtual const endpoint_uri_pair_t &get_endpoint () const = 0;
};
}

#endif


//========= end of #include "i_engine.hpp" ============


//========= begin of #include "i_encoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_I_ENCODER_HPP_INCLUDED__
#define __ZMQ_I_ENCODER_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"

namespace zmq
{
//  Forward declaration
class msg_t;

//  Interface to be implemented by message encoder.

struct i_encoder
{
    virtual ~i_encoder () {}

    //  The function returns a batch of binary data. The data
    //  are filled to a supplied buffer. If no buffer is supplied (data_
    //  is NULL) encoder will provide buffer of its own.
    //  Function returns 0 when a new message is required.
    virtual size_t encode (unsigned char **data_, size_t size_) = 0;

    //  Load a new message into encoder.
    virtual void load_msg (msg_t *msg_) = 0;
};
}

#endif


//========= end of #include "i_encoder.hpp" ============


//========= begin of #include "i_decoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_I_DECODER_HPP_INCLUDED__
#define __ZMQ_I_DECODER_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"

namespace zmq
{
class msg_t;

//  Interface to be implemented by message decoder.

class i_decoder
{
  public:
    virtual ~i_decoder () {}

    virtual void get_buffer (unsigned char **data_, size_t *size_) = 0;

    virtual void resize_buffer (size_t) = 0;
    //  Decodes data pointed to by data_.
    //  When a message is decoded, 1 is returned.
    //  When the decoder needs more data, 0 is returned.
    //  On error, -1 is returned and errno is set accordingly.
    virtual int
    decode (const unsigned char *data_, size_t size_, size_t &processed_) = 0;

    virtual msg_t *msg () = 0;
};
}

#endif


//========= end of #include "i_decoder.hpp" ============


//========= begin of #include "stream_engine.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_STREAM_ENGINE_HPP_INCLUDED__
#define __ZMQ_STREAM_ENGINE_HPP_INCLUDED__

#include <stddef.h>

// ans ignore: #include "fd.hpp"
// ans ignore: #include "i_engine.hpp"
// ans ignore: #include "io_object.hpp"
// ans ignore: #include "i_encoder.hpp"
// ans ignore: #include "i_decoder.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "metadata.hpp"
// ans ignore: #include "msg.hpp"

namespace zmq
{
//  Protocol revisions
enum
{
    ZMTP_1_0 = 0,
    ZMTP_2_0 = 1
};

class io_thread_t;
class session_base_t;
class mechanism_t;

//  This engine handles any socket with SOCK_STREAM semantics,
//  e.g. TCP socket or an UNIX domain socket.

class stream_engine_t : public io_object_t, public i_engine
{
  public:
    enum error_reason_t
    {
        protocol_error,
        connection_error,
        timeout_error
    };

    stream_engine_t (fd_t fd_,
                     const options_t &options_,
                     const endpoint_uri_pair_t &endpoint_uri_pair_);
    ~stream_engine_t ();

    //  i_engine interface implementation.
    void plug (zmq::io_thread_t *io_thread_, zmq::session_base_t *session_);
    void terminate ();
    bool restart_input ();
    void restart_output ();
    void zap_msg_available ();
    const endpoint_uri_pair_t &get_endpoint () const;

    //  i_poll_events interface implementation.
    void in_event ();
    void out_event ();
    void timer_event (int id_);

  private:
    bool in_event_internal ();

    //  Unplug the engine from the session.
    void unplug ();

    //  Function to handle network disconnections.
    void error (error_reason_t reason_);

    //  Detects the protocol used by the peer.
    bool handshake ();

    //  Receive the greeting from the peer.
    int receive_greeting ();
    void receive_greeting_versioned ();

    typedef bool (stream_engine_t::*handshake_fun_t) ();
    static handshake_fun_t select_handshake_fun (bool unversioned,
                                                 unsigned char revision);

    bool handshake_v1_0_unversioned ();
    bool handshake_v1_0 ();
    bool handshake_v2_0 ();
    bool handshake_v3_0 ();

    int routing_id_msg (msg_t *msg_);
    int process_routing_id_msg (msg_t *msg_);

    int next_handshake_command (msg_t *msg_);
    int process_handshake_command (msg_t *msg_);

    int pull_msg_from_session (msg_t *msg_);
    int push_msg_to_session (msg_t *msg_);

    int push_raw_msg_to_session (msg_t *msg_);

    int write_credential (msg_t *msg_);
    int pull_and_encode (msg_t *msg_);
    int decode_and_push (msg_t *msg_);
    int push_one_then_decode_and_push (msg_t *msg_);

    void mechanism_ready ();

    size_t add_property (unsigned char *ptr_,
                         const char *name_,
                         const void *value_,
                         size_t value_len_);

    void set_handshake_timer ();

    typedef metadata_t::dict_t properties_t;
    bool init_properties (properties_t &properties_);

    int process_command_message (msg_t *msg_);
    int produce_ping_message (msg_t *msg_);
    int process_heartbeat_message (msg_t *msg_);
    int produce_pong_message (msg_t *msg_);

    //  Underlying socket.
    fd_t _s;

    msg_t _tx_msg;
    //  Need to store PING payload for PONG
    msg_t _pong_msg;

    handle_t _handle;

    unsigned char *_inpos;
    size_t _insize;
    i_decoder *_decoder;

    unsigned char *_outpos;
    size_t _outsize;
    i_encoder *_encoder;

    //  Metadata to be attached to received messages. May be NULL.
    metadata_t *_metadata;

    //  When true, we are still trying to determine whether
    //  the peer is using versioned protocol, and if so, which
    //  version.  When false, normal message flow has started.
    bool _handshaking;

    static const size_t signature_size = 10;

    //  Size of ZMTP/1.0 and ZMTP/2.0 greeting message
    static const size_t v2_greeting_size = 12;

    //  Size of ZMTP/3.0 greeting message
    static const size_t v3_greeting_size = 64;

    //  Expected greeting size.
    size_t _greeting_size;

    //  Greeting received from, and sent to peer
    unsigned char _greeting_recv[v3_greeting_size];
    unsigned char _greeting_send[v3_greeting_size];

    //  Size of greeting received so far
    unsigned int _greeting_bytes_read;

    //  The session this engine is attached to.
    zmq::session_base_t *_session;

    const options_t _options;

    //  Representation of the connected endpoints.
    const endpoint_uri_pair_t _endpoint_uri_pair;

    bool _plugged;

    int (stream_engine_t::*_next_msg) (msg_t *msg_);

    int (stream_engine_t::*_process_msg) (msg_t *msg_);

    bool _io_error;

    //  Indicates whether the engine is to inject a phantom
    //  subscription message into the incoming stream.
    //  Needed to support old peers.
    bool _subscription_required;

    mechanism_t *_mechanism;

    //  True iff the engine couldn't consume the last decoded message.
    bool _input_stopped;

    //  True iff the engine doesn't have any message to encode.
    bool _output_stopped;

    //  ID of the handshake timer
    enum
    {
        handshake_timer_id = 0x40
    };

    //  True is linger timer is running.
    bool _has_handshake_timer;

    //  Heartbeat stuff
    enum
    {
        heartbeat_ivl_timer_id = 0x80,
        heartbeat_timeout_timer_id = 0x81,
        heartbeat_ttl_timer_id = 0x82
    };
    bool _has_ttl_timer;
    bool _has_timeout_timer;
    bool _has_heartbeat_timer;
    int _heartbeat_timeout;

    // Socket
    zmq::socket_base_t *_socket;

    const std::string _peer_address;

    stream_engine_t (const stream_engine_t &);
    const stream_engine_t &operator= (const stream_engine_t &);
};
}

#endif


//========= end of #include "stream_engine.hpp" ============


//========= begin of #include "session_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SESSION_BASE_HPP_INCLUDED__
#define __ZMQ_SESSION_BASE_HPP_INCLUDED__

#include <stdarg.h>

// ans ignore: #include "own.hpp"
// ans ignore: #include "io_object.hpp"
// ans ignore: #include "pipe.hpp"
// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "stream_engine.hpp"

namespace zmq
{
class io_thread_t;
struct i_engine;
struct address_t;

class session_base_t : public own_t, public io_object_t, public i_pipe_events
{
  public:
    //  Create a session of the particular type.
    static session_base_t *create (zmq::io_thread_t *io_thread_,
                                   bool active_,
                                   zmq::socket_base_t *socket_,
                                   const options_t &options_,
                                   address_t *addr_);

    //  To be used once only, when creating the session.
    void attach_pipe (zmq::pipe_t *pipe_);

    //  Following functions are the interface exposed towards the engine.
    virtual void reset ();
    void flush ();
    void rollback ();
    void engine_error (zmq::stream_engine_t::error_reason_t reason_);

    //  i_pipe_events interface implementation.
    void read_activated (zmq::pipe_t *pipe_);
    void write_activated (zmq::pipe_t *pipe_);
    void hiccuped (zmq::pipe_t *pipe_);
    void pipe_terminated (zmq::pipe_t *pipe_);

    //  Delivers a message. Returns 0 if successful; -1 otherwise.
    //  The function takes ownership of the message.
    virtual int push_msg (msg_t *msg_);

    int zap_connect ();
    bool zap_enabled ();

    //  Fetches a message. Returns 0 if successful; -1 otherwise.
    //  The caller is responsible for freeing the message when no
    //  longer used.
    virtual int pull_msg (msg_t *msg_);

    //  Receives message from ZAP socket.
    //  Returns 0 on success; -1 otherwise.
    //  The caller is responsible for freeing the message.
    int read_zap_msg (msg_t *msg_);

    //  Sends message to ZAP socket.
    //  Returns 0 on success; -1 otherwise.
    //  The function takes ownership of the message.
    int write_zap_msg (msg_t *msg_);

    socket_base_t *get_socket ();
    const endpoint_uri_pair_t &get_endpoint () const;

  protected:
    session_base_t (zmq::io_thread_t *io_thread_,
                    bool active_,
                    zmq::socket_base_t *socket_,
                    const options_t &options_,
                    address_t *addr_);
    virtual ~session_base_t ();

  private:
    void start_connecting (bool wait_);

    typedef own_t *(session_base_t::*connecter_factory_fun_t) (
      io_thread_t *io_thread, bool wait_);
    typedef std::pair<const std::string, connecter_factory_fun_t>
      connecter_factory_entry_t;
    static connecter_factory_entry_t _connecter_factories[];
    typedef std::map<std::string, connecter_factory_fun_t>
      connecter_factory_map_t;
    static connecter_factory_map_t _connecter_factories_map;

    own_t *create_connecter_vmci (io_thread_t *io_thread_, bool wait_);
    own_t *create_connecter_tipc (io_thread_t *io_thread_, bool wait_);
    own_t *create_connecter_ipc (io_thread_t *io_thread_, bool wait_);
    own_t *create_connecter_tcp (io_thread_t *io_thread_, bool wait_);

    typedef void (session_base_t::*start_connecting_fun_t) (
      io_thread_t *io_thread);
    typedef std::pair<const std::string, start_connecting_fun_t>
      start_connecting_entry_t;
    static start_connecting_entry_t _start_connecting_entries[];
    typedef std::map<std::string, start_connecting_fun_t>
      start_connecting_map_t;
    static start_connecting_map_t _start_connecting_map;

    void start_connecting_pgm (io_thread_t *io_thread_);
    void start_connecting_norm (io_thread_t *io_thread_);
    void start_connecting_udp (io_thread_t *io_thread_);

    void reconnect ();

    //  Handlers for incoming commands.
    void process_plug ();
    void process_attach (zmq::i_engine *engine_);
    void process_term (int linger_);

    //  i_poll_events handlers.
    void timer_event (int id_);

    //  Remove any half processed messages. Flush unflushed messages.
    //  Call this function when engine disconnect to get rid of leftovers.
    void clean_pipes ();

    //  If true, this session (re)connects to the peer. Otherwise, it's
    //  a transient session created by the listener.
    const bool _active;

    //  Pipe connecting the session to its socket.
    zmq::pipe_t *_pipe;

    //  Pipe used to exchange messages with ZAP socket.
    zmq::pipe_t *_zap_pipe;

    //  This set is added to with pipes we are disconnecting, but haven't yet completed
    std::set<pipe_t *> _terminating_pipes;

    //  This flag is true if the remainder of the message being processed
    //  is still in the in pipe.
    bool _incomplete_in;

    //  True if termination have been suspended to push the pending
    //  messages to the network.
    bool _pending;

    //  The protocol I/O engine connected to the session.
    zmq::i_engine *_engine;

    //  The socket the session belongs to.
    zmq::socket_base_t *_socket;

    //  I/O thread the session is living in. It will be used to plug in
    //  the engines into the same thread.
    zmq::io_thread_t *_io_thread;

    //  ID of the linger timer
    enum
    {
        linger_timer_id = 0x20
    };

    //  True is linger timer is running.
    bool _has_linger_timer;

    //  Protocol and address to use when connecting.
    address_t *_addr;

    session_base_t (const session_base_t &);
    const session_base_t &operator= (const session_base_t &);
};
}

#endif


//========= end of #include "session_base.hpp" ============


//========= begin of #include "mechanism_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MECHANISM_BASE_HPP_INCLUDED__
#define __ZMQ_MECHANISM_BASE_HPP_INCLUDED__

// ans ignore: #include "mechanism.hpp"

namespace zmq
{
class msg_t;

class mechanism_base_t : public mechanism_t
{
  protected:
    mechanism_base_t (session_base_t *const session_,
                      const options_t &options_);

    session_base_t *const session;

    int check_basic_command_structure (msg_t *msg_);

    void handle_error_reason (const char *error_reason_,
                              size_t error_reason_len_);

    bool zap_required () const;
};
}

#endif


//========= end of #include "mechanism_base.hpp" ============


//========= begin of #include "gssapi_mechanism_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_GSSAPI_MECHANISM_BASE_HPP_INCLUDED__
#define __ZMQ_GSSAPI_MECHANISM_BASE_HPP_INCLUDED__

#ifdef HAVE_LIBGSSAPI_KRB5

#if HAVE_GSSAPI_GSSAPI_GENERIC_H
#include <gssapi/gssapi_generic.h>
#endif
#include <gssapi/gssapi_krb5.h>

// ans ignore: #include "mechanism_base.hpp"
// ans ignore: #include "options.hpp"

namespace zmq
{
class msg_t;

/// Commonalities between clients and servers are captured here.
/// For example, clients and servers both need to produce and
/// process context-level GSSAPI tokens (via INITIATE commands)
/// and per-message GSSAPI tokens (via MESSAGE commands).
class gssapi_mechanism_base_t : public virtual mechanism_base_t
{
  public:
    gssapi_mechanism_base_t (session_base_t *session_,
                             const options_t &options_);
    virtual ~gssapi_mechanism_base_t () = 0;

  protected:
    //  Produce a context-level GSSAPI token (INITIATE command)
    //  during security context initialization.
    int produce_initiate (msg_t *msg_, void *data_, size_t data_len_);

    //  Process a context-level GSSAPI token (INITIATE command)
    //  during security context initialization.
    int process_initiate (msg_t *msg_, void **data_, size_t &data_len_);

    // Produce a metadata ready msg (READY) to conclude handshake
    int produce_ready (msg_t *msg_);

    // Process a metadata ready msg (READY)
    int process_ready (msg_t *msg_);

    //  Encode a per-message GSSAPI token (MESSAGE command) using
    //  the established security context.
    int encode_message (msg_t *msg_);

    //  Decode a per-message GSSAPI token (MESSAGE command) using
    //  the  established security context.
    int decode_message (msg_t *msg_);

    //  Convert ZMQ_GSSAPI_NT values to GSSAPI name_type
    static const gss_OID convert_nametype (int zmq_name_type_);

    //  Acquire security context credentials from the
    //  underlying mechanism.
    static int acquire_credentials (char *principal_name_,
                                    gss_cred_id_t *cred_,
                                    gss_OID name_type_);

  protected:
    //  Opaque GSSAPI token for outgoing data
    gss_buffer_desc send_tok;

    //  Opaque GSSAPI token for incoming data
    gss_buffer_desc recv_tok;

    //  Opaque GSSAPI representation of principal
    gss_name_t target_name;

    //  Human-readable principal name
    char *principal_name;

    //  Status code returned by GSSAPI functions
    OM_uint32 maj_stat;

    //  Status code returned by the underlying mechanism
    OM_uint32 min_stat;

    //  Status code returned by the underlying mechanism
    //  during context initialization
    OM_uint32 init_sec_min_stat;

    //  Flags returned by GSSAPI (ignored)
    OM_uint32 ret_flags;

    //  Flags returned by GSSAPI (ignored)
    OM_uint32 gss_flags;

    //  Credentials used to establish security context
    gss_cred_id_t cred;

    //  Opaque GSSAPI representation of the security context
    gss_ctx_id_t context;

    //  If true, use gss to encrypt messages. If false, only utilize gss for auth.
    bool do_encryption;
};
}

#endif

#endif


//========= end of #include "gssapi_mechanism_base.hpp" ============


//========= begin of #include "zap_client.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ZAP_CLIENT_HPP_INCLUDED__
#define __ZMQ_ZAP_CLIENT_HPP_INCLUDED__

// ans ignore: #include "mechanism_base.hpp"

namespace zmq
{
class zap_client_t : public virtual mechanism_base_t
{
  public:
    zap_client_t (session_base_t *const session_,
                  const std::string &peer_address_,
                  const options_t &options_);

    void send_zap_request (const char *mechanism_,
                           size_t mechanism_length_,
                           const uint8_t *credentials_,
                           size_t credentials_size_);

    void send_zap_request (const char *mechanism_,
                           size_t mechanism_length_,
                           const uint8_t **credentials_,
                           size_t *credentials_sizes_,
                           size_t credentials_count_);

    virtual int receive_and_process_zap_reply ();
    virtual void handle_zap_status_code ();

  protected:
    const std::string peer_address;

    //  Status code as received from ZAP handler
    std::string status_code;
};

class zap_client_common_handshake_t : public zap_client_t
{
  protected:
    enum state_t
    {
        waiting_for_hello,
        sending_welcome,
        waiting_for_initiate,
        waiting_for_zap_reply,
        sending_ready,
        sending_error,
        error_sent,
        ready
    };

    zap_client_common_handshake_t (session_base_t *const session_,
                                   const std::string &peer_address_,
                                   const options_t &options_,
                                   state_t zap_reply_ok_state_);

    //  methods from mechanism_t
    status_t status () const;
    int zap_msg_available ();

    //  zap_client_t methods
    int receive_and_process_zap_reply ();
    void handle_zap_status_code ();

    //  Current FSM state
    state_t state;

  private:
    const state_t _zap_reply_ok_state;
};
}

#endif


//========= end of #include "zap_client.hpp" ============


//========= begin of #include "gssapi_server.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_GSSAPI_SERVER_HPP_INCLUDED__
#define __ZMQ_GSSAPI_SERVER_HPP_INCLUDED__

#ifdef HAVE_LIBGSSAPI_KRB5

// ans ignore: #include "gssapi_mechanism_base.hpp"
// ans ignore: #include "zap_client.hpp"

namespace zmq
{
class msg_t;
class session_base_t;

class gssapi_server_t : public gssapi_mechanism_base_t, public zap_client_t
{
  public:
    gssapi_server_t (session_base_t *session_,
                     const std::string &peer_address,
                     const options_t &options_);
    virtual ~gssapi_server_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);
    virtual int encode (msg_t *msg_);
    virtual int decode (msg_t *msg_);
    virtual int zap_msg_available ();
    virtual status_t status () const;

  private:
    enum state_t
    {
        send_next_token,
        recv_next_token,
        expect_zap_reply,
        send_ready,
        recv_ready,
        connected
    };

    session_base_t *const session;

    const std::string peer_address;

    //  Current FSM state
    state_t state;

    //  True iff server considers the client authenticated
    bool security_context_established;

    //  The underlying mechanism type (ignored)
    gss_OID doid;

    void accept_context ();
    int produce_next_token (msg_t *msg_);
    int process_next_token (msg_t *msg_);
    void send_zap_request ();
};
}

#endif

#endif


//========= end of #include "gssapi_server.hpp" ============


//========= begin of #include "wire.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_WIRE_HPP_INCLUDED__
#define __ZMQ_WIRE_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"

namespace zmq
{
//  Helper functions to convert different integer types to/from network
//  byte order.

inline void put_uint8 (unsigned char *buffer_, uint8_t value_)
{
    *buffer_ = value_;
}

inline uint8_t get_uint8 (const unsigned char *buffer_)
{
    return *buffer_;
}

inline void put_uint16 (unsigned char *buffer_, uint16_t value_)
{
    buffer_[0] = static_cast<unsigned char> (((value_) >> 8) & 0xff);
    buffer_[1] = static_cast<unsigned char> (value_ & 0xff);
}

inline uint16_t get_uint16 (const unsigned char *buffer_)
{
    return ((static_cast<uint16_t> (buffer_[0])) << 8)
           | (static_cast<uint16_t> (buffer_[1]));
}

inline void put_uint32 (unsigned char *buffer_, uint32_t value_)
{
    buffer_[0] = static_cast<unsigned char> (((value_) >> 24) & 0xff);
    buffer_[1] = static_cast<unsigned char> (((value_) >> 16) & 0xff);
    buffer_[2] = static_cast<unsigned char> (((value_) >> 8) & 0xff);
    buffer_[3] = static_cast<unsigned char> (value_ & 0xff);
}

inline uint32_t get_uint32 (const unsigned char *buffer_)
{
    return ((static_cast<uint32_t> (buffer_[0])) << 24)
           | ((static_cast<uint32_t> (buffer_[1])) << 16)
           | ((static_cast<uint32_t> (buffer_[2])) << 8)
           | (static_cast<uint32_t> (buffer_[3]));
}

inline void put_uint64 (unsigned char *buffer_, uint64_t value_)
{
    buffer_[0] = static_cast<unsigned char> (((value_) >> 56) & 0xff);
    buffer_[1] = static_cast<unsigned char> (((value_) >> 48) & 0xff);
    buffer_[2] = static_cast<unsigned char> (((value_) >> 40) & 0xff);
    buffer_[3] = static_cast<unsigned char> (((value_) >> 32) & 0xff);
    buffer_[4] = static_cast<unsigned char> (((value_) >> 24) & 0xff);
    buffer_[5] = static_cast<unsigned char> (((value_) >> 16) & 0xff);
    buffer_[6] = static_cast<unsigned char> (((value_) >> 8) & 0xff);
    buffer_[7] = static_cast<unsigned char> (value_ & 0xff);
}

inline uint64_t get_uint64 (const unsigned char *buffer_)
{
    return ((static_cast<uint64_t> (buffer_[0])) << 56)
           | ((static_cast<uint64_t> (buffer_[1])) << 48)
           | ((static_cast<uint64_t> (buffer_[2])) << 40)
           | ((static_cast<uint64_t> (buffer_[3])) << 32)
           | ((static_cast<uint64_t> (buffer_[4])) << 24)
           | ((static_cast<uint64_t> (buffer_[5])) << 16)
           | ((static_cast<uint64_t> (buffer_[6])) << 8)
           | (static_cast<uint64_t> (buffer_[7]));
}
}

#endif


//========= end of #include "wire.hpp" ============


//========= begin of #include "precompiled.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PRECOMPILED_HPP_INCLUDED__
#define __ZMQ_PRECOMPILED_HPP_INCLUDED__

//  On AIX platform, poll.h has to be included first to get consistent
//  definition of pollfd structure (AIX uses 'reqevents' and 'retnevents'
//  instead of 'events' and 'revents' and defines macros to map from POSIX-y
//  names to AIX-specific names).
//  zmq.h must be included *after* poll.h for AIX to build properly.
//  precompiled.hpp includes include/zmq.h
#if defined ZMQ_POLL_BASED_ON_POLL && defined ZMQ_HAVE_AIX
#include <poll.h>
#endif

// ans ignore: #include "platform.hpp"

#define __STDC_LIMIT_MACROS

// This must be included before any windows headers are compiled.
#if defined ZMQ_HAVE_WINDOWS
// ans ignore: #include "windows.hpp"
#endif

#if defined ZMQ_HAVE_OPENBSD
#define ucred sockpeercred
#endif

// 0MQ definitions and exported functions
// ans ignore: #include "../include/zmq.h"

// 0MQ DRAFT definitions and exported functions
// ans ignore: #include "zmq_draft.h"

// TODO: expand pch implementation to non-windows builds.
#ifdef _MSC_VER

// standard C headers
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <io.h>
#include <ipexport.h>
#include <iphlpapi.h>
#include <limits.h>
#include <Mstcpip.h>
#include <mswsock.h>
#include <process.h>
#include <rpc.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// standard C++ headers
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if _MSC_VER >= 1800
#include <inttypes.h>
#endif

#if _MSC_VER >= 1700
#include <atomic>
#endif

#if defined _WIN32_WCE
#include <cmnintrin.h>
#else
#include <intrin.h>
#endif

#if defined HAVE_LIBGSSAPI_KRB5
// ans ignore: #include "err.hpp"
// ans ignore: #include "msg.hpp"
// ans ignore: #include "mechanism.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "gssapi_server.hpp"
// ans ignore: #include "wire.hpp"
#include <gssapi/gssapi.h>
#include <gssapi/gssapi_krb5.h>
#endif

// ans ignore: #include "options.hpp"

#endif // _MSC_VER

#endif //ifndef __ZMQ_PRECOMPILED_HPP_INCLUDED__


//========= end of #include "precompiled.hpp" ============


//========= begin of #include "udp_address.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_UDP_ADDRESS_HPP_INCLUDED__
#define __ZMQ_UDP_ADDRESS_HPP_INCLUDED__

#if !defined ZMQ_HAVE_WINDOWS
#include <sys/socket.h>
#include <netinet/in.h>
#endif

#include <string>

// ans ignore: #include "ip_resolver.hpp"

namespace zmq
{
class udp_address_t
{
  public:
    udp_address_t ();
    virtual ~udp_address_t ();

    int resolve (const char *name_, bool receiver_, bool ipv6_);

    //  The opposite to resolve()
    virtual int to_string (std::string &addr_);


    int family () const;

    bool is_mcast () const;

    const ip_addr_t *bind_addr () const;
    int bind_if () const;
    const ip_addr_t *target_addr () const;

  private:
    ip_addr_t _bind_address;
    int _bind_interface;
    ip_addr_t _target_address;
    bool _is_multicast;
    std::string _address;
};
}

#endif


//========= end of #include "udp_address.hpp" ============


//========= begin of #include "ipc_address.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_IPC_ADDRESS_HPP_INCLUDED__
#define __ZMQ_IPC_ADDRESS_HPP_INCLUDED__

#include <string>

#if !defined ZMQ_HAVE_WINDOWS && !defined ZMQ_HAVE_OPENVMS                     \
  && !defined ZMQ_HAVE_VXWORKS

#include <sys/socket.h>
#include <sys/un.h>

namespace zmq
{
class ipc_address_t
{
  public:
    ipc_address_t ();
    ipc_address_t (const sockaddr *sa_, socklen_t sa_len_);
    ~ipc_address_t ();

    //  This function sets up the address for UNIX domain transport.
    int resolve (const char *path_);

    //  The opposite to resolve()
    int to_string (std::string &addr_) const;

    const sockaddr *addr () const;
    socklen_t addrlen () const;

  private:
    struct sockaddr_un _address;
    size_t _addrlen;

    ipc_address_t (const ipc_address_t &);
    const ipc_address_t &operator= (const ipc_address_t &);
};
}

#endif

#endif


//========= end of #include "ipc_address.hpp" ============


//========= begin of #include "tipc_address.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TIPC_ADDRESS_HPP_INCLUDED__
#define __ZMQ_TIPC_ADDRESS_HPP_INCLUDED__

#include <string>

// ans ignore: #include "platform.hpp"

#if defined ZMQ_HAVE_TIPC

#include <sys/socket.h>
#if defined ZMQ_HAVE_VXWORKS
#include <tipc/tipc.h>
#else
#include <linux/tipc.h>
#endif

namespace zmq
{
class tipc_address_t
{
  public:
    tipc_address_t ();
    tipc_address_t (const sockaddr *sa, socklen_t sa_len);

    //  This function sets up the address "{type, lower, upper}" for TIPC transport
    int resolve (const char *name);

    //  The opposite to resolve()
    int to_string (std::string &addr_) const;

    // Handling different TIPC address types
    bool is_service () const;
    bool is_random () const;
    void set_random ();

    const sockaddr *addr () const;
    socklen_t addrlen () const;

  private:
    bool _random;
    struct sockaddr_tipc address;
};
}

#endif

#endif


//========= end of #include "tipc_address.hpp" ============


//========= begin of #include "vmci_address.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_VMCI_ADDRESS_HPP_INCLUDED__
#define __ZMQ_VMCI_ADDRESS_HPP_INCLUDED__

#include <string>

// ans ignore: #include "platform.hpp"
// ans ignore: #include "ctx.hpp"

#if defined(ZMQ_HAVE_VMCI)
#include <vmci_sockets.h>

namespace zmq
{
class vmci_address_t
{
  public:
    vmci_address_t (ctx_t *parent_);
    vmci_address_t (const sockaddr *sa, socklen_t sa_len, ctx_t *parent_);
    ~vmci_address_t ();

    //  This function sets up the address for VMCI transport.
    int resolve (const char *path_);

    //  The opposite to resolve()
    int to_string (std::string &addr_);

    const sockaddr *addr () const;
    socklen_t addrlen () const;

  private:
    struct sockaddr_vm address;
    ctx_t *parent;

    vmci_address_t ();
    vmci_address_t (const vmci_address_t &);
    const vmci_address_t &operator= (const vmci_address_t &);
};
}

#endif

#endif


//========= end of #include "vmci_address.hpp" ============


//========= begin of #include "fq.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_FQ_HPP_INCLUDED__
#define __ZMQ_FQ_HPP_INCLUDED__

// ans ignore: #include "array.hpp"
// ans ignore: #include "blob.hpp"

namespace zmq
{
class msg_t;
class pipe_t;

//  Class manages a set of inbound pipes. On receive it performs fair
//  queueing so that senders gone berserk won't cause denial of
//  service for decent senders.

class fq_t
{
  public:
    fq_t ();
    ~fq_t ();

    void attach (pipe_t *pipe_);
    void activated (pipe_t *pipe_);
    void pipe_terminated (pipe_t *pipe_);

    int recv (msg_t *msg_);
    int recvpipe (msg_t *msg_, pipe_t **pipe_);
    bool has_in ();

  private:
    //  Inbound pipes.
    typedef array_t<pipe_t, 1> pipes_t;
    pipes_t _pipes;

    //  Number of active pipes. All the active pipes are located at the
    //  beginning of the pipes array.
    pipes_t::size_type _active;

    //  Pointer to the last pipe we received message from.
    //  NULL when no message has been received or the pipe
    //  has terminated.
    pipe_t *_last_in;

    //  Index of the next bound pipe to read a message from.
    pipes_t::size_type _current;

    //  If true, part of a multipart message was already received, but
    //  there are following parts still waiting in the current pipe.
    bool _more;

    fq_t (const fq_t &);
    const fq_t &operator= (const fq_t &);
};
}

#endif


//========= end of #include "fq.hpp" ============


//========= begin of #include "lb.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_LB_HPP_INCLUDED__
#define __ZMQ_LB_HPP_INCLUDED__

// ans ignore: #include "array.hpp"

namespace zmq
{
class msg_t;
class pipe_t;

//  This class manages a set of outbound pipes. On send it load balances
//  messages fairly among the pipes.

class lb_t
{
  public:
    lb_t ();
    ~lb_t ();

    void attach (pipe_t *pipe_);
    void activated (pipe_t *pipe_);
    void pipe_terminated (pipe_t *pipe_);

    int send (msg_t *msg_);

    //  Sends a message and stores the pipe that was used in pipe_.
    //  It is possible for this function to return success but keep pipe_
    //  unset if the rest of a multipart message to a terminated pipe is
    //  being dropped. For the first frame, this will never happen.
    int sendpipe (msg_t *msg_, pipe_t **pipe_);

    bool has_out ();

  private:
    //  List of outbound pipes.
    typedef array_t<pipe_t, 2> pipes_t;
    pipes_t _pipes;

    //  Number of active pipes. All the active pipes are located at the
    //  beginning of the pipes array.
    pipes_t::size_type _active;

    //  Points to the last pipe that the most recent message was sent to.
    pipes_t::size_type _current;

    //  True if last we are in the middle of a multipart message.
    bool _more;

    //  True if we are dropping current message.
    bool _dropping;

    lb_t (const lb_t &);
    const lb_t &operator= (const lb_t &);
};
}

#endif


//========= end of #include "lb.hpp" ============


//========= begin of #include "client.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CLIENT_HPP_INCLUDED__
#define __ZMQ_CLIENT_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "fq.hpp"
// ans ignore: #include "lb.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;
class io_thread_t;

class client_t : public socket_base_t
{
  public:
    client_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~client_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Messages are fair-queued from inbound pipes. And load-balanced to
    //  the outbound pipes.
    fq_t _fq;
    lb_t _lb;

    client_t (const client_t &);
    const client_t &operator= (const client_t &);
};
}

#endif


//========= end of #include "client.hpp" ============


//========= begin of #include "condition_variable.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CONDITON_VARIABLE_HPP_INCLUDED__
#define __ZMQ_CONDITON_VARIABLE_HPP_INCLUDED__

// ans ignore: #include "err.hpp"
// ans ignore: #include "mutex.hpp"

//  Condition variable class encapsulates OS mutex in a platform-independent way.

#if defined(ZMQ_USE_CV_IMPL_NONE)

namespace zmq
{
class condition_variable_t
{
  public:
    inline condition_variable_t () { zmq_assert (false); }

    inline ~condition_variable_t () {}

    inline int wait (mutex_t *mutex_, int timeout_)
    {
        zmq_assert (false);
        return -1;
    }

    inline void broadcast () { zmq_assert (false); }

  private:
    //  Disable copy construction and assignment.
    condition_variable_t (const condition_variable_t &);
    void operator= (const condition_variable_t &);
};
}

#elif defined(ZMQ_USE_CV_IMPL_WIN32API)

// ans ignore: #include "windows.hpp"

namespace zmq
{
class condition_variable_t
{
  public:
    inline condition_variable_t () { InitializeConditionVariable (&_cv); }

    inline ~condition_variable_t () {}

    inline int wait (mutex_t *mutex_, int timeout_)
    {
        int rc = SleepConditionVariableCS (&_cv, mutex_->get_cs (), timeout_);

        if (rc != 0)
            return 0;

        rc = GetLastError ();

        if (rc != ERROR_TIMEOUT)
            win_assert (rc);

        errno = EAGAIN;
        return -1;
    }

    inline void broadcast () { WakeAllConditionVariable (&_cv); }

  private:
    CONDITION_VARIABLE _cv;

    //  Disable copy construction and assignment.
    condition_variable_t (const condition_variable_t &);
    void operator= (const condition_variable_t &);
};
}

#elif defined(ZMQ_USE_CV_IMPL_STL11)

#include <condition_variable>

namespace zmq
{
class condition_variable_t
{
  public:
    inline condition_variable_t () {}

    inline ~condition_variable_t () {}

    inline int wait (mutex_t *mutex_, int timeout_)
    {
        // this assumes that the mutex mutex_ has been locked by the caller
        int res = 0;
        if (timeout_ == -1) {
            _cv.wait (
              *mutex_); // unlock mtx and wait cv.notify_all(), lock mtx after cv.notify_all()
        } else if (_cv.wait_for (*mutex_, std::chrono::milliseconds (timeout_))
                   == std::cv_status::timeout) {
            // time expired
            errno = EAGAIN;
            res = -1;
        }
        return res;
    }

    inline void broadcast ()
    {
        // this assumes that the mutex associated with _cv has been locked by the caller
        _cv.notify_all ();
    }

  private:
    std::condition_variable_any _cv;

    //  Disable copy construction and assignment.
    condition_variable_t (const condition_variable_t &);
    void operator= (const condition_variable_t &);
};
}

#elif defined(ZMQ_USE_CV_IMPL_VXWORKS)

#include <sysLib.h>

namespace zmq
{
class condition_variable_t
{
  public:
    inline condition_variable_t () {}

    inline ~condition_variable_t ()
    {
        scoped_lock_t l (_listenersMutex);
        for (size_t i = 0; i < _listeners.size (); i++) {
            semDelete (_listeners[i]);
        }
    }

    inline int wait (mutex_t *mutex_, int timeout_)
    {
        //Atomically releases lock, blocks the current executing thread,
        //and adds it to the list of threads waiting on *this. The thread
        //will be unblocked when broadcast() is executed.
        //It may also be unblocked spuriously. When unblocked, regardless
        //of the reason, lock is reacquired and wait exits.

        SEM_ID sem = semBCreate (SEM_Q_PRIORITY, SEM_EMPTY);
        {
            scoped_lock_t l (_listenersMutex);
            _listeners.push_back (sem);
        }
        mutex_->unlock ();

        int rc;
        if (timeout_ < 0)
            rc = semTake (sem, WAIT_FOREVER);
        else {
            int ticksPerSec = sysClkRateGet ();
            int timeoutTicks = (timeout_ * ticksPerSec) / 1000 + 1;
            rc = semTake (sem, timeoutTicks);
        }

        {
            scoped_lock_t l (_listenersMutex);
            // remove sem from listeners
            for (size_t i = 0; i < _listeners.size (); i++) {
                if (_listeners[i] == sem) {
                    _listeners.erase (_listeners.begin () + i);
                    break;
                }
            }
            semDelete (sem);
        }
        mutex_->lock ();

        if (rc == 0)
            return 0;

        if (rc == S_objLib_OBJ_TIMEOUT) {
            errno = EAGAIN;
            return -1;
        }

        return -1;
    }

    inline void broadcast ()
    {
        scoped_lock_t l (_listenersMutex);
        for (size_t i = 0; i < _listeners.size (); i++) {
            semGive (_listeners[i]);
        }
    }

  private:
    mutex_t _listenersMutex;
    std::vector<SEM_ID> _listeners;

    // Disable copy construction and assignment.
    condition_variable_t (const condition_variable_t &);
    const condition_variable_t &operator= (const condition_variable_t &);
};
}

#elif defined(ZMQ_USE_CV_IMPL_PTHREADS)

#include <pthread.h>

#if defined(__ANDROID_API__) && __ANDROID_API__ < 21
#define ANDROID_LEGACY
extern "C" int pthread_cond_timedwait_monotonic_np (pthread_cond_t *,
                                                    pthread_mutex_t *,
                                                    const struct timespec *);
#endif

namespace zmq
{
class condition_variable_t
{
  public:
    inline condition_variable_t ()
    {
        pthread_condattr_t attr;
        pthread_condattr_init (&attr);
#if !defined(ZMQ_HAVE_OSX) && !defined(ANDROID_LEGACY)
        pthread_condattr_setclock (&attr, CLOCK_MONOTONIC);
#endif
        int rc = pthread_cond_init (&_cond, &attr);
        posix_assert (rc);
    }

    inline ~condition_variable_t ()
    {
        int rc = pthread_cond_destroy (&_cond);
        posix_assert (rc);
    }

    inline int wait (mutex_t *mutex_, int timeout_)
    {
        int rc;

        if (timeout_ != -1) {
            struct timespec timeout;

#ifdef ZMQ_HAVE_OSX
            timeout.tv_sec = 0;
            timeout.tv_nsec = 0;
#else
            clock_gettime (CLOCK_MONOTONIC, &timeout);
#endif

            timeout.tv_sec += timeout_ / 1000;
            timeout.tv_nsec += (timeout_ % 1000) * 1000000;

            if (timeout.tv_nsec > 1000000000) {
                timeout.tv_sec++;
                timeout.tv_nsec -= 1000000000;
            }
#ifdef ZMQ_HAVE_OSX
            rc = pthread_cond_timedwait_relative_np (
              &_cond, mutex_->get_mutex (), &timeout);
#elif defined(ANDROID_LEGACY)
            rc = pthread_cond_timedwait_monotonic_np (
              &_cond, mutex_->get_mutex (), &timeout);
#else
            rc =
              pthread_cond_timedwait (&_cond, mutex_->get_mutex (), &timeout);
#endif
        } else
            rc = pthread_cond_wait (&_cond, mutex_->get_mutex ());

        if (rc == 0)
            return 0;

        if (rc == ETIMEDOUT) {
            errno = EAGAIN;
            return -1;
        }

        posix_assert (rc);
        return -1;
    }

    inline void broadcast ()
    {
        int rc = pthread_cond_broadcast (&_cond);
        posix_assert (rc);
    }

  private:
    pthread_cond_t _cond;

    // Disable copy construction and assignment.
    condition_variable_t (const condition_variable_t &);
    const condition_variable_t &operator= (const condition_variable_t &);
};
}

#endif

#endif


//========= end of #include "condition_variable.hpp" ============


//========= begin of #include "io_thread.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_IO_THREAD_HPP_INCLUDED__
#define __ZMQ_IO_THREAD_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"
// ans ignore: #include "object.hpp"
// ans ignore: #include "poller.hpp"
// ans ignore: #include "i_poll_events.hpp"
// ans ignore: #include "mailbox.hpp"

namespace zmq
{
class ctx_t;

//  Generic part of the I/O thread. Polling-mechanism-specific features
//  are implemented in separate "polling objects".

class io_thread_t : public object_t, public i_poll_events
{
  public:
    io_thread_t (zmq::ctx_t *ctx_, uint32_t tid_);

    //  Clean-up. If the thread was started, it's necessary to call 'stop'
    //  before invoking destructor. Otherwise the destructor would hang up.
    ~io_thread_t ();

    //  Launch the physical thread.
    void start ();

    //  Ask underlying thread to stop.
    void stop ();

    //  Returns mailbox associated with this I/O thread.
    mailbox_t *get_mailbox ();

    //  i_poll_events implementation.
    void in_event ();
    void out_event ();
    void timer_event (int id_);

    //  Used by io_objects to retrieve the associated poller object.
    poller_t *get_poller ();

    //  Command handlers.
    void process_stop ();

    //  Returns load experienced by the I/O thread.
    int get_load ();

  private:
    //  I/O thread accesses incoming commands via this mailbox.
    mailbox_t _mailbox;

    //  Handle associated with mailbox' file descriptor.
    poller_t::handle_t _mailbox_handle;

    //  I/O multiplexing is performed using a poller object.
    poller_t *_poller;

    io_thread_t (const io_thread_t &);
    const io_thread_t &operator= (const io_thread_t &);
};
}

#endif


//========= end of #include "io_thread.hpp" ============


//========= begin of #include "reaper.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_REAPER_HPP_INCLUDED__
#define __ZMQ_REAPER_HPP_INCLUDED__

// ans ignore: #include "object.hpp"
// ans ignore: #include "mailbox.hpp"
// ans ignore: #include "poller.hpp"
// ans ignore: #include "i_poll_events.hpp"

namespace zmq
{
class ctx_t;
class socket_base_t;

class reaper_t : public object_t, public i_poll_events
{
  public:
    reaper_t (zmq::ctx_t *ctx_, uint32_t tid_);
    ~reaper_t ();

    mailbox_t *get_mailbox ();

    void start ();
    void stop ();

    //  i_poll_events implementation.
    void in_event ();
    void out_event ();
    void timer_event (int id_);

  private:
    //  Command handlers.
    void process_stop ();
    void process_reap (zmq::socket_base_t *socket_);
    void process_reaped ();

    //  Reaper thread accesses incoming commands via this mailbox.
    mailbox_t _mailbox;

    //  Handle associated with mailbox' file descriptor.
    poller_t::handle_t _mailbox_handle;

    //  I/O multiplexing is performed using a poller object.
    poller_t *_poller;

    //  Number of sockets being reaped at the moment.
    int _sockets;

    //  If true, we were already asked to terminate.
    bool _terminating;

    reaper_t (const reaper_t &);
    const reaper_t &operator= (const reaper_t &);

#ifdef HAVE_FORK
    // the process that created this context. Used to detect forking.
    pid_t _pid;
#endif
};
}

#endif


//========= end of #include "reaper.hpp" ============


//========= begin of #include "random.hpp" ============

/*
    Copyright (c) 2007-2017 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_RANDOM_HPP_INCLUDED__
#define __ZMQ_RANDOM_HPP_INCLUDED__

// ans ignore: #include "stdint.hpp"

namespace zmq
{
//  Seeds the random number generator.
void seed_random ();

//  Generates random value.
uint32_t generate_random ();

//  [De-]Initialise crypto library, if needed.
//  Serialised and refcounted, so that it can be called
//  from multiple threads, each with its own context, and from
//  the various zmq_utils curve functions safely.
void random_open ();
void random_close ();
}

#endif


//========= end of #include "random.hpp" ============


//========= begin of #include "tweetnacl.h" ============

/*
    Copyright (c) 2016-2017 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TWEETNACL_H
#define TWEETNACL_H

#if defined(ZMQ_USE_TWEETNACL)

#define crypto_box_SECRETKEYBYTES 32
#define crypto_box_BOXZEROBYTES 16
#define crypto_box_NONCEBYTES 24
#define crypto_box_ZEROBYTES 32
#define crypto_box_PUBLICKEYBYTES 32
#define crypto_box_BEFORENMBYTES 32
#define crypto_secretbox_KEYBYTES 32
#define crypto_secretbox_NONCEBYTES 24
#define crypto_secretbox_ZEROBYTES 32
#define crypto_secretbox_BOXZEROBYTES 16
typedef unsigned char u8;
typedef unsigned long u32;
typedef unsigned long long u64;
typedef long long i64;
typedef i64 gf[16];

#ifdef __cplusplus
extern "C" {
#endif
void randombytes (unsigned char *, unsigned long long);
/* Do not call manually! Use random_close from random.hpp */
int randombytes_close (void);
/* Do not call manually! Use random_open from random.hpp */
int sodium_init (void);

int crypto_box_keypair (u8 *y_, u8 *x_);
int crypto_box_afternm (
  u8 *c_, const u8 *m_, u64 d_, const u8 *n_, const u8 *k_);
int crypto_box_open_afternm (
  u8 *m_, const u8 *c_, u64 d_, const u8 *n_, const u8 *k_);
int crypto_box (
  u8 *c_, const u8 *m_, u64 d_, const u8 *n_, const u8 *y_, const u8 *x_);
int crypto_box_open (
  u8 *m_, const u8 *c_, u64 d_, const u8 *n_, const u8 *y_, const u8 *x_);
int crypto_box_beforenm (u8 *k_, const u8 *y_, const u8 *x_);
int crypto_scalarmult_base (u8 *q_, const u8 *n_);
int crypto_secretbox (u8 *c_, const u8 *m_, u64 d_, const u8 *n_, const u8 *k_);
int crypto_secretbox_open (
  u8 *m_, const u8 *c_, u64 d_, const u8 *n_, const u8 *k_);
#ifdef __cplusplus
}
#endif

#endif

#endif


//========= end of #include "tweetnacl.h" ============


//========= begin of #include "curve_mechanism_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CURVE_MECHANISM_BASE_HPP_INCLUDED__
#define __ZMQ_CURVE_MECHANISM_BASE_HPP_INCLUDED__

#ifdef ZMQ_HAVE_CURVE

#if defined(ZMQ_USE_TWEETNACL)
// ans ignore: #include "tweetnacl.h"
#elif defined(ZMQ_USE_LIBSODIUM)
// ans ignore: #include "sodium.h"
#endif

#if crypto_box_NONCEBYTES != 24 || crypto_box_PUBLICKEYBYTES != 32             \
  || crypto_box_SECRETKEYBYTES != 32 || crypto_box_ZEROBYTES != 32             \
  || crypto_box_BOXZEROBYTES != 16 || crypto_secretbox_NONCEBYTES != 24        \
  || crypto_secretbox_ZEROBYTES != 32 || crypto_secretbox_BOXZEROBYTES != 16
#error "CURVE library not built properly"
#endif

// ans ignore: #include "mechanism_base.hpp"
// ans ignore: #include "options.hpp"

namespace zmq
{
class curve_mechanism_base_t : public virtual mechanism_base_t
{
  public:
    curve_mechanism_base_t (session_base_t *session_,
                            const options_t &options_,
                            const char *encode_nonce_prefix_,
                            const char *decode_nonce_prefix_);

    // mechanism implementation
    virtual int encode (msg_t *msg_);
    virtual int decode (msg_t *msg_);

  protected:
    const char *encode_nonce_prefix;
    const char *decode_nonce_prefix;

    uint64_t cn_nonce;
    uint64_t cn_peer_nonce;

    //  Intermediary buffer used to speed up boxing and unboxing.
    uint8_t cn_precom[crypto_box_BEFORENMBYTES];
};
}

#endif

#endif


//========= end of #include "curve_mechanism_base.hpp" ============


//========= begin of #include "curve_client_tools.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CURVE_CLIENT_TOOLS_HPP_INCLUDED__
#define __ZMQ_CURVE_CLIENT_TOOLS_HPP_INCLUDED__

#ifdef ZMQ_HAVE_CURVE

#if defined(ZMQ_USE_TWEETNACL)
// ans ignore: #include "tweetnacl.h"
#elif defined(ZMQ_USE_LIBSODIUM)
// ans ignore: #include "sodium.h"
#endif

#if crypto_box_NONCEBYTES != 24 || crypto_box_PUBLICKEYBYTES != 32             \
  || crypto_box_SECRETKEYBYTES != 32 || crypto_box_ZEROBYTES != 32             \
  || crypto_box_BOXZEROBYTES != 16
#error "CURVE library not built properly"
#endif

// ans ignore: #include "wire.hpp"
// ans ignore: #include "err.hpp"

namespace zmq
{
struct curve_client_tools_t
{
    static int produce_hello (void *data_,
                              const uint8_t *server_key_,
                              const uint64_t cn_nonce_,
                              const uint8_t *cn_public_,
                              const uint8_t *cn_secret_)
    {
        uint8_t hello_nonce[crypto_box_NONCEBYTES];
        uint8_t hello_plaintext[crypto_box_ZEROBYTES + 64];
        uint8_t hello_box[crypto_box_BOXZEROBYTES + 80];

        //  Prepare the full nonce
        memcpy (hello_nonce, "CurveZMQHELLO---", 16);
        put_uint64 (hello_nonce + 16, cn_nonce_);

        //  Create Box [64 * %x0](C'->S)
        memset (hello_plaintext, 0, sizeof hello_plaintext);

        int rc = crypto_box (hello_box, hello_plaintext, sizeof hello_plaintext,
                             hello_nonce, server_key_, cn_secret_);
        if (rc == -1)
            return -1;

        uint8_t *hello = static_cast<uint8_t *> (data_);

        memcpy (hello, "\x05HELLO", 6);
        //  CurveZMQ major and minor version numbers
        memcpy (hello + 6, "\1\0", 2);
        //  Anti-amplification padding
        memset (hello + 8, 0, 72);
        //  Client public connection key
        memcpy (hello + 80, cn_public_, crypto_box_PUBLICKEYBYTES);
        //  Short nonce, prefixed by "CurveZMQHELLO---"
        memcpy (hello + 112, hello_nonce + 16, 8);
        //  Signature, Box [64 * %x0](C'->S)
        memcpy (hello + 120, hello_box + crypto_box_BOXZEROBYTES, 80);

        return 0;
    }

    static int process_welcome (const uint8_t *msg_data_,
                                size_t msg_size_,
                                const uint8_t *server_key_,
                                const uint8_t *cn_secret_,
                                uint8_t *cn_server_,
                                uint8_t *cn_cookie_,
                                uint8_t *cn_precom_)
    {
        if (msg_size_ != 168) {
            errno = EPROTO;
            return -1;
        }

        uint8_t welcome_nonce[crypto_box_NONCEBYTES];
        uint8_t welcome_plaintext[crypto_box_ZEROBYTES + 128];
        uint8_t welcome_box[crypto_box_BOXZEROBYTES + 144];

        //  Open Box [S' + cookie](C'->S)
        memset (welcome_box, 0, crypto_box_BOXZEROBYTES);
        memcpy (welcome_box + crypto_box_BOXZEROBYTES, msg_data_ + 24, 144);

        memcpy (welcome_nonce, "WELCOME-", 8);
        memcpy (welcome_nonce + 8, msg_data_ + 8, 16);

        int rc =
          crypto_box_open (welcome_plaintext, welcome_box, sizeof welcome_box,
                           welcome_nonce, server_key_, cn_secret_);
        if (rc != 0) {
            errno = EPROTO;
            return -1;
        }

        memcpy (cn_server_, welcome_plaintext + crypto_box_ZEROBYTES, 32);
        memcpy (cn_cookie_, welcome_plaintext + crypto_box_ZEROBYTES + 32,
                16 + 80);

        //  Message independent precomputation
        rc = crypto_box_beforenm (cn_precom_, cn_server_, cn_secret_);
        zmq_assert (rc == 0);

        return 0;
    }

    static int produce_initiate (void *data_,
                                 size_t size_,
                                 const uint64_t cn_nonce_,
                                 const uint8_t *server_key_,
                                 const uint8_t *public_key_,
                                 const uint8_t *secret_key_,
                                 const uint8_t *cn_public_,
                                 const uint8_t *cn_secret_,
                                 const uint8_t *cn_server_,
                                 const uint8_t *cn_cookie_,
                                 const uint8_t *metadata_plaintext_,
                                 const size_t metadata_length_)
    {
        uint8_t vouch_nonce[crypto_box_NONCEBYTES];
        uint8_t vouch_plaintext[crypto_box_ZEROBYTES + 64];
        uint8_t vouch_box[crypto_box_BOXZEROBYTES + 80];

        //  Create vouch = Box [C',S](C->S')
        memset (vouch_plaintext, 0, crypto_box_ZEROBYTES);
        memcpy (vouch_plaintext + crypto_box_ZEROBYTES, cn_public_, 32);
        memcpy (vouch_plaintext + crypto_box_ZEROBYTES + 32, server_key_, 32);

        memcpy (vouch_nonce, "VOUCH---", 8);
        randombytes (vouch_nonce + 8, 16);

        int rc = crypto_box (vouch_box, vouch_plaintext, sizeof vouch_plaintext,
                             vouch_nonce, cn_server_, secret_key_);
        if (rc == -1)
            return -1;

        uint8_t initiate_nonce[crypto_box_NONCEBYTES];
        uint8_t *initiate_box = static_cast<uint8_t *> (
          malloc (crypto_box_BOXZEROBYTES + 144 + metadata_length_));
        alloc_assert (initiate_box);
        uint8_t *initiate_plaintext = static_cast<uint8_t *> (
          malloc (crypto_box_ZEROBYTES + 128 + metadata_length_));
        alloc_assert (initiate_plaintext);

        //  Create Box [C + vouch + metadata](C'->S')
        memset (initiate_plaintext, 0, crypto_box_ZEROBYTES);
        memcpy (initiate_plaintext + crypto_box_ZEROBYTES, public_key_, 32);
        memcpy (initiate_plaintext + crypto_box_ZEROBYTES + 32, vouch_nonce + 8,
                16);
        memcpy (initiate_plaintext + crypto_box_ZEROBYTES + 48,
                vouch_box + crypto_box_BOXZEROBYTES, 80);
        memcpy (initiate_plaintext + crypto_box_ZEROBYTES + 48 + 80,
                metadata_plaintext_, metadata_length_);

        memcpy (initiate_nonce, "CurveZMQINITIATE", 16);
        put_uint64 (initiate_nonce + 16, cn_nonce_);

        rc = crypto_box (initiate_box, initiate_plaintext,
                         crypto_box_ZEROBYTES + 128 + metadata_length_,
                         initiate_nonce, cn_server_, cn_secret_);
        free (initiate_plaintext);

        if (rc == -1)
            return -1;

        uint8_t *initiate = static_cast<uint8_t *> (data_);

        zmq_assert (size_
                    == 113 + 128 + crypto_box_BOXZEROBYTES + metadata_length_);

        memcpy (initiate, "\x08INITIATE", 9);
        //  Cookie provided by the server in the WELCOME command
        memcpy (initiate + 9, cn_cookie_, 96);
        //  Short nonce, prefixed by "CurveZMQINITIATE"
        memcpy (initiate + 105, initiate_nonce + 16, 8);
        //  Box [C + vouch + metadata](C'->S')
        memcpy (initiate + 113, initiate_box + crypto_box_BOXZEROBYTES,
                128 + metadata_length_ + crypto_box_BOXZEROBYTES);
        free (initiate_box);

        return 0;
    }

    static bool is_handshake_command_welcome (const uint8_t *msg_data_,
                                              const size_t msg_size_)
    {
        return is_handshake_command (msg_data_, msg_size_, "\7WELCOME");
    }

    static bool is_handshake_command_ready (const uint8_t *msg_data_,
                                            const size_t msg_size_)
    {
        return is_handshake_command (msg_data_, msg_size_, "\5READY");
    }

    static bool is_handshake_command_error (const uint8_t *msg_data_,
                                            const size_t msg_size_)
    {
        return is_handshake_command (msg_data_, msg_size_, "\5ERROR");
    }

    //  non-static functions
    curve_client_tools_t (
      const uint8_t (&curve_public_key_)[crypto_box_PUBLICKEYBYTES],
      const uint8_t (&curve_secret_key_)[crypto_box_SECRETKEYBYTES],
      const uint8_t (&curve_server_key_)[crypto_box_PUBLICKEYBYTES])
    {
        int rc;
        memcpy (public_key, curve_public_key_, crypto_box_PUBLICKEYBYTES);
        memcpy (secret_key, curve_secret_key_, crypto_box_SECRETKEYBYTES);
        memcpy (server_key, curve_server_key_, crypto_box_PUBLICKEYBYTES);

        //  Generate short-term key pair
        rc = crypto_box_keypair (cn_public, cn_secret);
        zmq_assert (rc == 0);
    }

    int produce_hello (void *data_, const uint64_t cn_nonce_) const
    {
        return produce_hello (data_, server_key, cn_nonce_, cn_public,
                              cn_secret);
    }

    int process_welcome (const uint8_t *msg_data_,
                         size_t msg_size_,
                         uint8_t *cn_precom_)
    {
        return process_welcome (msg_data_, msg_size_, server_key, cn_secret,
                                cn_server, cn_cookie, cn_precom_);
    }

    int produce_initiate (void *data_,
                          size_t size_,
                          const uint64_t cn_nonce_,
                          const uint8_t *metadata_plaintext_,
                          const size_t metadata_length_)
    {
        return produce_initiate (data_, size_, cn_nonce_, server_key,
                                 public_key, secret_key, cn_public, cn_secret,
                                 cn_server, cn_cookie, metadata_plaintext_,
                                 metadata_length_);
    }

    //  Our public key (C)
    uint8_t public_key[crypto_box_PUBLICKEYBYTES];

    //  Our secret key (c)
    uint8_t secret_key[crypto_box_SECRETKEYBYTES];

    //  Our short-term public key (C')
    uint8_t cn_public[crypto_box_PUBLICKEYBYTES];

    //  Our short-term secret key (c')
    uint8_t cn_secret[crypto_box_SECRETKEYBYTES];

    //  Server's public key (S)
    uint8_t server_key[crypto_box_PUBLICKEYBYTES];

    //  Server's short-term public key (S')
    uint8_t cn_server[crypto_box_PUBLICKEYBYTES];

    //  Cookie received from server
    uint8_t cn_cookie[16 + 80];

  private:
    template <size_t N>
    static bool is_handshake_command (const uint8_t *msg_data_,
                                      const size_t msg_size_,
                                      const char (&prefix_)[N])
    {
        return msg_size_ >= (N - 1) && !memcmp (msg_data_, prefix_, N - 1);
    }
};
}

#endif

#endif


//========= end of #include "curve_client_tools.hpp" ============


//========= begin of #include "curve_client.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CURVE_CLIENT_HPP_INCLUDED__
#define __ZMQ_CURVE_CLIENT_HPP_INCLUDED__

#ifdef ZMQ_HAVE_CURVE

// ans ignore: #include "curve_mechanism_base.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "curve_client_tools.hpp"

namespace zmq
{
class msg_t;
class session_base_t;

class curve_client_t : public curve_mechanism_base_t
{
  public:
    curve_client_t (session_base_t *session_, const options_t &options_);
    virtual ~curve_client_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);
    virtual int encode (msg_t *msg_);
    virtual int decode (msg_t *msg_);
    virtual status_t status () const;

  private:
    enum state_t
    {
        send_hello,
        expect_welcome,
        send_initiate,
        expect_ready,
        error_received,
        connected
    };

    //  Current FSM state
    state_t _state;

    //  CURVE protocol tools
    curve_client_tools_t _tools;

    int produce_hello (msg_t *msg_);
    int process_welcome (const uint8_t *cmd_data_, size_t data_size_);
    int produce_initiate (msg_t *msg_);
    int process_ready (const uint8_t *cmd_data_, size_t data_size_);
    int process_error (const uint8_t *cmd_data_, size_t data_size_);
};
}

#endif

#endif


//========= end of #include "curve_client.hpp" ============


//========= begin of #include "curve_server.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_CURVE_SERVER_HPP_INCLUDED__
#define __ZMQ_CURVE_SERVER_HPP_INCLUDED__

#ifdef ZMQ_HAVE_CURVE

// ans ignore: #include "curve_mechanism_base.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "zap_client.hpp"

namespace zmq
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4250)
#endif
class curve_server_t : public zap_client_common_handshake_t,
                       public curve_mechanism_base_t
{
  public:
    curve_server_t (session_base_t *session_,
                    const std::string &peer_address_,
                    const options_t &options_);
    virtual ~curve_server_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);
    virtual int encode (msg_t *msg_);
    virtual int decode (msg_t *msg_);

  private:
    //  Our secret key (s)
    uint8_t _secret_key[crypto_box_SECRETKEYBYTES];

    //  Our short-term public key (S')
    uint8_t _cn_public[crypto_box_PUBLICKEYBYTES];

    //  Our short-term secret key (s')
    uint8_t _cn_secret[crypto_box_SECRETKEYBYTES];

    //  Client's short-term public key (C')
    uint8_t _cn_client[crypto_box_PUBLICKEYBYTES];

    //  Key used to produce cookie
    uint8_t _cookie_key[crypto_secretbox_KEYBYTES];

    int process_hello (msg_t *msg_);
    int produce_welcome (msg_t *msg_);
    int process_initiate (msg_t *msg_);
    int produce_ready (msg_t *msg_);
    int produce_error (msg_t *msg_) const;

    void send_zap_request (const uint8_t *key_);
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

#endif

#endif


//========= end of #include "curve_server.hpp" ============


//========= begin of #include "dbuffer.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DBUFFER_HPP_INCLUDED__
#define __ZMQ_DBUFFER_HPP_INCLUDED__

#include <stdlib.h>
#include <stddef.h>
#include <algorithm>

// ans ignore: #include "mutex.hpp"
// ans ignore: #include "msg.hpp"

namespace zmq
{
//  dbuffer is a single-producer single-consumer double-buffer
//  implementation.
//
//  The producer writes to a back buffer and then tries to swap
//  pointers between the back and front buffers. If it fails,
//  due to the consumer reading from the front buffer, it just
//  gives up, which is ok since writes are many and redundant.
//
//  The reader simply reads from the front buffer.
//
//  has_msg keeps track of whether there has been a not yet read
//  value written, it is used by ypipe_conflate to mimic ypipe
//  functionality regarding a reader being asleep

template <typename T> class dbuffer_t;

template <> class dbuffer_t<msg_t>
{
  public:
    inline dbuffer_t () :
        _back (&_storage[0]),
        _front (&_storage[1]),
        _has_msg (false)
    {
        _back->init ();
        _front->init ();
    }

    inline ~dbuffer_t ()
    {
        _back->close ();
        _front->close ();
    }

    inline void write (const msg_t &value_)
    {
        msg_t &xvalue = const_cast<msg_t &> (value_);

        zmq_assert (xvalue.check ());
        _back->move (xvalue); // cannot just overwrite, might leak

        zmq_assert (_back->check ());

        if (_sync.try_lock ()) {
            std::swap (_back, _front);
            _has_msg = true;

            _sync.unlock ();
        }
    }

    inline bool read (msg_t *value_)
    {
        if (!value_)
            return false;

        {
            scoped_lock_t lock (_sync);
            if (!_has_msg)
                return false;

            zmq_assert (_front->check ());

            *value_ = *_front;
            _front->init (); // avoid double free

            _has_msg = false;
            return true;
        }
    }


    inline bool check_read ()
    {
        scoped_lock_t lock (_sync);

        return _has_msg;
    }

    inline bool probe (bool (*fn_) (const msg_t &))
    {
        scoped_lock_t lock (_sync);
        return (*fn_) (*_front);
    }


  private:
    msg_t _storage[2];
    msg_t *_back, *_front;

    mutex_t _sync;
    bool _has_msg;

    //  Disable copying of dbuffer.
    dbuffer_t (const dbuffer_t &);
    const dbuffer_t &operator= (const dbuffer_t &);
};
}

#endif


//========= end of #include "dbuffer.hpp" ============


//========= begin of #include "dealer.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DEALER_HPP_INCLUDED__
#define __ZMQ_DEALER_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "fq.hpp"
// ans ignore: #include "lb.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;
class io_thread_t;
class socket_base_t;

class dealer_t : public socket_base_t
{
  public:
    dealer_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~dealer_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

    //  Send and recv - knowing which pipe was used.
    int sendpipe (zmq::msg_t *msg_, zmq::pipe_t **pipe_);
    int recvpipe (zmq::msg_t *msg_, zmq::pipe_t **pipe_);

  private:
    //  Messages are fair-queued from inbound pipes. And load-balanced to
    //  the outbound pipes.
    fq_t _fq;
    lb_t _lb;

    // if true, send an empty message to every connected router peer
    bool _probe_router;

    dealer_t (const dealer_t &);
    const dealer_t &operator= (const dealer_t &);
};
}

#endif


//========= end of #include "dealer.hpp" ============


//========= begin of #include "decoder_allocators.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DECODER_ALLOCATORS_HPP_INCLUDED__
#define __ZMQ_DECODER_ALLOCATORS_HPP_INCLUDED__

#include <cstddef>
#include <cstdlib>

// ans ignore: #include "atomic_counter.hpp"
// ans ignore: #include "msg.hpp"
// ans ignore: #include "err.hpp"

namespace zmq
{
// Static buffer policy.
class c_single_allocator
{
  public:
    explicit c_single_allocator (std::size_t bufsize_) :
        _buf_size (bufsize_),
        _buf (static_cast<unsigned char *> (std::malloc (_buf_size)))
    {
        alloc_assert (_buf);
    }

    ~c_single_allocator () { std::free (_buf); }

    unsigned char *allocate () { return _buf; }

    void deallocate () {}

    std::size_t size () const { return _buf_size; }

    void resize (std::size_t new_size_) { _buf_size = new_size_; }

  private:
    std::size_t _buf_size;
    unsigned char *_buf;

    c_single_allocator (c_single_allocator const &);
    c_single_allocator &operator= (c_single_allocator const &);
};

// This allocator allocates a reference counted buffer which is used by v2_decoder_t
// to use zero-copy msg::init_data to create messages with memory from this buffer as
// data storage.
//
// The buffer is allocated with a reference count of 1 to make sure that is is alive while
// decoding messages. Otherwise, it is possible that e.g. the first message increases the count
// from zero to one, gets passed to the user application, processed in the user thread and deleted
// which would then deallocate the buffer. The drawback is that the buffer may be allocated longer
// than necessary because it is only deleted when allocate is called the next time.
class shared_message_memory_allocator
{
  public:
    explicit shared_message_memory_allocator (std::size_t bufsize_);

    // Create an allocator for a maximum number of messages
    shared_message_memory_allocator (std::size_t bufsize_,
                                     std::size_t max_messages_);

    ~shared_message_memory_allocator ();

    // Allocate a new buffer
    //
    // This releases the current buffer to be bound to the lifetime of the messages
    // created on this buffer.
    unsigned char *allocate ();

    // force deallocation of buffer.
    void deallocate ();

    // Give up ownership of the buffer. The buffer's lifetime is now coupled to
    // the messages constructed on top of it.
    unsigned char *release ();

    void inc_ref ();

    static void call_dec_ref (void *, void *buffer_);

    std::size_t size () const;

    // Return pointer to the first message data byte.
    unsigned char *data ();

    // Return pointer to the first byte of the buffer.
    unsigned char *buffer () { return _buf; }

    void resize (std::size_t new_size_) { _buf_size = new_size_; }

    zmq::msg_t::content_t *provide_content () { return _msg_content; }

    void advance_content () { _msg_content++; }

  private:
    void clear ();

    unsigned char *_buf;
    std::size_t _buf_size;
    const std::size_t _max_size;
    zmq::msg_t::content_t *_msg_content;
    std::size_t _max_counters;
};
}

#endif


//========= end of #include "decoder_allocators.hpp" ============


//========= begin of #include "decoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DECODER_HPP_INCLUDED__
#define __ZMQ_DECODER_HPP_INCLUDED__

#include <algorithm>
#include <cstddef>
#include <cstring>

// ans ignore: #include "decoder_allocators.hpp"
// ans ignore: #include "err.hpp"
// ans ignore: #include "i_decoder.hpp"
// ans ignore: #include "stdint.hpp"

namespace zmq
{
//  Helper base class for decoders that know the amount of data to read
//  in advance at any moment. Knowing the amount in advance is a property
//  of the protocol used. 0MQ framing protocol is based size-prefixed
//  paradigm, which qualifies it to be parsed by this class.
//  On the other hand, XML-based transports (like XMPP or SOAP) don't allow
//  for knowing the size of data to read in advance and should use different
//  decoding algorithms.
//
//  This class implements the state machine that parses the incoming buffer.
//  Derived class should implement individual state machine actions.
//
//  Buffer management is done by an allocator policy.
template <typename T, typename A = c_single_allocator>
class decoder_base_t : public i_decoder
{
  public:
    explicit decoder_base_t (const size_t buf_size_) :
        _next (NULL),
        _read_pos (NULL),
        _to_read (0),
        _allocator (buf_size_)
    {
        _buf = _allocator.allocate ();
    }

    //  The destructor doesn't have to be virtual. It is made virtual
    //  just to keep ICC and code checking tools from complaining.
    virtual ~decoder_base_t () { _allocator.deallocate (); }

    //  Returns a buffer to be filled with binary data.
    void get_buffer (unsigned char **data_, std::size_t *size_)
    {
        _buf = _allocator.allocate ();

        //  If we are expected to read large message, we'll opt for zero-
        //  copy, i.e. we'll ask caller to fill the data directly to the
        //  message. Note that subsequent read(s) are non-blocking, thus
        //  each single read reads at most SO_RCVBUF bytes at once not
        //  depending on how large is the chunk returned from here.
        //  As a consequence, large messages being received won't block
        //  other engines running in the same I/O thread for excessive
        //  amounts of time.
        if (_to_read >= _allocator.size ()) {
            *data_ = _read_pos;
            *size_ = _to_read;
            return;
        }

        *data_ = _buf;
        *size_ = _allocator.size ();
    }

    //  Processes the data in the buffer previously allocated using
    //  get_buffer function. size_ argument specifies number of bytes
    //  actually filled into the buffer. Function returns 1 when the
    //  whole message was decoded or 0 when more data is required.
    //  On error, -1 is returned and errno set accordingly.
    //  Number of bytes processed is returned in bytes_used_.
    int decode (const unsigned char *data_,
                std::size_t size_,
                std::size_t &bytes_used_)
    {
        bytes_used_ = 0;

        //  In case of zero-copy simply adjust the pointers, no copying
        //  is required. Also, run the state machine in case all the data
        //  were processed.
        if (data_ == _read_pos) {
            zmq_assert (size_ <= _to_read);
            _read_pos += size_;
            _to_read -= size_;
            bytes_used_ = size_;

            while (!_to_read) {
                const int rc =
                  (static_cast<T *> (this)->*_next) (data_ + bytes_used_);
                if (rc != 0)
                    return rc;
            }
            return 0;
        }

        while (bytes_used_ < size_) {
            //  Copy the data from buffer to the message.
            const size_t to_copy = std::min (_to_read, size_ - bytes_used_);
            // Only copy when destination address is different from the
            // current address in the buffer.
            if (_read_pos != data_ + bytes_used_) {
                memcpy (_read_pos, data_ + bytes_used_, to_copy);
            }

            _read_pos += to_copy;
            _to_read -= to_copy;
            bytes_used_ += to_copy;
            //  Try to get more space in the message to fill in.
            //  If none is available, return.
            while (_to_read == 0) {
                // pass current address in the buffer
                const int rc =
                  (static_cast<T *> (this)->*_next) (data_ + bytes_used_);
                if (rc != 0)
                    return rc;
            }
        }

        return 0;
    }

    virtual void resize_buffer (std::size_t new_size_)
    {
        _allocator.resize (new_size_);
    }

  protected:
    //  Prototype of state machine action. Action should return false if
    //  it is unable to push the data to the system.
    typedef int (T::*step_t) (unsigned char const *);

    //  This function should be called from derived class to read data
    //  from the buffer and schedule next state machine action.
    void next_step (void *read_pos_, std::size_t to_read_, step_t next_)
    {
        _read_pos = static_cast<unsigned char *> (read_pos_);
        _to_read = to_read_;
        _next = next_;
    }

    A &get_allocator () { return _allocator; }

  private:
    //  Next step. If set to NULL, it means that associated data stream
    //  is dead. Note that there can be still data in the process in such
    //  case.
    step_t _next;

    //  Where to store the read data.
    unsigned char *_read_pos;

    //  How much data to read before taking next step.
    std::size_t _to_read;

    //  The duffer for data to decode.
    A _allocator;
    unsigned char *_buf;

    decoder_base_t (const decoder_base_t &);
    const decoder_base_t &operator= (const decoder_base_t &);
};
}

#endif


//========= end of #include "decoder.hpp" ============


//========= begin of #include "dgram.hpp" ============

/*
    Copyright (c) 2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DGRAM_HPP_INCLUDED__
#define __ZMQ_DGRAM_HPP_INCLUDED__

// ans ignore: #include "blob.hpp"
// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;
class io_thread_t;

class dgram_t : public socket_base_t
{
  public:
    dgram_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~dgram_t ();

    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    zmq::pipe_t *_pipe;

    zmq::pipe_t *_last_in;

    //  If true, more outgoing message parts are expected.
    bool _more_out;

    dgram_t (const dgram_t &);
    const dgram_t &operator= (const dgram_t &);
};
}

#endif


//========= end of #include "dgram.hpp" ============


//========= begin of #include "dist.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DIST_HPP_INCLUDED__
#define __ZMQ_DIST_HPP_INCLUDED__

#include <vector>

// ans ignore: #include "array.hpp"

namespace zmq
{
class pipe_t;
class msg_t;

//  Class manages a set of outbound pipes. It sends each messages to
//  each of them.
class dist_t
{
  public:
    dist_t ();
    ~dist_t ();

    //  Adds the pipe to the distributor object.
    void attach (zmq::pipe_t *pipe_);

    //  Activates pipe that have previously reached high watermark.
    void activated (zmq::pipe_t *pipe_);

    //  Mark the pipe as matching. Subsequent call to send_to_matching
    //  will send message also to this pipe.
    void match (zmq::pipe_t *pipe_);

    //  Marks all pipes that are not matched as matched and vice-versa.
    void reverse_match ();

    //  Mark all pipes as non-matching.
    void unmatch ();

    //  Removes the pipe from the distributor object.
    void pipe_terminated (zmq::pipe_t *pipe_);

    //  Send the message to the matching outbound pipes.
    int send_to_matching (zmq::msg_t *msg_);

    //  Send the message to all the outbound pipes.
    int send_to_all (zmq::msg_t *msg_);

    bool has_out ();

    // check HWM of all pipes matching
    bool check_hwm ();

  private:
    //  Write the message to the pipe. Make the pipe inactive if writing
    //  fails. In such a case false is returned.
    bool write (zmq::pipe_t *pipe_, zmq::msg_t *msg_);

    //  Put the message to all active pipes.
    void distribute (zmq::msg_t *msg_);

    //  List of outbound pipes.
    typedef array_t<zmq::pipe_t, 2> pipes_t;
    pipes_t _pipes;

    //  Number of all the pipes to send the next message to.
    pipes_t::size_type _matching;

    //  Number of active pipes. All the active pipes are located at the
    //  beginning of the pipes array. These are the pipes the messages
    //  can be sent to at the moment.
    pipes_t::size_type _active;

    //  Number of pipes eligible for sending messages to. This includes all
    //  the active pipes plus all the pipes that we can in theory send
    //  messages to (the HWM is not yet reached), but sending a message
    //  to them would result in partial message being delivered, ie. message
    //  with initial parts missing.
    pipes_t::size_type _eligible;

    //  True if last we are in the middle of a multipart message.
    bool _more;

    dist_t (const dist_t &);
    const dist_t &operator= (const dist_t &);
};
}

#endif


//========= end of #include "dist.hpp" ============


//========= begin of #include "dish.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_DISH_HPP_INCLUDED__
#define __ZMQ_DISH_HPP_INCLUDED__

#include <string>

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "dist.hpp"
// ans ignore: #include "fq.hpp"
// ans ignore: #include "msg.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;
class io_thread_t;

class dish_t : public socket_base_t
{
  public:
    dish_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~dish_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xhiccuped (pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);
    int xjoin (const char *group_);
    int xleave (const char *group_);

  private:
    int xxrecv (zmq::msg_t *msg_);

    //  Send subscriptions to a pipe
    void send_subscriptions (pipe_t *pipe_);

    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    //  Object for distributing the subscriptions upstream.
    dist_t _dist;

    //  The repository of subscriptions.
    typedef std::set<std::string> subscriptions_t;
    subscriptions_t _subscriptions;

    //  If true, 'message' contains a matching message to return on the
    //  next recv call.
    bool _has_message;
    msg_t _message;

    dish_t (const dish_t &);
    const dish_t &operator= (const dish_t &);
};

class dish_session_t : public session_base_t
{
  public:
    dish_session_t (zmq::io_thread_t *io_thread_,
                    bool connect_,
                    zmq::socket_base_t *socket_,
                    const options_t &options_,
                    address_t *addr_);
    ~dish_session_t ();

    //  Overrides of the functions from session_base_t.
    int push_msg (msg_t *msg_);
    int pull_msg (msg_t *msg_);
    void reset ();

  private:
    enum
    {
        group,
        body
    } _state;

    msg_t _group_msg;

    dish_session_t (const dish_session_t &);
    const dish_session_t &operator= (const dish_session_t &);
};
}

#endif


//========= end of #include "dish.hpp" ============


//========= begin of #include "encoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ENCODER_HPP_INCLUDED__
#define __ZMQ_ENCODER_HPP_INCLUDED__

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>

// ans ignore: #include "err.hpp"
// ans ignore: #include "i_encoder.hpp"
// ans ignore: #include "msg.hpp"

namespace zmq
{
//  Helper base class for encoders. It implements the state machine that
//  fills the outgoing buffer. Derived classes should implement individual
//  state machine actions.

template <typename T> class encoder_base_t : public i_encoder
{
  public:
    inline explicit encoder_base_t (size_t bufsize_) :
        _write_pos (0),
        _to_write (0),
        _next (NULL),
        _new_msg_flag (false),
        _buf_size (bufsize_),
        _buf (static_cast<unsigned char *> (malloc (bufsize_))),
        _in_progress (NULL)
    {
        alloc_assert (_buf);
    }

    //  The destructor doesn't have to be virtual. It is made virtual
    //  just to keep ICC and code checking tools from complaining.
    inline virtual ~encoder_base_t () { free (_buf); }

    //  The function returns a batch of binary data. The data
    //  are filled to a supplied buffer. If no buffer is supplied (data_
    //  points to NULL) decoder object will provide buffer of its own.
    inline size_t encode (unsigned char **data_, size_t size_)
    {
        unsigned char *buffer = !*data_ ? _buf : *data_;
        size_t buffersize = !*data_ ? _buf_size : size_;

        if (in_progress () == NULL)
            return 0;

        size_t pos = 0;
        while (pos < buffersize) {
            //  If there are no more data to return, run the state machine.
            //  If there are still no data, return what we already have
            //  in the buffer.
            if (!_to_write) {
                if (_new_msg_flag) {
                    int rc = _in_progress->close ();
                    errno_assert (rc == 0);
                    rc = _in_progress->init ();
                    errno_assert (rc == 0);
                    _in_progress = NULL;
                    break;
                }
                (static_cast<T *> (this)->*_next) ();
            }

            //  If there are no data in the buffer yet and we are able to
            //  fill whole buffer in a single go, let's use zero-copy.
            //  There's no disadvantage to it as we cannot stuck multiple
            //  messages into the buffer anyway. Note that subsequent
            //  write(s) are non-blocking, thus each single write writes
            //  at most SO_SNDBUF bytes at once not depending on how large
            //  is the chunk returned from here.
            //  As a consequence, large messages being sent won't block
            //  other engines running in the same I/O thread for excessive
            //  amounts of time.
            if (!pos && !*data_ && _to_write >= buffersize) {
                *data_ = _write_pos;
                pos = _to_write;
                _write_pos = NULL;
                _to_write = 0;
                return pos;
            }

            //  Copy data to the buffer. If the buffer is full, return.
            size_t to_copy = std::min (_to_write, buffersize - pos);
            memcpy (buffer + pos, _write_pos, to_copy);
            pos += to_copy;
            _write_pos += to_copy;
            _to_write -= to_copy;
        }

        *data_ = buffer;
        return pos;
    }

    void load_msg (msg_t *msg_)
    {
        zmq_assert (in_progress () == NULL);
        _in_progress = msg_;
        (static_cast<T *> (this)->*_next) ();
    }

  protected:
    //  Prototype of state machine action.
    typedef void (T::*step_t) ();

    //  This function should be called from derived class to write the data
    //  to the buffer and schedule next state machine action.
    inline void next_step (void *write_pos_,
                           size_t to_write_,
                           step_t next_,
                           bool new_msg_flag_)
    {
        _write_pos = static_cast<unsigned char *> (write_pos_);
        _to_write = to_write_;
        _next = next_;
        _new_msg_flag = new_msg_flag_;
    }

    msg_t *in_progress () { return _in_progress; }

  private:
    //  Where to get the data to write from.
    unsigned char *_write_pos;

    //  How much data to write before next step should be executed.
    size_t _to_write;

    //  Next step. If set to NULL, it means that associated data stream
    //  is dead.
    step_t _next;

    bool _new_msg_flag;

    //  The buffer for encoded data.
    const size_t _buf_size;
    unsigned char *const _buf;

    encoder_base_t (const encoder_base_t &);
    void operator= (const encoder_base_t &);

    msg_t *_in_progress;
};
}

#endif


//========= end of #include "encoder.hpp" ============


//========= begin of #include "gather.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_GATHER_HPP_INCLUDED__
#define __ZMQ_GATHER_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "fq.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;
class msg_t;

class gather_t : public socket_base_t
{
  public:
    gather_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~gather_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    gather_t (const gather_t &);
    const gather_t &operator= (const gather_t &);
};
}

#endif


//========= end of #include "gather.hpp" ============


//========= begin of #include "generic_mtrie.hpp" ============

/*
Copyright (c) 2018 Contributors as noted in the AUTHORS file

This file is part of libzmq, the ZeroMQ core engine in C++.

libzmq is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (LGPL) as published
by the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

As a special exception, the Contributors give you permission to link
this library with independent modules to produce an executable,
regardless of the license terms of these independent modules, and to
copy and distribute the resulting executable under terms of your choice,
provided that you also meet, for each linked independent module, the
terms and conditions of the license of that module. An independent
module is a module which is not derived from or based on this library.
If you modify this library, you must extend this exception to your
version of the library.

libzmq is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_GENERIC_MTRIE_HPP_INCLUDED__
#define __ZMQ_GENERIC_MTRIE_HPP_INCLUDED__

#include <stddef.h>
#include <set>

// ans ignore: #include "stdint.hpp"

namespace zmq
{
//  Multi-trie (prefix tree). Each node in the trie is a set of pointers.
template <typename T> class generic_mtrie_t
{
  public:
    typedef T value_t;
    typedef const unsigned char *prefix_t;

    enum rm_result
    {
        not_found,
        last_value_removed,
        values_remain
    };

    generic_mtrie_t ();
    ~generic_mtrie_t ();

    //  Add key to the trie. Returns true iff no entry with the same prefix_
    //  and size_ existed before.
    bool add (prefix_t prefix_, size_t size_, value_t *value_);

    //  Remove all entries with a specific value from the trie.
    //  The call_on_uniq_ flag controls if the callback is invoked
    //  when there are no entries left on a prefix only (true)
    //  or on every removal (false). The arg_ argument is passed
    //  through to the callback function.
    template <typename Arg>
    void rm (value_t *value_,
             void (*func_) (const unsigned char *data_, size_t size_, Arg arg_),
             Arg arg_,
             bool call_on_uniq_);

    //  Removes a specific entry from the trie.
    //  Returns the result of the operation.
    rm_result rm (prefix_t prefix_, size_t size_, value_t *value_);

    //  Calls a callback function for all matching entries, i.e. any node
    //  corresponding to data_ or a prefix of it. The arg_ argument
    //  is passed through to the callback function.
    template <typename Arg>
    void match (prefix_t data_,
                size_t size_,
                void (*func_) (value_t *value_, Arg arg_),
                Arg arg_);

  private:
    bool add_helper (prefix_t prefix_, size_t size_, value_t *value_);
    template <typename Arg>
    void rm_helper (value_t *value_,
                    unsigned char **buff_,
                    size_t buffsize_,
                    size_t maxbuffsize_,
                    void (*func_) (prefix_t data_, size_t size_, Arg arg_),
                    Arg arg_,
                    bool call_on_uniq_);
    template <typename Arg>
    void rm_helper_multiple_subnodes (unsigned char **buff_,
                                      size_t buffsize_,
                                      size_t maxbuffsize_,
                                      void (*func_) (prefix_t data_,
                                                     size_t size_,
                                                     Arg arg_),
                                      Arg arg_,
                                      bool call_on_uniq_,
                                      value_t *pipe_);

    rm_result rm_helper (prefix_t prefix_, size_t size_, value_t *value_);
    bool is_redundant () const;

    typedef std::set<value_t *> pipes_t;
    pipes_t *_pipes;

    unsigned char _min;
    unsigned short _count;
    unsigned short _live_nodes;
    union
    {
        class generic_mtrie_t<value_t> *node;
        class generic_mtrie_t<value_t> **table;
    } _next;

    generic_mtrie_t (const generic_mtrie_t<value_t> &);
    const generic_mtrie_t<value_t> &
    operator= (const generic_mtrie_t<value_t> &);
};
}

#endif


//========= end of #include "generic_mtrie.hpp" ============


//========= begin of #include "generic_mtrie_impl.hpp" ============

/*
Copyright (c) 2018 Contributors as noted in the AUTHORS file

This file is part of libzmq, the ZeroMQ core engine in C++.

libzmq is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (LGPL) as published
by the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

As a special exception, the Contributors give you permission to link
this library with independent modules to produce an executable,
regardless of the license terms of these independent modules, and to
copy and distribute the resulting executable under terms of your choice,
provided that you also meet, for each linked independent module, the
terms and conditions of the license of that module. An independent
module is a module which is not derived from or based on this library.
If you modify this library, you must extend this exception to your
version of the library.

libzmq is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_GENERIC_MTRIE_IMPL_HPP_INCLUDED__
#define __ZMQ_GENERIC_MTRIE_IMPL_HPP_INCLUDED__


#include <stdlib.h>

#include <new>
#include <algorithm>

// ans ignore: #include "err.hpp"
// ans ignore: #include "macros.hpp"
// ans ignore: #include "generic_mtrie.hpp"

template <typename T>
zmq::generic_mtrie_t<T>::generic_mtrie_t () :
    _pipes (0),
    _min (0),
    _count (0),
    _live_nodes (0)
{
}

template <typename T> zmq::generic_mtrie_t<T>::~generic_mtrie_t ()
{
    LIBZMQ_DELETE (_pipes);

    if (_count == 1) {
        zmq_assert (_next.node);
        LIBZMQ_DELETE (_next.node);
    } else if (_count > 1) {
        for (unsigned short i = 0; i != _count; ++i) {
            LIBZMQ_DELETE (_next.table[i]);
        }
        free (_next.table);
    }
}

template <typename T>
bool zmq::generic_mtrie_t<T>::add (prefix_t prefix_,
                                   size_t size_,
                                   value_t *pipe_)
{
    return add_helper (prefix_, size_, pipe_);
}

template <typename T>
bool zmq::generic_mtrie_t<T>::add_helper (prefix_t prefix_,
                                          size_t size_,
                                          value_t *pipe_)
{
    //  We are at the node corresponding to the prefix. We are done.
    if (!size_) {
        const bool result = !_pipes;
        if (!_pipes) {
            _pipes = new (std::nothrow) pipes_t;
            alloc_assert (_pipes);
        }
        _pipes->insert (pipe_);
        return result;
    }

    const unsigned char c = *prefix_;
    if (c < _min || c >= _min + _count) {
        //  The character is out of range of currently handled
        //  characters. We have to extend the table.
        if (!_count) {
            _min = c;
            _count = 1;
            _next.node = NULL;
        } else if (_count == 1) {
            const unsigned char oldc = _min;
            generic_mtrie_t *oldp = _next.node;
            _count = (_min < c ? c - _min : _min - c) + 1;
            _next.table = static_cast<generic_mtrie_t **> (
              malloc (sizeof (generic_mtrie_t *) * _count));
            alloc_assert (_next.table);
            for (unsigned short i = 0; i != _count; ++i)
                _next.table[i] = 0;
            _min = std::min (_min, c);
            _next.table[oldc - _min] = oldp;
        } else if (_min < c) {
            //  The new character is above the current character range.
            const unsigned short old_count = _count;
            _count = c - _min + 1;
            _next.table = static_cast<generic_mtrie_t **> (
              realloc (_next.table, sizeof (generic_mtrie_t *) * _count));
            alloc_assert (_next.table);
            for (unsigned short i = old_count; i != _count; i++)
                _next.table[i] = NULL;
        } else {
            //  The new character is below the current character range.
            const unsigned short old_count = _count;
            _count = (_min + old_count) - c;
            _next.table = static_cast<generic_mtrie_t **> (
              realloc (_next.table, sizeof (generic_mtrie_t *) * _count));
            alloc_assert (_next.table);
            memmove (_next.table + _min - c, _next.table,
                     old_count * sizeof (generic_mtrie_t *));
            for (unsigned short i = 0; i != _min - c; i++)
                _next.table[i] = NULL;
            _min = c;
        }
    }

    //  If next node does not exist, create one.
    if (_count == 1) {
        if (!_next.node) {
            _next.node = new (std::nothrow) generic_mtrie_t;
            alloc_assert (_next.node);
            ++_live_nodes;
        }
        return _next.node->add_helper (prefix_ + 1, size_ - 1, pipe_);
    }
    if (!_next.table[c - _min]) {
        _next.table[c - _min] = new (std::nothrow) generic_mtrie_t;
        alloc_assert (_next.table[c - _min]);
        ++_live_nodes;
    }
    return _next.table[c - _min]->add_helper (prefix_ + 1, size_ - 1, pipe_);
}


template <typename T>
template <typename Arg>
void zmq::generic_mtrie_t<T>::rm (value_t *pipe_,
                                  void (*func_) (prefix_t data_,
                                                 size_t size_,
                                                 Arg arg_),
                                  Arg arg_,
                                  bool call_on_uniq_)
{
    unsigned char *buff = NULL;
    rm_helper (pipe_, &buff, 0, 0, func_, arg_, call_on_uniq_);
    free (buff);
}

template <typename T>
template <typename Arg>
void zmq::generic_mtrie_t<T>::rm_helper (value_t *pipe_,
                                         unsigned char **buff_,
                                         size_t buffsize_,
                                         size_t maxbuffsize_,
                                         void (*func_) (prefix_t data_,
                                                        size_t size_,
                                                        Arg arg_),
                                         Arg arg_,
                                         bool call_on_uniq_)
{
    //  Remove the subscription from this node.
    if (_pipes && _pipes->erase (pipe_)) {
        if (!call_on_uniq_ || _pipes->empty ()) {
            func_ (*buff_, buffsize_, arg_);
        }

        if (_pipes->empty ()) {
            LIBZMQ_DELETE (_pipes);
        }
    }

    //  Adjust the buffer.
    if (buffsize_ >= maxbuffsize_) {
        maxbuffsize_ = buffsize_ + 256;
        *buff_ = static_cast<unsigned char *> (realloc (*buff_, maxbuffsize_));
        alloc_assert (*buff_);
    }

    switch (_count) {
        case 0:
            //  If there are no subnodes in the trie, return.
            break;
        case 1:
            //  If there's one subnode (optimisation).

            (*buff_)[buffsize_] = _min;
            buffsize_++;
            _next.node->rm_helper (pipe_, buff_, buffsize_, maxbuffsize_, func_,
                                   arg_, call_on_uniq_);

            //  Prune the node if it was made redundant by the removal
            if (_next.node->is_redundant ()) {
                LIBZMQ_DELETE (_next.node);
                _count = 0;
                --_live_nodes;
                zmq_assert (_live_nodes == 0);
            }
            break;
        default:
            //  If there are multiple subnodes.
            rm_helper_multiple_subnodes (buff_, buffsize_, maxbuffsize_, func_,
                                         arg_, call_on_uniq_, pipe_);
            break;
    }
}

template <typename T>
template <typename Arg>
void zmq::generic_mtrie_t<T>::rm_helper_multiple_subnodes (
  unsigned char **buff_,
  size_t buffsize_,
  size_t maxbuffsize_,
  void (*func_) (prefix_t data_, size_t size_, Arg arg_),
  Arg arg_,
  bool call_on_uniq_,
  value_t *pipe_)
{
    //  New min non-null character in the node table after the removal
    unsigned char new_min = _min + _count - 1;
    //  New max non-null character in the node table after the removal
    unsigned char new_max = _min;
    for (unsigned short c = 0; c != _count; c++) {
        (*buff_)[buffsize_] = _min + c;
        if (_next.table[c]) {
            _next.table[c]->rm_helper (pipe_, buff_, buffsize_ + 1,
                                       maxbuffsize_, func_, arg_,
                                       call_on_uniq_);

            //  Prune redundant nodes from the mtrie
            if (_next.table[c]->is_redundant ()) {
                LIBZMQ_DELETE (_next.table[c]);

                zmq_assert (_live_nodes > 0);
                --_live_nodes;
            } else {
                //  The node is not redundant, so it's a candidate for being
                //  the new min/max node.
                //
                //  We loop through the node array from left to right, so the
                //  first non-null, non-redundant node encountered is the new
                //  minimum index. Conversely, the last non-redundant, non-null
                //  node encountered is the new maximum index.
                if (c + _min < new_min)
                    new_min = c + _min;
                if (c + _min > new_max)
                    new_max = c + _min;
            }
        }
    }

    zmq_assert (_count > 1);

    //  Free the node table if it's no longer used.
    switch (_live_nodes) {
        case 0:
            free (_next.table);
            _next.table = NULL;
            _count = 0;
            break;
        case 1:
            //  Compact the node table if possible

            //  If there's only one live node in the table we can
            //  switch to using the more compact single-node
            //  representation
            zmq_assert (new_min == new_max);
            zmq_assert (new_min >= _min && new_min < _min + _count);
            {
                generic_mtrie_t *node = _next.table[new_min - _min];
                zmq_assert (node);
                free (_next.table);
                _next.node = node;
            }
            _count = 1;
            _min = new_min;
            break;
        default:
            if (new_min > _min || new_max < _min + _count - 1) {
                zmq_assert (new_max - new_min + 1 > 1);

                generic_mtrie_t **old_table = _next.table;
                zmq_assert (new_min > _min || new_max < _min + _count - 1);
                zmq_assert (new_min >= _min);
                zmq_assert (new_max <= _min + _count - 1);
                zmq_assert (new_max - new_min + 1 < _count);

                _count = new_max - new_min + 1;
                _next.table = static_cast<generic_mtrie_t **> (
                  malloc (sizeof (generic_mtrie_t *) * _count));
                alloc_assert (_next.table);

                memmove (_next.table, old_table + (new_min - _min),
                         sizeof (generic_mtrie_t *) * _count);
                free (old_table);

                _min = new_min;
            }
    }
}
template <typename T>
typename zmq::generic_mtrie_t<T>::rm_result
zmq::generic_mtrie_t<T>::rm (prefix_t prefix_, size_t size_, value_t *pipe_)
{
    return rm_helper (prefix_, size_, pipe_);
}

template <typename T>
typename zmq::generic_mtrie_t<T>::rm_result zmq::generic_mtrie_t<T>::rm_helper (
  prefix_t prefix_, size_t size_, value_t *pipe_)
{
    if (!size_) {
        if (!_pipes)
            return not_found;

        typename pipes_t::size_type erased = _pipes->erase (pipe_);
        if (_pipes->empty ()) {
            zmq_assert (erased == 1);
            LIBZMQ_DELETE (_pipes);
            return last_value_removed;
        }
        return (erased == 1) ? values_remain : not_found;
    }

    const unsigned char c = *prefix_;
    if (!_count || c < _min || c >= _min + _count)
        return not_found;

    generic_mtrie_t *next_node =
      _count == 1 ? _next.node : _next.table[c - _min];

    if (!next_node)
        return not_found;

    const rm_result ret = next_node->rm_helper (prefix_ + 1, size_ - 1, pipe_);

    if (next_node->is_redundant ()) {
        LIBZMQ_DELETE (next_node);
        zmq_assert (_count > 0);

        if (_count == 1) {
            _next.node = 0;
            _count = 0;
            --_live_nodes;
            zmq_assert (_live_nodes == 0);
        } else {
            _next.table[c - _min] = 0;
            zmq_assert (_live_nodes > 1);
            --_live_nodes;

            //  Compact the table if possible
            if (_live_nodes == 1) {
                //  If there's only one live node in the table we can
                //  switch to using the more compact single-node
                //  representation
                unsigned short i;
                for (i = 0; i < _count; ++i)
                    if (_next.table[i])
                        break;

                zmq_assert (i < _count);
                _min += i;
                _count = 1;
                generic_mtrie_t *oldp = _next.table[i];
                free (_next.table);
                _next.node = oldp;
            } else if (c == _min) {
                //  We can compact the table "from the left"
                unsigned short i;
                for (i = 1; i < _count; ++i)
                    if (_next.table[i])
                        break;

                zmq_assert (i < _count);
                _min += i;
                _count -= i;
                generic_mtrie_t **old_table = _next.table;
                _next.table = static_cast<generic_mtrie_t **> (
                  malloc (sizeof (generic_mtrie_t *) * _count));
                alloc_assert (_next.table);
                memmove (_next.table, old_table + i,
                         sizeof (generic_mtrie_t *) * _count);
                free (old_table);
            } else if (c == _min + _count - 1) {
                //  We can compact the table "from the right"
                unsigned short i;
                for (i = 1; i < _count; ++i)
                    if (_next.table[_count - 1 - i])
                        break;

                zmq_assert (i < _count);
                _count -= i;
                generic_mtrie_t **old_table = _next.table;
                _next.table = static_cast<generic_mtrie_t **> (
                  malloc (sizeof (generic_mtrie_t *) * _count));
                alloc_assert (_next.table);
                memmove (_next.table, old_table,
                         sizeof (generic_mtrie_t *) * _count);
                free (old_table);
            }
        }
    }

    return ret;
}

template <typename T>
template <typename Arg>
void zmq::generic_mtrie_t<T>::match (prefix_t data_,
                                     size_t size_,
                                     void (*func_) (value_t *pipe_, Arg arg_),
                                     Arg arg_)
{
    generic_mtrie_t *current = this;
    while (true) {
        //  Signal the pipes attached to this node.
        if (current->_pipes) {
            for (typename pipes_t::iterator it = current->_pipes->begin ();
                 it != current->_pipes->end (); ++it)
                func_ (*it, arg_);
        }

        //  If we are at the end of the message, there's nothing more to match.
        if (!size_)
            break;

        //  If there are no subnodes in the trie, return.
        if (current->_count == 0)
            break;

        //  If there's one subnode (optimisation).
        if (current->_count == 1) {
            if (data_[0] != current->_min)
                break;
            current = current->_next.node;
            data_++;
            size_--;
            continue;
        }

        //  If there are multiple subnodes.
        if (data_[0] < current->_min
            || data_[0] >= current->_min + current->_count)
            break;
        if (!current->_next.table[data_[0] - current->_min])
            break;
        current = current->_next.table[data_[0] - current->_min];
        data_++;
        size_--;
    }
}

template <typename T> bool zmq::generic_mtrie_t<T>::is_redundant () const
{
    return !_pipes && _live_nodes == 0;
}


#endif


//========= end of #include "generic_mtrie_impl.hpp" ============


//========= begin of #include "gssapi_client.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_GSSAPI_CLIENT_HPP_INCLUDED__
#define __ZMQ_GSSAPI_CLIENT_HPP_INCLUDED__

#ifdef HAVE_LIBGSSAPI_KRB5

// ans ignore: #include "gssapi_mechanism_base.hpp"

namespace zmq
{
class msg_t;
class session_base_t;

class gssapi_client_t : public gssapi_mechanism_base_t
{
  public:
    gssapi_client_t (session_base_t *session_, const options_t &options_);
    virtual ~gssapi_client_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);
    virtual int encode (msg_t *msg_);
    virtual int decode (msg_t *msg_);
    virtual status_t status () const;

  private:
    enum state_t
    {
        call_next_init,
        send_next_token,
        recv_next_token,
        send_ready,
        recv_ready,
        connected
    };

    //  Human-readable principal name of the service we are connecting to
    char *service_name;

    gss_OID service_name_type;

    //  Current FSM state
    state_t state;

    //  Points to either send_tok or recv_tok
    //  during context initialization
    gss_buffer_desc *token_ptr;

    //  The desired underlying mechanism
    gss_OID_set_desc mechs;

    //  True iff client considers the server authenticated
    bool security_context_established;

    int initialize_context ();
    int produce_next_token (msg_t *msg_);
    int process_next_token (msg_t *msg_);
};
}

#endif

#endif


//========= end of #include "gssapi_client.hpp" ============


//========= begin of #include "ip.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_IP_HPP_INCLUDED__
#define __ZMQ_IP_HPP_INCLUDED__

#include <string>
// ans ignore: #include "fd.hpp"

namespace zmq
{
//  Same as socket(2), but allows for transparent tweaking the options.
fd_t open_socket (int domain_, int type_, int protocol_);

//  Sets the socket into non-blocking mode.
void unblock_socket (fd_t s_);

//  Enable IPv4-mapping of addresses in case it is disabled by default.
void enable_ipv4_mapping (fd_t s_);

//  Returns string representation of peer's address.
//  Socket sockfd_ must be connected. Returns true iff successful.
int get_peer_ip_address (fd_t sockfd_, std::string &ip_addr_);

// Sets the IP Type-Of-Service for the underlying socket
void set_ip_type_of_service (fd_t s_, int iptos_);

// Sets the SO_NOSIGPIPE option for the underlying socket.
// Return 0 on success, -1 if the connection has been closed by the peer
int set_nosigpipe (fd_t s_);

// Binds the underlying socket to the given device, eg. VRF or interface
int bind_to_device (fd_t s_, const std::string &bound_device_);

// Initialize network subsystem. May be called multiple times. Each call must be matched by a call to shutdown_network.
bool initialize_network ();

// Shutdown network subsystem. Must be called once for each call to initialize_network before terminating.
void shutdown_network ();

// Creates a pair of sockets (using signaler_port on OS using TCP sockets).
// Returns -1 if we could not make the socket pair successfully
int make_fdpair (fd_t *r_, fd_t *w_);

// Makes a socket non-inheritable to child processes.
// Asserts on any failure.
void make_socket_noninheritable (fd_t sock_);
}

#endif


//========= end of #include "ip.hpp" ============


//========= begin of #include "tcp.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TCP_HPP_INCLUDED__
#define __ZMQ_TCP_HPP_INCLUDED__

// ans ignore: #include "fd.hpp"

namespace zmq
{
class tcp_address_t;
struct options_t;

//  Tunes the supplied TCP socket for the best latency.
int tune_tcp_socket (fd_t s_);

//  Sets the socket send buffer size.
int set_tcp_send_buffer (fd_t sockfd_, int bufsize_);

//  Sets the socket receive buffer size.
int set_tcp_receive_buffer (fd_t sockfd_, int bufsize_);

//  Tunes TCP keep-alives
int tune_tcp_keepalives (fd_t s_,
                         int keepalive_,
                         int keepalive_cnt_,
                         int keepalive_idle_,
                         int keepalive_intvl_);

//  Tunes TCP max retransmit timeout
int tune_tcp_maxrt (fd_t sockfd_, int timeout_);

//  Writes data to the socket. Returns the number of bytes actually
//  written (even zero is to be considered to be a success). In case
//  of error or orderly shutdown by the other peer -1 is returned.
int tcp_write (fd_t s_, const void *data_, size_t size_);

//  Reads data from the socket (up to 'size' bytes).
//  Returns the number of bytes actually read or -1 on error.
//  Zero indicates the peer has closed the connection.
int tcp_read (fd_t s_, void *data_, size_t size_);

//  Asserts that an internal error did not occur.  Does not assert
//  on network errors such as reset or aborted connections.
void tcp_assert_tuning_error (fd_t s_, int rc_);

void tcp_tune_loopback_fast_path (const fd_t socket_);

//  Resolves the given address_ string, opens a socket and sets socket options
//  according to the passed options_. On success, returns the socket
//  descriptor and assigns the resolved address to out_tcp_addr_. In case of
//  an error, retired_fd is returned, and the value of out_tcp_addr_ is undefined.
//  errno is set to an error code describing the cause of the error.
fd_t tcp_open_socket (const char *address_,
                      const options_t &options_,
                      bool local_,
                      bool fallback_to_ipv4_,
                      tcp_address_t *out_tcp_addr_);
}

#endif


//========= end of #include "tcp.hpp" ============


//========= begin of #include "stream_connecter_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __STREAM_CONNECTER_BASE_HPP_INCLUDED__
#define __STREAM_CONNECTER_BASE_HPP_INCLUDED__

// ans ignore: #include "fd.hpp"
// ans ignore: #include "own.hpp"
// ans ignore: #include "io_object.hpp"

namespace zmq
{
class io_thread_t;
class session_base_t;
struct address_t;

class stream_connecter_base_t : public own_t, public io_object_t
{
  public:
    //  If 'delayed_start' is true connecter first waits for a while,
    //  then starts connection process.
    stream_connecter_base_t (zmq::io_thread_t *io_thread_,
                             zmq::session_base_t *session_,
                             const options_t &options_,
                             address_t *addr_,
                             bool delayed_start_);

    ~stream_connecter_base_t ();

  protected:
    //  Handlers for incoming commands.
    void process_plug ();
    void process_term (int linger_);

    //  Handlers for I/O events.
    void in_event ();
    void timer_event (int id_);

    //  Internal function to create the engine after connection was established.
    void create_engine (fd_t fd, const std::string &local_address_);

    //  Internal function to add a reconnect timer
    void add_reconnect_timer ();

    //  Removes the handle from the poller.
    void rm_handle ();

    //  Close the connecting socket.
    void close ();

    //  Address to connect to. Owned by session_base_t.
    //  It is non-const since some parts may change during opening.
    address_t *const _addr;

    //  Underlying socket.
    fd_t _s;

    //  Handle corresponding to the listening socket, if file descriptor is
    //  registered with the poller, or NULL.
    handle_t _handle;

    // String representation of endpoint to connect to
    std::string _endpoint;

    // Socket
    zmq::socket_base_t *const _socket;

  private:
    //  ID of the timer used to delay the reconnection.
    enum
    {
        reconnect_timer_id = 1
    };

    //  Internal function to return a reconnect backoff delay.
    //  Will modify the current_reconnect_ivl used for next call
    //  Returns the currently used interval
    int get_new_reconnect_ivl ();

    virtual void start_connecting () = 0;

    //  If true, connecter is waiting a while before trying to connect.
    const bool _delayed_start;

    //  True iff a timer has been started.
    bool _reconnect_timer_started;

    //  Reference to the session we belong to.
    zmq::session_base_t *const _session;

    //  Current reconnect ivl, updated for backoff strategy
    int _current_reconnect_ivl;

    stream_connecter_base_t (const stream_connecter_base_t &);
    const stream_connecter_base_t &operator= (const stream_connecter_base_t &);
};
}

#endif


//========= end of #include "stream_connecter_base.hpp" ============


//========= begin of #include "ipc_connecter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __IPC_CONNECTER_HPP_INCLUDED__
#define __IPC_CONNECTER_HPP_INCLUDED__

#if !defined ZMQ_HAVE_WINDOWS && !defined ZMQ_HAVE_OPENVMS                     \
  && !defined ZMQ_HAVE_VXWORKS

// ans ignore: #include "fd.hpp"
// ans ignore: #include "stream_connecter_base.hpp"

namespace zmq
{
class ipc_connecter_t : public stream_connecter_base_t
{
  public:
    //  If 'delayed_start' is true connecter first waits for a while,
    //  then starts connection process.
    ipc_connecter_t (zmq::io_thread_t *io_thread_,
                     zmq::session_base_t *session_,
                     const options_t &options_,
                     address_t *addr_,
                     bool delayed_start_);

  private:
    //  Handlers for I/O events.
    void out_event ();

    //  Internal function to start the actual connection establishment.
    void start_connecting ();

    //  Open IPC connecting socket. Returns -1 in case of error,
    //  0 if connect was successful immediately. Returns -1 with
    //  EAGAIN errno if async connect was launched.
    int open ();

    //  Get the file descriptor of newly created connection. Returns
    //  retired_fd if the connection was unsuccessful.
    fd_t connect ();

    ipc_connecter_t (const ipc_connecter_t &);
    const ipc_connecter_t &operator= (const ipc_connecter_t &);
};
}

#endif

#endif


//========= end of #include "ipc_connecter.hpp" ============


//========= begin of #include "stream_listener_base.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_STREAM_LISTENER_BASE_HPP_INCLUDED__
#define __ZMQ_STREAM_LISTENER_BASE_HPP_INCLUDED__

#include <string>

// ans ignore: #include "fd.hpp"
// ans ignore: #include "own.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "io_object.hpp"
// ans ignore: #include "address.hpp"

namespace zmq
{
class io_thread_t;
class socket_base_t;

class stream_listener_base_t : public own_t, public io_object_t
{
  public:
    stream_listener_base_t (zmq::io_thread_t *io_thread_,
                            zmq::socket_base_t *socket_,
                            const options_t &options_);
    ~stream_listener_base_t ();

    // Get the bound address for use with wildcards
    int get_local_address (std::string &addr_) const;

  protected:
    virtual std::string get_socket_name (fd_t fd_,
                                         socket_end_t socket_end_) const = 0;

  private:
    //  Handlers for incoming commands.
    void process_plug ();
    void process_term (int linger_);

  protected:
    //  Close the listening socket.
    virtual int close ();

    void create_engine (fd_t fd);

    //  Underlying socket.
    fd_t _s;

    //  Handle corresponding to the listening socket.
    handle_t _handle;

    //  Socket the listener belongs to.
    zmq::socket_base_t *_socket;

    // String representation of endpoint to bind to
    std::string _endpoint;

  private:
    stream_listener_base_t (const stream_listener_base_t &);
    const stream_listener_base_t &operator= (const stream_listener_base_t &);
};
}

#endif


//========= end of #include "stream_listener_base.hpp" ============


//========= begin of #include "ipc_listener.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_IPC_LISTENER_HPP_INCLUDED__
#define __ZMQ_IPC_LISTENER_HPP_INCLUDED__

#if !defined ZMQ_HAVE_WINDOWS && !defined ZMQ_HAVE_OPENVMS                     \
  && !defined ZMQ_HAVE_VXWORKS

#include <string>

// ans ignore: #include "fd.hpp"
// ans ignore: #include "stream_listener_base.hpp"

namespace zmq
{
class ipc_listener_t : public stream_listener_base_t
{
  public:
    ipc_listener_t (zmq::io_thread_t *io_thread_,
                    zmq::socket_base_t *socket_,
                    const options_t &options_);

    //  Set address to listen on.
    int set_local_address (const char *addr_);

  protected:
    std::string get_socket_name (fd_t fd_, socket_end_t socket_end_) const;

  private:
    //  Handlers for I/O events.
    void in_event ();

    // Create wildcard path address
    static int create_wildcard_address (std::string &path_, std::string &file_);

    //  Filter new connections if the OS provides a mechanism to get
    //  the credentials of the peer process.  Called from accept().
#if defined ZMQ_HAVE_SO_PEERCRED || defined ZMQ_HAVE_LOCAL_PEERCRED
    bool filter (fd_t sock_);
#endif

    int close ();

    //  Accept the new connection. Returns the file descriptor of the
    //  newly created connection. The function may return retired_fd
    //  if the connection was dropped while waiting in the listen backlog.
    fd_t accept ();

    //  True, if the underlying file for UNIX domain socket exists.
    bool _has_file;

    //  Name of the temporary directory (if any) that has the
    //  the UNIX domain socket
    std::string _tmp_socket_dirname;

    //  Name of the file associated with the UNIX domain address.
    std::string _filename;

    // Acceptable temporary directory environment variables
    static const char *tmp_env_vars[];

    ipc_listener_t (const ipc_listener_t &);
    const ipc_listener_t &operator= (const ipc_listener_t &);
};
}

#endif

#endif


//========= end of #include "ipc_listener.hpp" ============


//========= begin of #include "mailbox_safe.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MAILBOX_SAFE_HPP_INCLUDED__
#define __ZMQ_MAILBOX_SAFE_HPP_INCLUDED__

#include <vector>
#include <stddef.h>

// ans ignore: #include "signaler.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "config.hpp"
// ans ignore: #include "command.hpp"
// ans ignore: #include "ypipe.hpp"
// ans ignore: #include "mutex.hpp"
// ans ignore: #include "i_mailbox.hpp"
// ans ignore: #include "condition_variable.hpp"

namespace zmq
{
class mailbox_safe_t : public i_mailbox
{
  public:
    mailbox_safe_t (mutex_t *sync_);
    ~mailbox_safe_t ();

    void send (const command_t &cmd_);
    int recv (command_t *cmd_, int timeout_);

    // Add signaler to mailbox which will be called when a message is ready
    void add_signaler (signaler_t *signaler_);
    void remove_signaler (signaler_t *signaler_);
    void clear_signalers ();

#ifdef HAVE_FORK
    // close the file descriptors in the signaller. This is used in a forked
    // child process to close the file descriptors so that they do not interfere
    // with the context in the parent process.
    void forked ()
    {
        // TODO: call fork on the condition variable
    }
#endif

  private:
    //  The pipe to store actual commands.
    typedef ypipe_t<command_t, command_pipe_granularity> cpipe_t;
    cpipe_t _cpipe;

    //  Condition variable to pass signals from writer thread to reader thread.
    condition_variable_t _cond_var;

    //  Synchronize access to the mailbox from receivers and senders
    mutex_t *const _sync;

    std::vector<zmq::signaler_t *> _signalers;

    //  Disable copying of mailbox_t object.
    mailbox_safe_t (const mailbox_safe_t &);
    const mailbox_safe_t &operator= (const mailbox_safe_t &);
};
}

#endif


//========= end of #include "mailbox_safe.hpp" ============


//========= begin of #include "mtrie.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_MTRIE_HPP_INCLUDED__
#define __ZMQ_MTRIE_HPP_INCLUDED__

// ans ignore: #include "generic_mtrie.hpp"

#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER > 1600)
#define ZMQ_HAS_EXTERN_TEMPLATE 1
#else
#define ZMQ_HAS_EXTERN_TEMPLATE 0
#endif

namespace zmq
{
class pipe_t;

#if ZMQ_HAS_EXTERN_TEMPLATE
extern template class generic_mtrie_t<pipe_t>;
#endif

typedef generic_mtrie_t<pipe_t> mtrie_t;
}

#endif


//========= end of #include "mtrie.hpp" ============


//========= begin of #include "v2_decoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_V2_DECODER_HPP_INCLUDED__
#define __ZMQ_V2_DECODER_HPP_INCLUDED__

// ans ignore: #include "decoder.hpp"
// ans ignore: #include "decoder_allocators.hpp"

namespace zmq
{
//  Decoder for ZMTP/2.x framing protocol. Converts data stream into messages.
//  The class has to inherit from shared_message_memory_allocator because
//  the base class calls allocate in its constructor.
class v2_decoder_t
    : public decoder_base_t<v2_decoder_t, shared_message_memory_allocator>
{
  public:
    v2_decoder_t (size_t bufsize_, int64_t maxmsgsize_, bool zero_copy_);
    virtual ~v2_decoder_t ();

    //  i_decoder interface.
    virtual msg_t *msg () { return &_in_progress; }

  private:
    int flags_ready (unsigned char const *);
    int one_byte_size_ready (unsigned char const *);
    int eight_byte_size_ready (unsigned char const *);
    int message_ready (unsigned char const *);

    int size_ready (uint64_t size_, unsigned char const *);

    unsigned char _tmpbuf[8];
    unsigned char _msg_flags;
    msg_t _in_progress;

    const bool _zero_copy;
    const int64_t _max_msg_size;

    v2_decoder_t (const v2_decoder_t &);
    void operator= (const v2_decoder_t &);
};
}

#endif


//========= end of #include "v2_decoder.hpp" ============


//========= begin of #include "v2_encoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_V2_ENCODER_HPP_INCLUDED__
#define __ZMQ_V2_ENCODER_HPP_INCLUDED__

// ans ignore: #include "encoder.hpp"

namespace zmq
{
//  Encoder for 0MQ framing protocol. Converts messages into data stream.

class v2_encoder_t : public encoder_base_t<v2_encoder_t>
{
  public:
    v2_encoder_t (size_t bufsize_);
    virtual ~v2_encoder_t ();

  private:
    void size_ready ();
    void message_ready ();

    unsigned char _tmp_buf[9];

    v2_encoder_t (const v2_encoder_t &);
    const v2_encoder_t &operator= (const v2_encoder_t &);
};
}

#endif


//========= end of #include "v2_encoder.hpp" ============


//========= begin of #include "norm_engine.hpp" ============


#ifndef __ZMQ_NORM_ENGINE_HPP_INCLUDED__
#define __ZMQ_NORM_ENGINE_HPP_INCLUDED__

#if defined ZMQ_HAVE_NORM

// ans ignore: #include "io_object.hpp"
// ans ignore: #include "i_engine.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "v2_decoder.hpp"
// ans ignore: #include "v2_encoder.hpp"

#include <normApi.h>

namespace zmq
{
class io_thread_t;
class msg_t;
class session_base_t;

class norm_engine_t : public io_object_t, public i_engine
{
  public:
    norm_engine_t (zmq::io_thread_t *parent_, const options_t &options_);
    ~norm_engine_t ();

    // create NORM instance, session, etc
    int init (const char *network_, bool send, bool recv);
    void shutdown ();

    //  i_engine interface implementation.
    //  Plug the engine to the session.
    virtual void plug (zmq::io_thread_t *io_thread_,
                       class session_base_t *session_);

    //  Terminate and deallocate the engine. Note that 'detached'
    //  events are not fired on termination.
    virtual void terminate ();

    //  This method is called by the session to signalise that more
    //  messages can be written to the pipe.
    virtual bool restart_input ();

    //  This method is called by the session to signalise that there
    //  are messages to send available.
    virtual void restart_output ();

    virtual void zap_msg_available (){};

    virtual const endpoint_uri_pair_t &get_endpoint () const;

    // i_poll_events interface implementation.
    // (we only need in_event() for NormEvent notification)
    // (i.e., don't have any output events or timers (yet))
    void in_event ();

  private:
    void unplug ();
    void send_data ();
    void recv_data (NormObjectHandle stream);


    enum
    {
        BUFFER_SIZE = 2048
    };

    // Used to keep track of streams from multiple senders
    class NormRxStreamState
    {
      public:
        NormRxStreamState (NormObjectHandle normStream,
                           int64_t maxMsgSize,
                           bool zeroCopy,
                           int inBatchSize);
        ~NormRxStreamState ();

        NormObjectHandle GetStreamHandle () const { return norm_stream; }

        bool Init ();

        void SetRxReady (bool state) { rx_ready = state; }
        bool IsRxReady () const { return rx_ready; }

        void SetSync (bool state) { in_sync = state; }
        bool InSync () const { return in_sync; }

        // These are used to feed data to decoder
        // and its underlying "msg" buffer
        char *AccessBuffer () { return (char *) (buffer_ptr + buffer_count); }
        size_t GetBytesNeeded () const { return buffer_size - buffer_count; }
        void IncrementBufferCount (size_t count) { buffer_count += count; }
        msg_t *AccessMsg () { return zmq_decoder->msg (); }
        // This invokes the decoder "decode" method
        // returning 0 if more data is needed,
        // 1 if the message is complete, If an error
        // occurs the 'sync' is dropped and the
        // decoder re-initialized
        int Decode ();

        class List
        {
          public:
            List ();
            ~List ();

            void Append (NormRxStreamState &item);
            void Remove (NormRxStreamState &item);

            bool IsEmpty () const { return NULL == head; }

            void Destroy ();

            class Iterator
            {
              public:
                Iterator (const List &list);
                NormRxStreamState *GetNextItem ();

              private:
                NormRxStreamState *next_item;
            };
            friend class Iterator;

          private:
            NormRxStreamState *head;
            NormRxStreamState *tail;

        }; // end class zmq::norm_engine_t::NormRxStreamState::List

        friend class List;

        List *AccessList () { return list; }


      private:
        NormObjectHandle norm_stream;
        int64_t max_msg_size;
        bool zero_copy;
        int in_batch_size;
        bool in_sync;
        bool rx_ready;
        v2_decoder_t *zmq_decoder;
        bool skip_norm_sync;
        unsigned char *buffer_ptr;
        size_t buffer_size;
        size_t buffer_count;

        NormRxStreamState *prev;
        NormRxStreamState *next;
        NormRxStreamState::List *list;

    }; // end class zmq::norm_engine_t::NormRxStreamState

    const endpoint_uri_pair_t _empty_endpoint;

    session_base_t *zmq_session;
    options_t options;
    NormInstanceHandle norm_instance;
    handle_t norm_descriptor_handle;
    NormSessionHandle norm_session;
    bool is_sender;
    bool is_receiver;
    // Sender state
    msg_t tx_msg;
    v2_encoder_t zmq_encoder; // for tx messages (we use v2 for now)
    NormObjectHandle norm_tx_stream;
    bool tx_first_msg;
    bool tx_more_bit;
    bool zmq_output_ready; // zmq has msg(s) to send
    bool norm_tx_ready;    // norm has tx queue vacancy
    // TBD - maybe don't need buffer if can access zmq message buffer directly?
    char tx_buffer[BUFFER_SIZE];
    unsigned int tx_index;
    unsigned int tx_len;

    // Receiver state
    // Lists of norm rx streams from remote senders
    bool zmq_input_ready; // zmq ready to receive msg(s)
    NormRxStreamState::List
      rx_pending_list; // rx streams waiting for data reception
    NormRxStreamState::List
      rx_ready_list; // rx streams ready for NormStreamRead()
    NormRxStreamState::List
      msg_ready_list; // rx streams w/ msg ready for push to zmq


}; // end class norm_engine_t
}

#endif // ZMQ_HAVE_NORM

#endif // !__ZMQ_NORM_ENGINE_HPP_INCLUDED__


//========= end of #include "norm_engine.hpp" ============


//========= begin of #include "v2_protocol.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_V2_PROTOCOL_HPP_INCLUDED__
#define __ZMQ_V2_PROTOCOL_HPP_INCLUDED__

namespace zmq
{
//  Definition of constants for ZMTP/2.0 transport protocol.
class v2_protocol_t
{
  public:
    //  Message flags.
    enum
    {
        more_flag = 1,
        large_flag = 2,
        command_flag = 4
    };
};
}

#endif


//========= end of #include "v2_protocol.hpp" ============


//========= begin of #include "null_mechanism.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_NULL_MECHANISM_HPP_INCLUDED__
#define __ZMQ_NULL_MECHANISM_HPP_INCLUDED__

// ans ignore: #include "mechanism.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "zap_client.hpp"

namespace zmq
{
class msg_t;
class session_base_t;

class null_mechanism_t : public zap_client_t
{
  public:
    null_mechanism_t (session_base_t *session_,
                      const std::string &peer_address_,
                      const options_t &options_);
    virtual ~null_mechanism_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);
    virtual int zap_msg_available ();
    virtual status_t status () const;

  private:
    bool _ready_command_sent;
    bool _error_command_sent;
    bool _ready_command_received;
    bool _error_command_received;
    bool _zap_request_sent;
    bool _zap_reply_received;

    int process_ready_command (const unsigned char *cmd_data_,
                               size_t data_size_);
    int process_error_command (const unsigned char *cmd_data_,
                               size_t data_size_);

    void send_zap_request ();
};
}

#endif


//========= end of #include "null_mechanism.hpp" ============


//========= begin of #include "pair.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PAIR_HPP_INCLUDED__
#define __ZMQ_PAIR_HPP_INCLUDED__

// ans ignore: #include "blob.hpp"
// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;
class io_thread_t;

class pair_t : public socket_base_t
{
  public:
    pair_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~pair_t ();

    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    zmq::pipe_t *_pipe;

    zmq::pipe_t *_last_in;

    pair_t (const pair_t &);
    const pair_t &operator= (const pair_t &);
};
}

#endif


//========= end of #include "pair.hpp" ============


//========= begin of #include "v1_decoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_V1_DECODER_HPP_INCLUDED__
#define __ZMQ_V1_DECODER_HPP_INCLUDED__

// ans ignore: #include "decoder.hpp"

namespace zmq
{
//  Decoder for ZMTP/1.0 protocol. Converts data batches into messages.

class v1_decoder_t : public decoder_base_t<v1_decoder_t>
{
  public:
    v1_decoder_t (size_t bufsize_, int64_t maxmsgsize_);
    ~v1_decoder_t ();

    virtual msg_t *msg () { return &_in_progress; }

  private:
    int one_byte_size_ready (unsigned char const *);
    int eight_byte_size_ready (unsigned char const *);
    int flags_ready (unsigned char const *);
    int message_ready (unsigned char const *);

    unsigned char _tmpbuf[8];
    msg_t _in_progress;

    const int64_t _max_msg_size;

    v1_decoder_t (const v1_decoder_t &);
    void operator= (const v1_decoder_t &);
};
}

#endif


//========= end of #include "v1_decoder.hpp" ============


//========= begin of #include "pgm_socket.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __PGM_SOCKET_HPP_INCLUDED__
#define __PGM_SOCKET_HPP_INCLUDED__

#if defined ZMQ_HAVE_OPENPGM

#ifdef ZMQ_HAVE_WINDOWS
#define __PGM_WININT_H__
#endif

#include <pgm/pgm.h>

#if defined(ZMQ_HAVE_OSX) || defined(ZMQ_HAVE_NETBSD)
#include <pgm/in.h>
#endif

// ans ignore: #include "fd.hpp"
// ans ignore: #include "options.hpp"

namespace zmq
{
//  Encapsulates PGM socket.
class pgm_socket_t
{
  public:
    //  If receiver_ is true PGM transport is not generating SPM packets.
    pgm_socket_t (bool receiver_, const options_t &options_);

    //  Closes the transport.
    ~pgm_socket_t ();

    //  Initialize PGM network structures (GSI, GSRs).
    int init (bool udp_encapsulation_, const char *network_);

    //  Resolve PGM socket address.
    static int init_address (const char *network_,
                             struct pgm_addrinfo_t **addr,
                             uint16_t *port_number);

    //   Get receiver fds and store them into user allocated memory.
    void get_receiver_fds (fd_t *receive_fd_, fd_t *waiting_pipe_fd_);

    //   Get sender and receiver fds and store it to user allocated
    //   memory. Receive fd is used to process NAKs from peers.
    void get_sender_fds (fd_t *send_fd_,
                         fd_t *receive_fd_,
                         fd_t *rdata_notify_fd_,
                         fd_t *pending_notify_fd_);

    //  Send data as one APDU, transmit window owned memory.
    size_t send (unsigned char *data_, size_t data_len_);

    //  Returns max tsdu size without fragmentation.
    size_t get_max_tsdu_size ();

    //  Receive data from pgm socket.
    ssize_t receive (void **data_, const pgm_tsi_t **tsi_);

    long get_rx_timeout ();
    long get_tx_timeout ();

    //  POLLIN on sender side should mean NAK or SPMR receiving.
    //  process_upstream function is used to handle such a situation.
    void process_upstream ();

  private:
    //  Compute size of the buffer based on rate and recovery interval.
    int compute_sqns (int tpdu_);

    //  OpenPGM transport.
    pgm_sock_t *sock;

    int last_rx_status, last_tx_status;

    //  Associated socket options.
    options_t options;

    //  true when pgm_socket should create receiving side.
    bool receiver;

    //  Array of pgm_msgv_t structures to store received data
    //  from the socket (pgm_transport_recvmsgv).
    pgm_msgv_t *pgm_msgv;

    //  Size of pgm_msgv array.
    size_t pgm_msgv_len;

    // How many bytes were read from pgm socket.
    size_t nbytes_rec;

    //  How many bytes were processed from last pgm socket read.
    size_t nbytes_processed;

    //  How many messages from pgm_msgv were already sent up.
    size_t pgm_msgv_processed;
};
}
#endif

#endif


//========= end of #include "pgm_socket.hpp" ============


//========= begin of #include "pgm_receiver.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PGM_RECEIVER_HPP_INCLUDED__
#define __ZMQ_PGM_RECEIVER_HPP_INCLUDED__

#if defined ZMQ_HAVE_OPENPGM

#include <map>
#include <algorithm>

// ans ignore: #include "io_object.hpp"
// ans ignore: #include "i_engine.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "v1_decoder.hpp"
// ans ignore: #include "pgm_socket.hpp"

namespace zmq
{
class io_thread_t;
class session_base_t;

class pgm_receiver_t : public io_object_t, public i_engine
{
  public:
    pgm_receiver_t (zmq::io_thread_t *parent_, const options_t &options_);
    ~pgm_receiver_t ();

    int init (bool udp_encapsulation_, const char *network_);

    //  i_engine interface implementation.
    void plug (zmq::io_thread_t *io_thread_, zmq::session_base_t *session_);
    void terminate ();
    bool restart_input ();
    void restart_output ();
    void zap_msg_available () {}
    const endpoint_uri_pair_t &get_endpoint () const;

    //  i_poll_events interface implementation.
    void in_event ();
    void timer_event (int token);

  private:
    //  Unplug the engine from the session.
    void unplug ();

    //  Decode received data (inpos, insize) and forward decoded
    //  messages to the session.
    int process_input (v1_decoder_t *decoder);

    //  PGM is not able to move subscriptions upstream. Thus, drop all
    //  the pending subscriptions.
    void drop_subscriptions ();

    //  RX timeout timer ID.
    enum
    {
        rx_timer_id = 0xa1
    };

    const endpoint_uri_pair_t _empty_endpoint;

    //  RX timer is running.
    bool has_rx_timer;

    //  If joined is true we are already getting messages from the peer.
    //  It it's false, we are getting data but still we haven't seen
    //  beginning of a message.
    struct peer_info_t
    {
        bool joined;
        v1_decoder_t *decoder;
    };

    struct tsi_comp
    {
        bool operator() (const pgm_tsi_t &ltsi, const pgm_tsi_t &rtsi) const
        {
            uint32_t ll[2], rl[2];
            memcpy (ll, &ltsi, sizeof (ll));
            memcpy (rl, &rtsi, sizeof (rl));
            return (ll[0] < rl[0]) || (ll[0] == rl[0] && ll[1] < rl[1]);
        }
    };

    typedef std::map<pgm_tsi_t, peer_info_t, tsi_comp> peers_t;
    peers_t peers;

    //  PGM socket.
    pgm_socket_t pgm_socket;

    //  Socket options.
    options_t options;

    //  Associated session.
    zmq::session_base_t *session;

    const pgm_tsi_t *active_tsi;

    //  Number of bytes not consumed by the decoder due to pipe overflow.
    size_t insize;

    //  Pointer to data still waiting to be processed by the decoder.
    const unsigned char *inpos;

    //  Poll handle associated with PGM socket.
    handle_t socket_handle;

    //  Poll handle associated with engine PGM waiting pipe.
    handle_t pipe_handle;

    pgm_receiver_t (const pgm_receiver_t &);
    const pgm_receiver_t &operator= (const pgm_receiver_t &);
};
}

#endif

#endif


//========= end of #include "pgm_receiver.hpp" ============


//========= begin of #include "v1_encoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_V1_ENCODER_HPP_INCLUDED__
#define __ZMQ_V1_ENCODER_HPP_INCLUDED__

// ans ignore: #include "encoder.hpp"

namespace zmq
{
//  Encoder for ZMTP/1.0 protocol. Converts messages into data batches.

class v1_encoder_t : public encoder_base_t<v1_encoder_t>
{
  public:
    v1_encoder_t (size_t bufsize_);
    ~v1_encoder_t ();

  private:
    void size_ready ();
    void message_ready ();

    unsigned char _tmpbuf[10];

    v1_encoder_t (const v1_encoder_t &);
    const v1_encoder_t &operator= (const v1_encoder_t &);
};
}

#endif


//========= end of #include "v1_encoder.hpp" ============


//========= begin of #include "pgm_sender.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PGM_SENDER_HPP_INCLUDED__
#define __ZMQ_PGM_SENDER_HPP_INCLUDED__

#if defined ZMQ_HAVE_OPENPGM

// ans ignore: #include "stdint.hpp"
// ans ignore: #include "io_object.hpp"
// ans ignore: #include "i_engine.hpp"
// ans ignore: #include "options.hpp"
// ans ignore: #include "pgm_socket.hpp"
// ans ignore: #include "v1_encoder.hpp"
// ans ignore: #include "msg.hpp"

namespace zmq
{
class io_thread_t;
class session_base_t;

class pgm_sender_t : public io_object_t, public i_engine
{
  public:
    pgm_sender_t (zmq::io_thread_t *parent_, const options_t &options_);
    ~pgm_sender_t ();

    int init (bool udp_encapsulation_, const char *network_);

    //  i_engine interface implementation.
    void plug (zmq::io_thread_t *io_thread_, zmq::session_base_t *session_);
    void terminate ();
    bool restart_input ();
    void restart_output ();
    void zap_msg_available () {}
    const endpoint_uri_pair_t &get_endpoint () const;

    //  i_poll_events interface implementation.
    void in_event ();
    void out_event ();
    void timer_event (int token);

  private:
    //  Unplug the engine from the session.
    void unplug ();

    //  TX and RX timeout timer ID's.
    enum
    {
        tx_timer_id = 0xa0,
        rx_timer_id = 0xa1
    };

    const endpoint_uri_pair_t _empty_endpoint;

    //  Timers are running.
    bool has_tx_timer;
    bool has_rx_timer;

    session_base_t *session;

    //  Message encoder.
    v1_encoder_t encoder;

    msg_t msg;

    //  Keeps track of message boundaries.
    bool more_flag;

    //  PGM socket.
    pgm_socket_t pgm_socket;

    //  Socket options.
    options_t options;

    //  Poll handle associated with PGM socket.
    handle_t handle;
    handle_t uplink_handle;
    handle_t rdata_notify_handle;
    handle_t pending_notify_handle;

    //  Output buffer from pgm_socket.
    unsigned char *out_buffer;

    //  Output buffer size.
    size_t out_buffer_size;

    //  Number of bytes in the buffer to be written to the socket.
    //  If zero, there are no data to be sent.
    size_t write_size;

    pgm_sender_t (const pgm_sender_t &);
    const pgm_sender_t &operator= (const pgm_sender_t &);
};
}
#endif

#endif


//========= end of #include "pgm_sender.hpp" ============


//========= begin of #include "ypipe_conflate.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_YPIPE_CONFLATE_HPP_INCLUDED__
#define __ZMQ_YPIPE_CONFLATE_HPP_INCLUDED__

// ans ignore: #include "platform.hpp"
// ans ignore: #include "dbuffer.hpp"
// ans ignore: #include "ypipe_base.hpp"

namespace zmq
{
//  Adapter for dbuffer, to plug it in instead of a queue for the sake
//  of implementing the conflate socket option, which, if set, makes
//  the receiving side to discard all incoming messages but the last one.
//
//  reader_awake flag is needed here to mimic ypipe delicate behaviour
//  around the reader being asleep (see 'c' pointer being NULL in ypipe.hpp)

template <typename T> class ypipe_conflate_t : public ypipe_base_t<T>
{
  public:
    //  Initialises the pipe.
    inline ypipe_conflate_t () : reader_awake (false) {}

    //  The destructor doesn't have to be virtual. It is made virtual
    //  just to keep ICC and code checking tools from complaining.
    inline virtual ~ypipe_conflate_t () {}

    //  Following function (write) deliberately copies uninitialised data
    //  when used with zmq_msg. Initialising the VSM body for
    //  non-VSM messages won't be good for performance.

#ifdef ZMQ_HAVE_OPENVMS
#pragma message save
#pragma message disable(UNINIT)
#endif
    inline void write (const T &value_, bool incomplete_)
    {
        (void) incomplete_;

        dbuffer.write (value_);
    }

#ifdef ZMQ_HAVE_OPENVMS
#pragma message restore
#endif

    // There are no incomplete items for conflate ypipe
    inline bool unwrite (T *) { return false; }

    //  Flush is no-op for conflate ypipe. Reader asleep behaviour
    //  is as of the usual ypipe.
    //  Returns false if the reader thread is sleeping. In that case,
    //  caller is obliged to wake the reader up before using the pipe again.
    inline bool flush () { return reader_awake; }

    //  Check whether item is available for reading.
    inline bool check_read ()
    {
        bool res = dbuffer.check_read ();
        if (!res)
            reader_awake = false;

        return res;
    }

    //  Reads an item from the pipe. Returns false if there is no value.
    //  available.
    inline bool read (T *value_)
    {
        if (!check_read ())
            return false;

        return dbuffer.read (value_);
    }

    //  Applies the function fn to the first elemenent in the pipe
    //  and returns the value returned by the fn.
    //  The pipe mustn't be empty or the function crashes.
    inline bool probe (bool (*fn_) (const T &)) { return dbuffer.probe (fn_); }

  protected:
    dbuffer_t<T> dbuffer;
    bool reader_awake;

    //  Disable copying of ypipe object.
    ypipe_conflate_t (const ypipe_conflate_t &);
    const ypipe_conflate_t &operator= (const ypipe_conflate_t &);
};
}

#endif


//========= end of #include "ypipe_conflate.hpp" ============


//========= begin of #include "plain_client.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PLAIN_CLIENT_HPP_INCLUDED__
#define __ZMQ_PLAIN_CLIENT_HPP_INCLUDED__

// ans ignore: #include "mechanism_base.hpp"
// ans ignore: #include "options.hpp"

namespace zmq
{
class msg_t;

class plain_client_t : public mechanism_base_t
{
  public:
    plain_client_t (session_base_t *const session_, const options_t &options_);
    virtual ~plain_client_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);
    virtual status_t status () const;

  private:
    enum state_t
    {
        sending_hello,
        waiting_for_welcome,
        sending_initiate,
        waiting_for_ready,
        error_command_received,
        ready
    };

    state_t _state;

    void produce_hello (msg_t *msg_) const;
    void produce_initiate (msg_t *msg_) const;

    int process_welcome (const unsigned char *cmd_data_, size_t data_size_);
    int process_ready (const unsigned char *cmd_data_, size_t data_size_);
    int process_error (const unsigned char *cmd_data_, size_t data_size_);
};
}

#endif


//========= end of #include "plain_client.hpp" ============


//========= begin of #include "plain_common.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PLAIN_COMMON_HPP_INCLUDED__
#define __ZMQ_PLAIN_COMMON_HPP_INCLUDED__

namespace zmq
{
const char hello_prefix[] = "\x05HELLO";
const size_t hello_prefix_len = sizeof (hello_prefix) - 1;

const char welcome_prefix[] = "\x07WELCOME";
const size_t welcome_prefix_len = sizeof (welcome_prefix) - 1;

const char initiate_prefix[] = "\x08INITIATE";
const size_t initiate_prefix_len = sizeof (initiate_prefix) - 1;

const char ready_prefix[] = "\x05READY";
const size_t ready_prefix_len = sizeof (ready_prefix) - 1;

const char error_prefix[] = "\x05ERROR";
const size_t error_prefix_len = sizeof (error_prefix) - 1;

const size_t brief_len_size = sizeof (char);
}

#endif


//========= end of #include "plain_common.hpp" ============


//========= begin of #include "plain_server.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PLAIN_SERVER_HPP_INCLUDED__
#define __ZMQ_PLAIN_SERVER_HPP_INCLUDED__

// ans ignore: #include "options.hpp"
// ans ignore: #include "zap_client.hpp"

namespace zmq
{
class msg_t;
class session_base_t;

class plain_server_t : public zap_client_common_handshake_t
{
  public:
    plain_server_t (session_base_t *session_,
                    const std::string &peer_address_,
                    const options_t &options_);
    virtual ~plain_server_t ();

    // mechanism implementation
    virtual int next_handshake_command (msg_t *msg_);
    virtual int process_handshake_command (msg_t *msg_);

  private:
    void produce_welcome (msg_t *msg_) const;
    void produce_ready (msg_t *msg_) const;
    void produce_error (msg_t *msg_) const;

    int process_hello (msg_t *msg_);
    int process_initiate (msg_t *msg_);

    void send_zap_request (const std::string &username_,
                           const std::string &password_);
};
}

#endif


//========= end of #include "plain_server.hpp" ============


//========= begin of #include "polling_util.hpp" ============

/*
    Copyright (c) 2007-2018 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SOCKET_POLLING_UTIL_HPP_INCLUDED__
#define __ZMQ_SOCKET_POLLING_UTIL_HPP_INCLUDED__

#include <stdlib.h>
#include <vector>

// ans ignore: #include "stdint.hpp"
// ans ignore: #include "platform.hpp"
// ans ignore: #include "err.hpp"

namespace zmq
{
template <typename T, size_t S> class fast_vector_t
{
  public:
    explicit fast_vector_t (const size_t nitems_)
    {
        if (nitems_ > S) {
            _buf = static_cast<T *> (malloc (nitems_ * sizeof (T)));
            //  TODO since this function is called by a client, we could return errno == ENOMEM here
            alloc_assert (_buf);
        } else {
            _buf = _static_buf;
        }
    }

    T &operator[] (const size_t i) { return _buf[i]; }

    ~fast_vector_t ()
    {
        if (_buf != _static_buf)
            free (_buf);
    }

  private:
    fast_vector_t (const fast_vector_t &);
    fast_vector_t &operator= (const fast_vector_t &);

    T _static_buf[S];
    T *_buf;
};

template <typename T, size_t S> class resizable_fast_vector_t
{
  public:
    resizable_fast_vector_t () : _dynamic_buf (NULL) {}

    void resize (const size_t nitems_)
    {
        if (_dynamic_buf)
            _dynamic_buf->resize (nitems_);
        if (nitems_ > S) {
            _dynamic_buf = new (std::nothrow) std::vector<T>;
            //  TODO since this function is called by a client, we could return errno == ENOMEM here
            alloc_assert (_dynamic_buf);
        }
    }

    T *get_buf ()
    {
        // e.g. MSVC 2008 does not have std::vector::data, so we use &...[0]
        return _dynamic_buf ? &(*_dynamic_buf)[0] : _static_buf;
    }

    T &operator[] (const size_t i) { return get_buf ()[i]; }

    ~resizable_fast_vector_t () { delete _dynamic_buf; }

  private:
    resizable_fast_vector_t (const resizable_fast_vector_t &);
    resizable_fast_vector_t &operator= (const resizable_fast_vector_t &);

    T _static_buf[S];
    std::vector<T> *_dynamic_buf;
};

#if defined ZMQ_POLL_BASED_ON_POLL
typedef int timeout_t;

timeout_t compute_timeout (const bool first_pass_,
                           const long timeout_,
                           const uint64_t now_,
                           const uint64_t end_);

#elif defined ZMQ_POLL_BASED_ON_SELECT
inline size_t valid_pollset_bytes (const fd_set &pollset_)
{
#if defined ZMQ_HAVE_WINDOWS
    // On Windows we don't need to copy the whole fd_set.
    // SOCKETS are continuous from the beginning of fd_array in fd_set.
    // We just need to copy fd_count elements of fd_array.
    // We gain huge memcpy() improvement if number of used SOCKETs is much lower than FD_SETSIZE.
    return reinterpret_cast<const char *> (
             &pollset_.fd_array[pollset_.fd_count])
           - reinterpret_cast<const char *> (&pollset_);
#else
    return sizeof (fd_set);
#endif
}

#if defined ZMQ_HAVE_WINDOWS
// struct fd_set {
//  u_int   fd_count;
//  SOCKET  fd_array[1];
// };
// NOTE: offsetof(fd_set, fd_array)==sizeof(SOCKET) on both x86 and x64
//       due to alignment bytes for the latter.
class optimized_fd_set_t
{
  public:
    explicit optimized_fd_set_t (size_t nevents_) : _fd_set (1 + nevents_) {}

    fd_set *get () { return reinterpret_cast<fd_set *> (&_fd_set[0]); }

  private:
    fast_vector_t<SOCKET, 1 + ZMQ_POLLITEMS_DFLT> _fd_set;
};

class resizable_optimized_fd_set_t
{
  public:
    void resize (size_t nevents_) { _fd_set.resize (1 + nevents_); }

    fd_set *get () { return reinterpret_cast<fd_set *> (&_fd_set[0]); }

  private:
    resizable_fast_vector_t<SOCKET, 1 + ZMQ_POLLITEMS_DFLT> _fd_set;
};
#else
class optimized_fd_set_t
{
  public:
    explicit optimized_fd_set_t (size_t /*nevents_*/) {}

    fd_set *get () { return &_fd_set; }

  private:
    fd_set _fd_set;
};

class resizable_optimized_fd_set_t : public optimized_fd_set_t
{
  public:
    resizable_optimized_fd_set_t () : optimized_fd_set_t (0) {}

    void resize (size_t /*nevents_*/) {}
};
#endif
#endif
}

#endif


//========= end of #include "polling_util.hpp" ============


//========= begin of #include "proxy.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PROXY_HPP_INCLUDED__
#define __ZMQ_PROXY_HPP_INCLUDED__

namespace zmq
{
int proxy (class socket_base_t *frontend_,
           class socket_base_t *backend_,
           class socket_base_t *capture_,
           class socket_base_t *control_ =
             NULL); // backward compatibility without this argument
}

#endif


//========= end of #include "proxy.hpp" ============


//========= begin of #include "socket_poller.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SOCKET_POLLER_HPP_INCLUDED__
#define __ZMQ_SOCKET_POLLER_HPP_INCLUDED__

// ans ignore: #include "poller.hpp"

#if defined ZMQ_POLL_BASED_ON_POLL && !defined ZMQ_HAVE_WINDOWS
#include <poll.h>
#endif

#if defined ZMQ_HAVE_WINDOWS
// ans ignore: #include "windows.hpp"
#elif defined ZMQ_HAVE_VXWORKS
#include <unistd.h>
#include <sys/time.h>
#include <strings.h>
#else
#include <unistd.h>
#endif

#include <vector>

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "signaler.hpp"
// ans ignore: #include "polling_util.hpp"

namespace zmq
{
class socket_poller_t
{
  public:
    socket_poller_t ();
    ~socket_poller_t ();

    typedef struct event_t
    {
        socket_base_t *socket;
        fd_t fd;
        void *user_data;
        short events;
    } event_t;

    int add (socket_base_t *socket_, void *user_data_, short events_);
    int modify (socket_base_t *socket_, short events_);
    int remove (socket_base_t *socket_);

    int add_fd (fd_t fd_, void *user_data_, short events_);
    int modify_fd (fd_t fd_, short events_);
    int remove_fd (fd_t fd_);
    // Returns the signaler's fd if there is one, otherwise errors.
    int signaler_fd (fd_t *fd_);

    int wait (event_t *event_, int n_events_, long timeout_);

    inline int size () { return static_cast<int> (_items.size ()); };

    //  Return false if object is not a socket.
    bool check_tag ();

  private:
    void zero_trail_events (zmq::socket_poller_t::event_t *events_,
                            int n_events_,
                            int found_);
#if defined ZMQ_POLL_BASED_ON_POLL
    int check_events (zmq::socket_poller_t::event_t *events_, int n_events_);
#elif defined ZMQ_POLL_BASED_ON_SELECT
    int check_events (zmq::socket_poller_t::event_t *events_,
                      int n_events_,
                      fd_set &inset_,
                      fd_set &outset_,
                      fd_set &errset_);
#endif
    int adjust_timeout (zmq::clock_t &clock_,
                        long timeout_,
                        uint64_t &now_,
                        uint64_t &end_,
                        bool &first_pass_);
    int rebuild ();

    //  Used to check whether the object is a socket_poller.
    uint32_t _tag;

    //  Signaler used for thread safe sockets polling
    signaler_t *_signaler;

    typedef struct item_t
    {
        socket_base_t *socket;
        fd_t fd;
        void *user_data;
        short events;
#if defined ZMQ_POLL_BASED_ON_POLL
        int pollfd_index;
#endif
    } item_t;

    //  List of sockets
    typedef std::vector<item_t> items_t;
    items_t _items;

    //  Does the pollset needs rebuilding?
    bool _need_rebuild;

    //  Should the signaler be used for the thread safe polling?
    bool _use_signaler;

    //  Size of the pollset
    int _pollset_size;

#if defined ZMQ_POLL_BASED_ON_POLL
    pollfd *_pollfds;
#elif defined ZMQ_POLL_BASED_ON_SELECT
    resizable_optimized_fd_set_t _pollset_in;
    resizable_optimized_fd_set_t _pollset_out;
    resizable_optimized_fd_set_t _pollset_err;
    zmq::fd_t _max_fd;
#endif

    socket_poller_t (const socket_poller_t &);
    const socket_poller_t &operator= (const socket_poller_t &);
};
}

#endif


//========= end of #include "socket_poller.hpp" ============


//========= begin of #include "xpub.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_XPUB_HPP_INCLUDED__
#define __ZMQ_XPUB_HPP_INCLUDED__

#include <deque>

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "mtrie.hpp"
// ans ignore: #include "dist.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;
class io_thread_t;

class xpub_t : public socket_base_t
{
  public:
    xpub_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~xpub_t ();

    //  Implementations of virtual functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_ = false,
                       bool locally_initiated_ = false);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Function to be applied to the trie to send all the subscriptions
    //  upstream.
    static void send_unsubscription (zmq::mtrie_t::prefix_t data_,
                                     size_t size_,
                                     xpub_t *self_);

    //  Function to be applied to each matching pipes.
    static void mark_as_matching (zmq::pipe_t *pipe_, xpub_t *arg_);

    //  List of all subscriptions mapped to corresponding pipes.
    mtrie_t _subscriptions;

    //  List of manual subscriptions mapped to corresponding pipes.
    mtrie_t _manual_subscriptions;

    //  Distributor of messages holding the list of outbound pipes.
    dist_t _dist;

    // If true, send all subscription messages upstream, not just
    // unique ones
    bool _verbose_subs;

    // If true, send all unsubscription messages upstream, not just
    // unique ones
    bool _verbose_unsubs;

    //  True if we are in the middle of sending a multi-part message.
    bool _more;

    //  Drop messages if HWM reached, otherwise return with EAGAIN
    bool _lossy;

    //  Subscriptions will not bed added automatically, only after calling set option with ZMQ_SUBSCRIBE or ZMQ_UNSUBSCRIBE
    bool _manual;

    //  Send message to the last pipe, only used if xpub is on manual and after calling set option with ZMQ_SUBSCRIBE
    bool _send_last_pipe;

    //  Function to be applied to match the last pipe.
    static void mark_last_pipe_as_matching (zmq::pipe_t *pipe_, xpub_t *arg_);

    //  Last pipe that sent subscription message, only used if xpub is on manual
    pipe_t *_last_pipe;

    // Pipes that sent subscriptions messages that have not yet been processed, only used if xpub is on manual
    std::deque<pipe_t *> _pending_pipes;

    //  Welcome message to send to pipe when attached
    msg_t _welcome_msg;

    //  List of pending (un)subscriptions, ie. those that were already
    //  applied to the trie, but not yet received by the user.
    std::deque<blob_t> _pending_data;
    std::deque<metadata_t *> _pending_metadata;
    std::deque<unsigned char> _pending_flags;

    xpub_t (const xpub_t &);
    const xpub_t &operator= (const xpub_t &);
};
}

#endif


//========= end of #include "xpub.hpp" ============


//========= begin of #include "pub.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PUB_HPP_INCLUDED__
#define __ZMQ_PUB_HPP_INCLUDED__

// ans ignore: #include "xpub.hpp"

namespace zmq
{
class ctx_t;
class io_thread_t;
class socket_base_t;
class msg_t;

class pub_t : public xpub_t
{
  public:
    pub_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~pub_t ();

    //  Implementations of virtual functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_ = false,
                       bool locally_initiated_ = false);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();

  private:
    pub_t (const pub_t &);
    const pub_t &operator= (const pub_t &);
};
}

#endif


//========= end of #include "pub.hpp" ============


//========= begin of #include "pull.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PULL_HPP_INCLUDED__
#define __ZMQ_PULL_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "fq.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;
class msg_t;
class io_thread_t;

class pull_t : public socket_base_t
{
  public:
    pull_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~pull_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    pull_t (const pull_t &);
    const pull_t &operator= (const pull_t &);
};
}

#endif


//========= end of #include "pull.hpp" ============


//========= begin of #include "push.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_PUSH_HPP_INCLUDED__
#define __ZMQ_PUSH_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "lb.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;
class msg_t;
class io_thread_t;

class push_t : public socket_base_t
{
  public:
    push_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~push_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Load balancer managing the outbound pipes.
    lb_t _lb;

    push_t (const push_t &);
    const push_t &operator= (const push_t &);
};
}

#endif


//========= end of #include "push.hpp" ============


//========= begin of #include "radio.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_RADIO_HPP_INCLUDED__
#define __ZMQ_RADIO_HPP_INCLUDED__

#include <map>
#include <string>
#include <vector>

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "dist.hpp"
// ans ignore: #include "msg.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;
class io_thread_t;

class radio_t : public socket_base_t
{
  public:
    radio_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~radio_t ();

    //  Implementations of virtual functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_ = false,
                       bool locally_initiated_ = false);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  List of all subscriptions mapped to corresponding pipes.
    typedef std::multimap<std::string, pipe_t *> subscriptions_t;
    subscriptions_t _subscriptions;

    //  List of udp pipes
    typedef std::vector<pipe_t *> udp_pipes_t;
    udp_pipes_t _udp_pipes;

    //  Distributor of messages holding the list of outbound pipes.
    dist_t _dist;

    //  Drop messages if HWM reached, otherwise return with EAGAIN
    bool _lossy;

    radio_t (const radio_t &);
    const radio_t &operator= (const radio_t &);
};

class radio_session_t : public session_base_t
{
  public:
    radio_session_t (zmq::io_thread_t *io_thread_,
                     bool connect_,
                     zmq::socket_base_t *socket_,
                     const options_t &options_,
                     address_t *addr_);
    ~radio_session_t ();

    //  Overrides of the functions from session_base_t.
    int push_msg (msg_t *msg_);
    int pull_msg (msg_t *msg_);
    void reset ();

  private:
    enum
    {
        group,
        body
    } _state;

    msg_t _pending_msg;

    radio_session_t (const radio_session_t &);
    const radio_session_t &operator= (const radio_session_t &);
};
}

#endif


//========= end of #include "radio.hpp" ============


//========= begin of #include "radix_tree.hpp" ============

/*
    Copyright (c) 2018 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef RADIX_TREE_HPP
#define RADIX_TREE_HPP

#include <stddef.h>

// ans ignore: #include "stdint.hpp"

// Wrapper type for a node's data layout.
//
// There are 3 32-bit unsigned integers that act as a header. These
// integers represent the following values in this order:
//
// (1) The reference count of the key held by the node. This is 0 if
// the node doesn't hold a key.
//
// (2) The number of characters in the node's prefix. The prefix is a
// part of one or more keys in the tree, e.g. the prefix of each node
// in a trie consists of a single character.
//
// (3) The number of outgoing edges from this node.
//
// The rest of the layout consists of 3 chunks in this order:
//
// (1) The node's prefix as a sequence of one or more bytes. The root
// node always has an empty prefix, unlike other nodes in the tree.
//
// (2) The first byte of the prefix of each of this node's children.
//
// (3) The pointer to each child node.
//
// The link to each child is looked up using its index, e.g. the child
// with index 0 will have its first byte and node pointer at the start
// of the chunk of first bytes and node pointers respectively.
struct node_t
{
    explicit node_t (unsigned char *data_);

    bool operator== (node_t other_) const;
    bool operator!= (node_t other_) const;

    inline uint32_t refcount ();
    inline uint32_t prefix_length ();
    inline uint32_t edgecount ();
    inline unsigned char *prefix ();
    inline unsigned char *first_bytes ();
    inline unsigned char first_byte_at (size_t index_);
    inline unsigned char *node_pointers ();
    inline node_t node_at (size_t index_);
    inline void set_refcount (uint32_t value_);
    inline void set_prefix_length (uint32_t value_);
    inline void set_edgecount (uint32_t value_);
    inline void set_prefix (const unsigned char *prefix_);
    inline void set_first_bytes (const unsigned char *bytes_);
    inline void set_first_byte_at (size_t index_, unsigned char byte_);
    inline void set_node_pointers (const unsigned char *pointers_);
    inline void set_node_at (size_t index_, node_t node_);
    inline void
    set_edge_at (size_t index_, unsigned char first_byte_, node_t node_);
    void resize (size_t prefix_length_, size_t edgecount_);

    unsigned char *_data;
};

node_t make_node (size_t refcount_, size_t prefix_length_, size_t edgecount_);

struct match_result_t
{
    match_result_t (size_t key_bytes_matched_,
                    size_t prefix_bytes_matched_,
                    size_t edge_index_,
                    size_t parent_edge_index_,
                    node_t current_,
                    node_t parent_,
                    node_t grandparent);

    size_t _key_bytes_matched;
    size_t _prefix_bytes_matched;
    size_t _edge_index;
    size_t _parent_edge_index;
    node_t _current_node;
    node_t _parent_node;
    node_t _grandparent_node;
};

namespace zmq
{
class radix_tree_t
{
  public:
    radix_tree_t ();
    ~radix_tree_t ();

    //  Add key to the tree. Returns true if this was a new key rather
    //  than a duplicate.
    bool add (const unsigned char *key_, size_t key_size_);

    //  Remove key from the tree. Returns true if the item is actually
    //  removed from the tree.
    bool rm (const unsigned char *key_, size_t key_size_);

    //  Check whether particular key is in the tree.
    bool check (const unsigned char *key_, size_t key_size_);

    //  Apply the function supplied to each key in the tree.
    void apply (void (*func_) (unsigned char *data, size_t size, void *arg),
                void *arg_);

    size_t size () const;

  private:
    inline match_result_t
    match (const unsigned char *key_, size_t key_size_, bool is_lookup_) const;

    node_t _root;
    size_t _size;
};
}

#endif


//========= end of #include "radix_tree.hpp" ============


//========= begin of #include "raw_decoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_RAW_DECODER_HPP_INCLUDED__
#define __ZMQ_RAW_DECODER_HPP_INCLUDED__

// ans ignore: #include "msg.hpp"
// ans ignore: #include "i_decoder.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "decoder_allocators.hpp"

namespace zmq
{
//  Decoder for 0MQ v1 framing protocol. Converts data stream into messages.

class raw_decoder_t : public i_decoder
{
  public:
    raw_decoder_t (size_t bufsize_);
    virtual ~raw_decoder_t ();

    //  i_decoder interface.

    virtual void get_buffer (unsigned char **data_, size_t *size_);

    virtual int
    decode (const unsigned char *data_, size_t size_, size_t &bytes_used_);

    virtual msg_t *msg () { return &_in_progress; }

    virtual void resize_buffer (size_t) {}

  private:
    msg_t _in_progress;

    shared_message_memory_allocator _allocator;

    raw_decoder_t (const raw_decoder_t &);
    void operator= (const raw_decoder_t &);
};
}

#endif


//========= end of #include "raw_decoder.hpp" ============


//========= begin of #include "raw_encoder.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_RAW_ENCODER_HPP_INCLUDED__
#define __ZMQ_RAW_ENCODER_HPP_INCLUDED__

#include <stddef.h>
#include <string.h>
#include <stdlib.h>

// ans ignore: #include "encoder.hpp"

namespace zmq
{
//  Encoder for 0MQ framing protocol. Converts messages into data batches.

class raw_encoder_t : public encoder_base_t<raw_encoder_t>
{
  public:
    raw_encoder_t (size_t bufsize_);
    ~raw_encoder_t ();

  private:
    void raw_message_ready ();

    raw_encoder_t (const raw_encoder_t &);
    const raw_encoder_t &operator= (const raw_encoder_t &);
};
}

#endif


//========= end of #include "raw_encoder.hpp" ============


//========= begin of #include "router.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_ROUTER_HPP_INCLUDED__
#define __ZMQ_ROUTER_HPP_INCLUDED__

#include <map>

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "blob.hpp"
// ans ignore: #include "msg.hpp"
// ans ignore: #include "fq.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;

//  TODO: This class uses O(n) scheduling. Rewrite it to use O(1) algorithm.
class router_t : public routing_socket_base_t
{
  public:
    router_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~router_t ();

    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);
    int get_peer_state (const void *routing_id_, size_t routing_id_size_) const;

  protected:
    //  Rollback any message parts that were sent but not yet flushed.
    int rollback ();

  private:
    //  Receive peer id and update lookup map
    bool identify_peer (pipe_t *pipe_, bool locally_initiated_);

    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    //  True iff there is a message held in the pre-fetch buffer.
    bool _prefetched;

    //  If true, the receiver got the message part with
    //  the peer's identity.
    bool _routing_id_sent;

    //  Holds the prefetched identity.
    msg_t _prefetched_id;

    //  Holds the prefetched message.
    msg_t _prefetched_msg;

    //  The pipe we are currently reading from
    zmq::pipe_t *_current_in;

    //  Should current_in should be terminate after all parts received?
    bool _terminate_current_in;

    //  If true, more incoming message parts are expected.
    bool _more_in;

    //  We keep a set of pipes that have not been identified yet.
    std::set<pipe_t *> _anonymous_pipes;

    //  The pipe we are currently writing to.
    zmq::pipe_t *_current_out;

    //  If true, more outgoing message parts are expected.
    bool _more_out;

    //  Routing IDs are generated. It's a simple increment and wrap-over
    //  algorithm. This value is the next ID to use (if not used already).
    uint32_t _next_integral_routing_id;

    // If true, report EAGAIN to the caller instead of silently dropping
    // the message targeting an unknown peer.
    bool _mandatory;
    bool _raw_socket;

    // if true, send an empty message to every connected router peer
    bool _probe_router;

    // If true, the router will reassign an identity upon encountering a
    // name collision. The new pipe will take the identity, the old pipe
    // will be terminated.
    bool _handover;

    router_t (const router_t &);
    const router_t &operator= (const router_t &);
};
}

#endif


//========= end of #include "router.hpp" ============


//========= begin of #include "rep.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_REP_HPP_INCLUDED__
#define __ZMQ_REP_HPP_INCLUDED__

// ans ignore: #include "router.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class io_thread_t;
class socket_base_t;

class rep_t : public router_t
{
  public:
    rep_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~rep_t ();

    //  Overrides of functions from socket_base_t.
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();

  private:
    //  If true, we are in process of sending the reply. If false we are
    //  in process of receiving a request.
    bool _sending_reply;

    //  If true, we are starting to receive a request. The beginning
    //  of the request is the backtrace stack.
    bool _request_begins;

    rep_t (const rep_t &);
    const rep_t &operator= (const rep_t &);
};
}

#endif


//========= end of #include "rep.hpp" ============


//========= begin of #include "req.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_REQ_HPP_INCLUDED__
#define __ZMQ_REQ_HPP_INCLUDED__

// ans ignore: #include "dealer.hpp"
// ans ignore: #include "stdint.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class io_thread_t;
class socket_base_t;

class req_t : public dealer_t
{
  public:
    req_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~req_t ();

    //  Overrides of functions from socket_base_t.
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  protected:
    //  Receive only from the pipe the request was sent to, discarding
    //  frames from other pipes.
    int recv_reply_pipe (zmq::msg_t *msg_);

  private:
    //  If true, request was already sent and reply wasn't received yet or
    //  was received partially.
    bool _receiving_reply;

    //  If true, we are starting to send/recv a message. The first part
    //  of the message must be empty message part (backtrace stack bottom).
    bool _message_begins;

    //  The pipe the request was sent to and where the reply is expected.
    zmq::pipe_t *_reply_pipe;

    //  Whether request id frames shall be sent and expected.
    bool _request_id_frames_enabled;

    //  The current request id. It is incremented every time before a new
    //  request is sent.
    uint32_t _request_id;

    //  If false, send() will reset its internal state and terminate the
    //  reply_pipe's connection instead of failing if a previous request is
    //  still pending.
    bool _strict;

    req_t (const req_t &);
    const req_t &operator= (const req_t &);
};

class req_session_t : public session_base_t
{
  public:
    req_session_t (zmq::io_thread_t *io_thread_,
                   bool connect_,
                   zmq::socket_base_t *socket_,
                   const options_t &options_,
                   address_t *addr_);
    ~req_session_t ();

    //  Overrides of the functions from session_base_t.
    int push_msg (msg_t *msg_);
    void reset ();

  private:
    enum
    {
        bottom,
        request_id,
        body
    } _state;

    req_session_t (const req_session_t &);
    const req_session_t &operator= (const req_session_t &);
};
}

#endif


//========= end of #include "req.hpp" ============


//========= begin of #include "scatter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SCATTER_HPP_INCLUDED__
#define __ZMQ_SCATTER_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "lb.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;
class msg_t;
class io_thread_t;

class scatter_t : public socket_base_t
{
  public:
    scatter_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~scatter_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Load balancer managing the outbound pipes.
    lb_t _lb;

    scatter_t (const scatter_t &);
    const scatter_t &operator= (const scatter_t &);
};
}

#endif


//========= end of #include "scatter.hpp" ============


//========= begin of #include "server.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SERVER_HPP_INCLUDED__
#define __ZMQ_SERVER_HPP_INCLUDED__

#include <map>

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "blob.hpp"
// ans ignore: #include "fq.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class pipe_t;

//  TODO: This class uses O(n) scheduling. Rewrite it to use O(1) algorithm.
class server_t : public socket_base_t
{
  public:
    server_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~server_t ();

    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    struct outpipe_t
    {
        zmq::pipe_t *pipe;
        bool active;
    };

    //  Outbound pipes indexed by the peer IDs.
    typedef std::map<uint32_t, outpipe_t> out_pipes_t;
    out_pipes_t _out_pipes;

    //  Routing IDs are generated. It's a simple increment and wrap-over
    //  algorithm. This value is the next ID to use (if not used already).
    uint32_t _next_routing_id;

    server_t (const server_t &);
    const server_t &operator= (const server_t &);
};
}

#endif


//========= end of #include "server.hpp" ============


//========= begin of #include "tcp_connecter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __TCP_CONNECTER_HPP_INCLUDED__
#define __TCP_CONNECTER_HPP_INCLUDED__

// ans ignore: #include "fd.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "stream_connecter_base.hpp"

namespace zmq
{
class tcp_connecter_t : public stream_connecter_base_t
{
  public:
    //  If 'delayed_start' is true connecter first waits for a while,
    //  then starts connection process.
    tcp_connecter_t (zmq::io_thread_t *io_thread_,
                     zmq::session_base_t *session_,
                     const options_t &options_,
                     address_t *addr_,
                     bool delayed_start_);
    ~tcp_connecter_t ();

  private:
    //  ID of the timer used to check the connect timeout, must be different from stream_connecter_base_t::reconnect_timer_id.
    enum
    {
        connect_timer_id = 2
    };

    //  Handlers for incoming commands.
    void process_term (int linger_);

    //  Handlers for I/O events.
    void out_event ();
    void timer_event (int id_);

    //  Internal function to start the actual connection establishment.
    void start_connecting ();

    //  Internal function to add a connect timer
    void add_connect_timer ();

    //  Open TCP connecting socket. Returns -1 in case of error,
    //  0 if connect was successful immediately. Returns -1 with
    //  EAGAIN errno if async connect was launched.
    int open ();

    //  Get the file descriptor of newly created connection. Returns
    //  retired_fd if the connection was unsuccessful.
    fd_t connect ();

    //  Tunes a connected socket.
    bool tune_socket (fd_t fd_);

    //  True iff a timer has been started.
    bool _connect_timer_started;

    tcp_connecter_t (const tcp_connecter_t &);
    const tcp_connecter_t &operator= (const tcp_connecter_t &);
};
}

#endif


//========= end of #include "tcp_connecter.hpp" ============


//========= begin of #include "tipc_connecter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __TIPC_CONNECTER_HPP_INCLUDED__
#define __TIPC_CONNECTER_HPP_INCLUDED__

// ans ignore: #include "platform.hpp"

#if defined ZMQ_HAVE_TIPC

// ans ignore: #include "fd.hpp"
// ans ignore: #include "stream_connecter_base.hpp"

namespace zmq
{
class tipc_connecter_t : public stream_connecter_base_t
{
  public:
    //  If 'delayed_start' is true connecter first waits for a while,
    //  then starts connection process.
    tipc_connecter_t (zmq::io_thread_t *io_thread_,
                      zmq::session_base_t *session_,
                      const options_t &options_,
                      address_t *addr_,
                      bool delayed_start_);

  private:
    //  Handlers for I/O events.
    void out_event ();

    //  Internal function to start the actual connection establishment.
    void start_connecting ();

    //  Get the file descriptor of newly created connection. Returns
    //  retired_fd if the connection was unsuccessful.
    fd_t connect ();

    //  Open IPC connecting socket. Returns -1 in case of error,
    //  0 if connect was successful immediately. Returns -1 with
    //  EAGAIN errno if async connect was launched.
    int open ();

    tipc_connecter_t (const tipc_connecter_t &);
    const tipc_connecter_t &operator= (const tipc_connecter_t &);
};
}

#endif

#endif


//========= end of #include "tipc_connecter.hpp" ============


//========= begin of #include "socks.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SOCKS_HPP_INCLUDED__
#define __ZMQ_SOCKS_HPP_INCLUDED__

#include <string>
// ans ignore: #include "fd.hpp"
// ans ignore: #include "stdint.hpp"

namespace zmq
{
struct socks_greeting_t
{
    socks_greeting_t (uint8_t method_);
    socks_greeting_t (const uint8_t *methods_, uint8_t num_methods_);

    uint8_t methods[UINT8_MAX];
    const size_t num_methods;
};

class socks_greeting_encoder_t
{
  public:
    socks_greeting_encoder_t ();
    void encode (const socks_greeting_t &greeting_);
    int output (fd_t fd_);
    bool has_pending_data () const;
    void reset ();

  private:
    size_t _bytes_encoded;
    size_t _bytes_written;
    uint8_t _buf[2 + UINT8_MAX];
};

struct socks_choice_t
{
    socks_choice_t (uint8_t method_);

    uint8_t method;
};

class socks_choice_decoder_t
{
  public:
    socks_choice_decoder_t ();
    int input (fd_t fd_);
    bool message_ready () const;
    socks_choice_t decode ();
    void reset ();

  private:
    unsigned char _buf[2];
    size_t _bytes_read;
};


struct socks_basic_auth_request_t
{
    socks_basic_auth_request_t (std::string username_, std::string password_);

    const std::string username;
    const std::string password;
};

class socks_basic_auth_request_encoder_t
{
  public:
    socks_basic_auth_request_encoder_t ();
    void encode (const socks_basic_auth_request_t &req_);
    int output (fd_t fd_);
    bool has_pending_data () const;
    void reset ();

  private:
    size_t _bytes_encoded;
    size_t _bytes_written;
    uint8_t _buf[1 + 1 + UINT8_MAX + 1 + UINT8_MAX];
};

struct socks_auth_response_t
{
    socks_auth_response_t (uint8_t response_code_);
    uint8_t response_code;
};

class socks_auth_response_decoder_t
{
  public:
    socks_auth_response_decoder_t ();
    int input (fd_t fd_);
    bool message_ready () const;
    socks_auth_response_t decode ();
    void reset ();

  private:
    int8_t _buf[2];
    size_t _bytes_read;
};

struct socks_request_t
{
    socks_request_t (uint8_t command_, std::string hostname_, uint16_t port_);

    const uint8_t command;
    const std::string hostname;
    const uint16_t port;
};

class socks_request_encoder_t
{
  public:
    socks_request_encoder_t ();
    void encode (const socks_request_t &req_);
    int output (fd_t fd_);
    bool has_pending_data () const;
    void reset ();

  private:
    size_t _bytes_encoded;
    size_t _bytes_written;
    uint8_t _buf[4 + UINT8_MAX + 1 + 2];
};

struct socks_response_t
{
    socks_response_t (uint8_t response_code_,
                      std::string address_,
                      uint16_t port_);
    uint8_t response_code;
    std::string address;
    uint16_t port;
};

class socks_response_decoder_t
{
  public:
    socks_response_decoder_t ();
    int input (fd_t fd_);
    bool message_ready () const;
    socks_response_t decode ();
    void reset ();

  private:
    int8_t _buf[4 + UINT8_MAX + 1 + 2];
    size_t _bytes_read;
};
}

#endif


//========= end of #include "socks.hpp" ============


//========= begin of #include "socks_connecter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __SOCKS_CONNECTER_HPP_INCLUDED__
#define __SOCKS_CONNECTER_HPP_INCLUDED__

// ans ignore: #include "fd.hpp"
// ans ignore: #include "stream_connecter_base.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "socks.hpp"

namespace zmq
{
class io_thread_t;
class session_base_t;
struct address_t;

class socks_connecter_t : public stream_connecter_base_t
{
  public:
    //  If 'delayed_start' is true connecter first waits for a while,
    //  then starts connection process.
    socks_connecter_t (zmq::io_thread_t *io_thread_,
                       zmq::session_base_t *session_,
                       const options_t &options_,
                       address_t *addr_,
                       address_t *proxy_addr_,
                       bool delayed_start_);
    ~socks_connecter_t ();

    void set_auth_method_basic (const std::string username,
                                const std::string password);
    void set_auth_method_none ();


  private:
    enum
    {
        unplugged,
        waiting_for_reconnect_time,
        waiting_for_proxy_connection,
        sending_greeting,
        waiting_for_choice,
        sending_basic_auth_request,
        waiting_for_auth_response,
        sending_request,
        waiting_for_response
    };

    //  Method ID
    enum
    {
        socks_no_auth_required = 0x00,
        socks_basic_auth = 0x02,
        socks_no_acceptable_method = 0xff
    };

    //  Handlers for I/O events.
    virtual void in_event ();
    virtual void out_event ();

    //  Internal function to start the actual connection establishment.
    void start_connecting ();

    int process_server_response (const socks_choice_t &response_);
    int process_server_response (const socks_response_t &response_);
    int process_server_response (const socks_auth_response_t &response_);

    int parse_address (const std::string &address_,
                       std::string &hostname_,
                       uint16_t &port_);

    int connect_to_proxy ();

    void error ();

    //  Open TCP connecting socket. Returns -1 in case of error,
    //  0 if connect was successful immediately. Returns -1 with
    //  EAGAIN errno if async connect was launched.
    int open ();

    //  Get the file descriptor of newly created connection. Returns
    //  retired_fd if the connection was unsuccessful.
    zmq::fd_t check_proxy_connection ();

    socks_greeting_encoder_t _greeting_encoder;
    socks_choice_decoder_t _choice_decoder;
    socks_basic_auth_request_encoder_t _basic_auth_request_encoder;
    socks_auth_response_decoder_t _auth_response_decoder;
    socks_request_encoder_t _request_encoder;
    socks_response_decoder_t _response_decoder;

    //  SOCKS address; owned by this connecter.
    address_t *_proxy_addr;

    // User defined authentication method
    int _auth_method;

    // Credentials for basic authentication
    std::string _auth_username;
    std::string _auth_password;

    int _status;

    socks_connecter_t (const socks_connecter_t &);
    const socks_connecter_t &operator= (const socks_connecter_t &);
};
}

#endif


//========= end of #include "socks_connecter.hpp" ============


//========= begin of #include "vmci_connecter.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_VMCI_CONNECTER_HPP_INCLUDED__
#define __ZMQ_VMCI_CONNECTER_HPP_INCLUDED__

// ans ignore: #include "platform.hpp"

#if defined ZMQ_HAVE_VMCI

// ans ignore: #include "fd.hpp"
// ans ignore: #include "own.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "io_object.hpp"

namespace zmq
{
class io_thread_t;
class session_base_t;
struct address_t;

//  TODO consider refactoring this to derive from stream_connecter_base_t
class vmci_connecter_t : public own_t, public io_object_t
{
  public:
    //  If 'delayed_start' is true connecter first waits for a while,
    //  then starts connection process.
    vmci_connecter_t (zmq::io_thread_t *io_thread_,
                      zmq::session_base_t *session_,
                      const options_t &options_,
                      const address_t *addr_,
                      bool delayed_start_);
    ~vmci_connecter_t ();

  private:
    //  ID of the timer used to delay the reconnection.
    enum
    {
        reconnect_timer_id = 1
    };

    //  Handlers for incoming commands.
    void process_plug ();
    void process_term (int linger_);

    //  Handlers for I/O events.
    void in_event ();
    void out_event ();
    void timer_event (int id_);

    //  Internal function to start the actual connection establishment.
    void start_connecting ();

    //  Internal function to add a reconnect timer
    void add_reconnect_timer ();

    //  Internal function to return a reconnect backoff delay.
    //  Will modify the current_reconnect_ivl used for next call
    //  Returns the currently used interval
    int get_new_reconnect_ivl ();

    //  Open VMCI connecting socket. Returns -1 in case of error,
    //  0 if connect was successful immediately. Returns -1 with
    //  EAGAIN errno if async connect was launched.
    int open ();

    //  Close the connecting socket.
    void close ();

    //  Get the file descriptor of newly created connection. Returns
    //  retired_fd if the connection was unsuccessful.
    fd_t connect ();

    //  Address to connect to. Owned by session_base_t.
    const address_t *addr;

    //  Underlying socket.
    fd_t s;

    //  Handle corresponding to the listening socket.
    handle_t handle;

    //  If true file descriptor is registered with the poller and 'handle'
    //  contains valid value.
    bool handle_valid;

    //  If true, connecter is waiting a while before trying to connect.
    const bool delayed_start;

    //  True iff a timer has been started.
    bool timer_started;

    //  Reference to the session we belong to.
    zmq::session_base_t *session;

    //  Current reconnect ivl, updated for backoff strategy
    int current_reconnect_ivl;

    // String representation of endpoint to connect to
    std::string endpoint;

    // Socket
    zmq::socket_base_t *socket;

    vmci_connecter_t (const vmci_connecter_t &);
    const vmci_connecter_t &operator= (const vmci_connecter_t &);
};
}

#endif

#endif


//========= end of #include "vmci_connecter.hpp" ============


//========= begin of #include "udp_engine.hpp" ============


#ifndef __ZMQ_UDP_ENGINE_HPP_INCLUDED__
#define __ZMQ_UDP_ENGINE_HPP_INCLUDED__

// ans ignore: #include "io_object.hpp"
// ans ignore: #include "i_engine.hpp"
// ans ignore: #include "address.hpp"
// ans ignore: #include "msg.hpp"

#define MAX_UDP_MSG 8192

namespace zmq
{
class io_thread_t;
class session_base_t;

class udp_engine_t : public io_object_t, public i_engine
{
  public:
    udp_engine_t (const options_t &options_);
    ~udp_engine_t ();

    int init (address_t *address_, bool send_, bool recv_);

    //  i_engine interface implementation.
    //  Plug the engine to the session.
    void plug (zmq::io_thread_t *io_thread_, class session_base_t *session_);

    //  Terminate and deallocate the engine. Note that 'detached'
    //  events are not fired on termination.
    void terminate ();

    //  This method is called by the session to signalise that more
    //  messages can be written to the pipe.
    bool restart_input ();

    //  This method is called by the session to signalise that there
    //  are messages to send available.
    void restart_output ();

    void zap_msg_available (){};

    void in_event ();
    void out_event ();

    const endpoint_uri_pair_t &get_endpoint () const;

  private:
    int resolve_raw_address (char *addr_, size_t length_);
    void sockaddr_to_msg (zmq::msg_t *msg_, sockaddr_in *addr_);

    const endpoint_uri_pair_t _empty_endpoint;

    bool _plugged;

    fd_t _fd;
    session_base_t *_session;
    handle_t _handle;
    address_t *_address;

    options_t _options;

    sockaddr_in _raw_address;
    const struct sockaddr *_out_address;
    zmq_socklen_t _out_address_len;

    char _out_buffer[MAX_UDP_MSG];
    char _in_buffer[MAX_UDP_MSG];
    bool _send_enabled;
    bool _recv_enabled;
};
}

#endif


//========= end of #include "udp_engine.hpp" ============


//========= begin of #include "tcp_listener.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TCP_LISTENER_HPP_INCLUDED__
#define __ZMQ_TCP_LISTENER_HPP_INCLUDED__

// ans ignore: #include "fd.hpp"
// ans ignore: #include "tcp_address.hpp"
// ans ignore: #include "stream_listener_base.hpp"

namespace zmq
{
class tcp_listener_t : public stream_listener_base_t
{
  public:
    tcp_listener_t (zmq::io_thread_t *io_thread_,
                    zmq::socket_base_t *socket_,
                    const options_t &options_);

    //  Set address to listen on.
    int set_local_address (const char *addr_);

  protected:
    std::string get_socket_name (fd_t fd_, socket_end_t socket_end_) const;

  private:
    //  Handlers for I/O events.
    void in_event ();

    //  Accept the new connection. Returns the file descriptor of the
    //  newly created connection. The function may return retired_fd
    //  if the connection was dropped while waiting in the listen backlog
    //  or was denied because of accept filters.
    fd_t accept ();

    int create_socket (const char *addr_);

    //  Address to listen on.
    tcp_address_t _address;

    tcp_listener_t (const tcp_listener_t &);
    const tcp_listener_t &operator= (const tcp_listener_t &);
};
}

#endif


//========= end of #include "tcp_listener.hpp" ============


//========= begin of #include "tipc_listener.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TIPC_LISTENER_HPP_INCLUDED__
#define __ZMQ_TIPC_LISTENER_HPP_INCLUDED__

// ans ignore: #include "platform.hpp"

#if defined ZMQ_HAVE_TIPC

#include <string>

// ans ignore: #include "fd.hpp"
// ans ignore: #include "stream_listener_base.hpp"
// ans ignore: #include "tipc_address.hpp"

namespace zmq
{
class tipc_listener_t : public stream_listener_base_t
{
  public:
    tipc_listener_t (zmq::io_thread_t *io_thread_,
                     zmq::socket_base_t *socket_,
                     const options_t &options_);

    //  Set address to listen on.
    int set_local_address (const char *addr_);

  protected:
    std::string get_socket_name (fd_t fd_, socket_end_t socket_end_) const;

  private:
    //  Handlers for I/O events.
    void in_event ();

    //  Accept the new connection. Returns the file descriptor of the
    //  newly created connection. The function may return retired_fd
    //  if the connection was dropped while waiting in the listen backlog.
    fd_t accept ();

    // Address to listen on
    tipc_address_t _address;

    tipc_listener_t (const tipc_listener_t &);
    const tipc_listener_t &operator= (const tipc_listener_t &);
};
}

#endif

#endif


//========= end of #include "tipc_listener.hpp" ============


//========= begin of #include "vmci_listener.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_VMCI_LISTENER_HPP_INCLUDED__
#define __ZMQ_VMCI_LISTENER_HPP_INCLUDED__

// ans ignore: #include "platform.hpp"

#if defined ZMQ_HAVE_VMCI

#include <string>

// ans ignore: #include "fd.hpp"
// ans ignore: #include "own.hpp"
// ans ignore: #include "stdint.hpp"
// ans ignore: #include "io_object.hpp"

namespace zmq
{
class io_thread_t;
class socket_base_t;

//  TODO consider refactoring this to derive from stream_listener_base_t
class vmci_listener_t : public own_t, public io_object_t
{
  public:
    vmci_listener_t (zmq::io_thread_t *io_thread_,
                     zmq::socket_base_t *socket_,
                     const options_t &options_);
    ~vmci_listener_t ();

    //  Set address to listen on.
    int set_local_address (const char *addr_);

    // Get the bound address for use with wildcards
    int get_local_address (std::string &addr_);

  private:
    //  Handlers for incoming commands.
    void process_plug ();
    void process_term (int linger_);

    //  Handlers for I/O events.
    void in_event ();

    //  Close the listening socket.
    void close ();

    //  Accept the new connection. Returns the file descriptor of the
    //  newly created connection. The function may return retired_fd
    //  if the connection was dropped while waiting in the listen backlog.
    fd_t accept ();

    //  Underlying socket.
    fd_t s;

    //  Handle corresponding to the listening socket.
    handle_t handle;

    //  Socket the listerner belongs to.
    zmq::socket_base_t *socket;

    // String representation of endpoint to bind to
    std::string endpoint;

    vmci_listener_t (const vmci_listener_t &);
    const vmci_listener_t &operator= (const vmci_listener_t &);
};
}

#endif

#endif


//========= end of #include "vmci_listener.hpp" ============


//========= begin of #include "trie.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TRIE_HPP_INCLUDED__
#define __ZMQ_TRIE_HPP_INCLUDED__

#include <stddef.h>

// ans ignore: #include "stdint.hpp"

namespace zmq
{
class trie_t
{
  public:
    trie_t ();
    ~trie_t ();

    //  Add key to the trie. Returns true if this is a new item in the trie
    //  rather than a duplicate.
    bool add (unsigned char *prefix_, size_t size_);

    //  Remove key from the trie. Returns true if the item is actually
    //  removed from the trie.
    bool rm (unsigned char *prefix_, size_t size_);

    //  Check whether particular key is in the trie.
    bool check (unsigned char *data_, size_t size_);

    //  Apply the function supplied to each subscription in the trie.
    void apply (void (*func_) (unsigned char *data_, size_t size_, void *arg_),
                void *arg_);

  private:
    void apply_helper (unsigned char **buff_,
                       size_t buffsize_,
                       size_t maxbuffsize_,
                       void (*func_) (unsigned char *data_,
                                      size_t size_,
                                      void *arg_),
                       void *arg_) const;
    bool is_redundant () const;

    uint32_t _refcnt;
    unsigned char _min;
    unsigned short _count;
    unsigned short _live_nodes;
    union
    {
        class trie_t *node;
        class trie_t **table;
    } _next;

    trie_t (const trie_t &);
    const trie_t &operator= (const trie_t &);
};
}

#endif


//========= end of #include "trie.hpp" ============


//========= begin of #include "xsub.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_XSUB_HPP_INCLUDED__
#define __ZMQ_XSUB_HPP_INCLUDED__

// ans ignore: #include "socket_base.hpp"
// ans ignore: #include "session_base.hpp"
// ans ignore: #include "dist.hpp"
// ans ignore: #include "fq.hpp"
#ifdef ZMQ_USE_RADIX_TREE
// ans ignore: #include "radix_tree.hpp"
#else
// ans ignore: #include "trie.hpp"
#endif

namespace zmq
{
class ctx_t;
class pipe_t;
class io_thread_t;

class xsub_t : public socket_base_t
{
  public:
    xsub_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~xsub_t ();

  protected:
    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xwrite_activated (zmq::pipe_t *pipe_);
    void xhiccuped (pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);

  private:
    //  Check whether the message matches at least one subscription.
    bool match (zmq::msg_t *msg_);

    //  Function to be applied to the trie to send all the subsciptions
    //  upstream.
    static void
    send_subscription (unsigned char *data_, size_t size_, void *arg_);

    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    //  Object for distributing the subscriptions upstream.
    dist_t _dist;

    //  The repository of subscriptions.
#ifdef ZMQ_USE_RADIX_TREE
    radix_tree_t _subscriptions;
#else
    trie_t _subscriptions;
#endif

    //  If true, 'message' contains a matching message to return on the
    //  next recv call.
    bool _has_message;
    msg_t _message;

    //  If true, part of a multipart message was already received, but
    //  there are following parts still waiting.
    bool _more;

    xsub_t (const xsub_t &);
    const xsub_t &operator= (const xsub_t &);
};
}

#endif


//========= end of #include "xsub.hpp" ============


//========= begin of #include "sub.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SUB_HPP_INCLUDED__
#define __ZMQ_SUB_HPP_INCLUDED__

// ans ignore: #include "xsub.hpp"

namespace zmq
{
class ctx_t;
class msg_t;
class io_thread_t;
class socket_base_t;

class sub_t : public xsub_t
{
  public:
    sub_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~sub_t ();

  protected:
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);
    int xsend (zmq::msg_t *msg_);
    bool xhas_out ();

  private:
    sub_t (const sub_t &);
    const sub_t &operator= (const sub_t &);
};
}

#endif


//========= end of #include "sub.hpp" ============


//========= begin of #include "stream.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_STREAM_HPP_INCLUDED__
#define __ZMQ_STREAM_HPP_INCLUDED__

#include <map>

// ans ignore: #include "router.hpp"

namespace zmq
{
class ctx_t;
class pipe_t;

class stream_t : public routing_socket_base_t
{
  public:
    stream_t (zmq::ctx_t *parent_, uint32_t tid_, int sid_);
    ~stream_t ();

    //  Overrides of functions from socket_base_t.
    void xattach_pipe (zmq::pipe_t *pipe_,
                       bool subscribe_to_all_,
                       bool locally_initiated_);
    int xsend (zmq::msg_t *msg_);
    int xrecv (zmq::msg_t *msg_);
    bool xhas_in ();
    bool xhas_out ();
    void xread_activated (zmq::pipe_t *pipe_);
    void xpipe_terminated (zmq::pipe_t *pipe_);
    int xsetsockopt (int option_, const void *optval_, size_t optvallen_);

  private:
    //  Generate peer's id and update lookup map
    void identify_peer (pipe_t *pipe_, bool locally_initiated_);

    //  Fair queueing object for inbound pipes.
    fq_t _fq;

    //  True iff there is a message held in the pre-fetch buffer.
    bool _prefetched;

    //  If true, the receiver got the message part with
    //  the peer's identity.
    bool _routing_id_sent;

    //  Holds the prefetched identity.
    msg_t _prefetched_routing_id;

    //  Holds the prefetched message.
    msg_t _prefetched_msg;

    //  The pipe we are currently writing to.
    zmq::pipe_t *_current_out;

    //  If true, more outgoing message parts are expected.
    bool _more_out;

    //  Routing IDs are generated. It's a simple increment and wrap-over
    //  algorithm. This value is the next ID to use (if not used already).
    uint32_t _next_integral_routing_id;

    stream_t (const stream_t &);
    const stream_t &operator= (const stream_t &);
};
}

#endif


//========= end of #include "stream.hpp" ============


//========= begin of #include "timers.hpp" ============

/*
Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

This file is part of libzmq, the ZeroMQ core engine in C++.

libzmq is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (LGPL) as published
by the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

As a special exception, the Contributors give you permission to link
this library with independent modules to produce an executable,
regardless of the license terms of these independent modules, and to
copy and distribute the resulting executable under terms of your choice,
provided that you also meet, for each linked independent module, the
terms and conditions of the license of that module. An independent
module is a module which is not derived from or based on this library.
If you modify this library, you must extend this exception to your
version of the library.

libzmq is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_TIMERS_HPP_INCLUDED__
#define __ZMQ_TIMERS_HPP_INCLUDED__

#include <stddef.h>
#include <map>
#include <set>

// ans ignore: #include "clock.hpp"

namespace zmq
{
typedef void(timers_timer_fn) (int timer_id_, void *arg_);

class timers_t
{
  public:
    timers_t ();
    ~timers_t ();

    //  Add timer to the set, timer repeats forever, or until cancel is called.
    //  Returns a timer_id that is used to cancel the timer.
    //  Returns -1 if there was an error.
    int add (size_t interval_, timers_timer_fn handler_, void *arg_);

    //  Set the interval of the timer.
    //  This method is slow, cancelling exsting and adding a new timer yield better performance.
    //  Returns 0 on success and -1 on error.
    int set_interval (int timer_id_, size_t interval_);

    //  Reset the timer.
    //  This method is slow, cancelling exsting and adding a new timer yield better performance.
    //  Returns 0 on success and -1 on error.
    int reset (int timer_id_);

    //  Cancel a timer.
    //  Returns 0 on success and -1 on error.
    int cancel (int timer_id_);

    //  Returns the time in millisecond until the next timer.
    //  Returns -1 if no timer is due.
    long timeout ();

    //  Execute timers.
    //  Return 0 if all succeed and -1 if error.
    int execute ();

    //  Return false if object is not a timers class.
    bool check_tag ();

  private:
    //  Used to check whether the object is a timers class.
    uint32_t _tag;

    int _next_timer_id;

    //  Clock instance.
    clock_t _clock;

    typedef struct timer_t
    {
        int timer_id;
        size_t interval;
        timers_timer_fn *handler;
        void *arg;
    } timer_t;

    typedef std::multimap<uint64_t, timer_t> timersmap_t;
    timersmap_t _timers;

    typedef std::set<int> cancelled_timers_t;
    cancelled_timers_t _cancelled_timers;

    timers_t (const timers_t &);
    const timers_t &operator= (const timers_t &);

    struct match_by_id;
};
}

#endif


//========= end of #include "timers.hpp" ============


//========= begin of #include "vmci.hpp" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_VMCI_HPP_INCLUDED__
#define __ZMQ_VMCI_HPP_INCLUDED__

#include <string>

// ans ignore: #include "platform.hpp"
// ans ignore: #include "fd.hpp"
// ans ignore: #include "ctx.hpp"

#if defined ZMQ_HAVE_VMCI

#if defined ZMQ_HAVE_WINDOWS
// ans ignore: #include "windows.hpp"
#else
#include <sys/time.h>
#endif

namespace zmq
{
void tune_vmci_buffer_size (ctx_t *context_,
                            fd_t sockfd_,
                            uint64_t default_size_,
                            uint64_t min_size_,
                            uint64_t max_size_);

#if defined ZMQ_HAVE_WINDOWS
void tune_vmci_connect_timeout (ctx_t *context_, fd_t sockfd_, DWORD timeout_);
#else
void tune_vmci_connect_timeout (ctx_t *context_,
                                fd_t sockfd_,
                                struct timeval timeout_);
#endif
}

#endif

#endif


//========= end of #include "vmci.hpp" ============


//========= begin of #include "zmq.hpp" ============

/*
    Copyright (c) 2016-2017 ZeroMQ community
    Copyright (c) 2009-2011 250bpm s.r.o.
    Copyright (c) 2011 Botond Ballo
    Copyright (c) 2007-2009 iMatix Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
*/

#ifndef __ZMQ_HPP_INCLUDED__
#define __ZMQ_HPP_INCLUDED__

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// macros defined if has a specific standard or greater
#if (defined(__cplusplus) && __cplusplus >= 201103L)                                \
  || (defined(_MSC_VER) && _MSC_VER >= 1900)
#define ZMQ_CPP11
#endif
#if (defined(__cplusplus) && __cplusplus >= 201402L)                                \
  || (defined(_HAS_CXX14) && _HAS_CXX14 == 1)                                       \
  || (defined(_HAS_CXX17)                                                           \
      && _HAS_CXX17                                                                 \
           == 1) // _HAS_CXX14 might not be defined when using C++17 on MSVC
#define ZMQ_CPP14
#endif
#if (defined(__cplusplus) && __cplusplus >= 201703L)                                \
  || (defined(_HAS_CXX17) && _HAS_CXX17 == 1)
#define ZMQ_CPP17
#endif

#if defined(ZMQ_CPP14)
#define ZMQ_DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(_MSC_VER)
#define ZMQ_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__GNUC__)
#define ZMQ_DEPRECATED(msg) __attribute__((deprecated(msg)))
#endif

#if defined(ZMQ_CPP17)
#define ZMQ_NODISCARD [[nodiscard]]
#else
#define ZMQ_NODISCARD
#endif

#if defined(ZMQ_CPP11)
#define ZMQ_NOTHROW noexcept
#define ZMQ_EXPLICIT explicit
#define ZMQ_OVERRIDE override
#define ZMQ_NULLPTR nullptr
#define ZMQ_CONSTEXPR_FN constexpr
#define ZMQ_CONSTEXPR_VAR constexpr
#define ZMQ_CPP11_DEPRECATED(msg) ZMQ_DEPRECATED(msg)
#else
#define ZMQ_NOTHROW throw()
#define ZMQ_EXPLICIT
#define ZMQ_OVERRIDE
#define ZMQ_NULLPTR 0
#define ZMQ_CONSTEXPR_FN
#define ZMQ_CONSTEXPR_VAR const
#define ZMQ_CPP11_DEPRECATED(msg)
#endif
#if defined(ZMQ_CPP17)
#define ZMQ_INLINE_VAR inline
#else
#define ZMQ_INLINE_VAR
#endif

// ans ignore: #include "zmq.h"

#include <cassert>
#include <cstring>

#include <algorithm>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#ifdef ZMQ_CPP11
#include <array>
#include <chrono>
#include <tuple>
#include <memory>
#endif

#if defined(__has_include) && defined(ZMQ_CPP17)
#define CPPZMQ_HAS_INCLUDE_CPP17(X) __has_include(X)
#else
#define CPPZMQ_HAS_INCLUDE_CPP17(X) 0
#endif

#if CPPZMQ_HAS_INCLUDE_CPP17(<optional>) && !defined(CPPZMQ_HAS_OPTIONAL)
#define CPPZMQ_HAS_OPTIONAL 1
#endif
#ifndef CPPZMQ_HAS_OPTIONAL
#define CPPZMQ_HAS_OPTIONAL 0
#elif CPPZMQ_HAS_OPTIONAL
#include <optional>
#endif

#if CPPZMQ_HAS_INCLUDE_CPP17(<string_view>) && !defined(CPPZMQ_HAS_STRING_VIEW)
#define CPPZMQ_HAS_STRING_VIEW 1
#endif
#ifndef CPPZMQ_HAS_STRING_VIEW
#define CPPZMQ_HAS_STRING_VIEW 0
#elif CPPZMQ_HAS_STRING_VIEW
#include <string_view>
#endif

/*  Version macros for compile-time API version detection                     */
#define CPPZMQ_VERSION_MAJOR 4
#define CPPZMQ_VERSION_MINOR 7
#define CPPZMQ_VERSION_PATCH 0

#define CPPZMQ_VERSION                                                              \
    ZMQ_MAKE_VERSION(CPPZMQ_VERSION_MAJOR, CPPZMQ_VERSION_MINOR,                    \
                     CPPZMQ_VERSION_PATCH)

//  Detect whether the compiler supports C++11 rvalue references.
#if (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 2))   \
     && defined(__GXX_EXPERIMENTAL_CXX0X__))
#define ZMQ_HAS_RVALUE_REFS
#define ZMQ_DELETED_FUNCTION = delete
#elif defined(__clang__)
#if __has_feature(cxx_rvalue_references)
#define ZMQ_HAS_RVALUE_REFS
#endif

#if __has_feature(cxx_deleted_functions)
#define ZMQ_DELETED_FUNCTION = delete
#else
#define ZMQ_DELETED_FUNCTION
#endif
#elif defined(_MSC_VER) && (_MSC_VER >= 1900)
#define ZMQ_HAS_RVALUE_REFS
#define ZMQ_DELETED_FUNCTION = delete
#elif defined(_MSC_VER) && (_MSC_VER >= 1600)
#define ZMQ_HAS_RVALUE_REFS
#define ZMQ_DELETED_FUNCTION
#else
#define ZMQ_DELETED_FUNCTION
#endif

#if defined(ZMQ_CPP11) && !defined(__llvm__) && !defined(__INTEL_COMPILER)          \
  && defined(__GNUC__) && __GNUC__ < 5
#define ZMQ_CPP11_PARTIAL
#elif defined(__GLIBCXX__) && __GLIBCXX__ < 20160805
//the date here is the last date of gcc 4.9.4, which
// effectively means libstdc++ from gcc 5.5 and higher won't trigger this branch
#define ZMQ_CPP11_PARTIAL
#endif

#ifdef ZMQ_CPP11
#ifdef ZMQ_CPP11_PARTIAL
#define ZMQ_IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#include <type_traits>
#define ZMQ_IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif
#endif

#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(3, 3, 0)
#define ZMQ_NEW_MONITOR_EVENT_LAYOUT
#endif

#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 1, 0)
#define ZMQ_HAS_PROXY_STEERABLE
/*  Socket event data  */
typedef struct
{
    uint16_t event; // id of the event as bitfield
    int32_t value;  // value is either error code, fd or reconnect interval
} zmq_event_t;
#endif

// Avoid using deprecated message receive function when possible
#if ZMQ_VERSION < ZMQ_MAKE_VERSION(3, 2, 0)
#define zmq_msg_recv(msg, socket, flags) zmq_recvmsg(socket, msg, flags)
#endif


// In order to prevent unused variable warnings when building in non-debug
// mode use this macro to make assertions.
#ifndef NDEBUG
#define ZMQ_ASSERT(expression) assert(expression)
#else
#define ZMQ_ASSERT(expression) (void) (expression)
#endif

namespace zmq
{
#ifdef ZMQ_CPP11
namespace detail
{
namespace ranges
{
using std::begin;
using std::end;
template<class T> auto begin(T &&r) -> decltype(begin(std::forward<T>(r)))
{
    return begin(std::forward<T>(r));
}
template<class T> auto end(T &&r) -> decltype(end(std::forward<T>(r)))
{
    return end(std::forward<T>(r));
}
} // namespace ranges

template<class T> using void_t = void;

template<class Iter>
using iter_value_t = typename std::iterator_traits<Iter>::value_type;

template<class Range>
using range_iter_t = decltype(
  ranges::begin(std::declval<typename std::remove_reference<Range>::type &>()));

template<class Range> using range_value_t = iter_value_t<range_iter_t<Range>>;

template<class T, class = void> struct is_range : std::false_type
{
};

template<class T>
struct is_range<
  T,
  void_t<decltype(
    ranges::begin(std::declval<typename std::remove_reference<T>::type &>())
    == ranges::end(std::declval<typename std::remove_reference<T>::type &>()))>>
    : std::true_type
{
};

} // namespace detail
#endif

typedef zmq_free_fn free_fn;
typedef zmq_pollitem_t pollitem_t;

class error_t : public std::exception
{
  public:
    error_t() : errnum(zmq_errno()) {}
    virtual const char *what() const ZMQ_NOTHROW ZMQ_OVERRIDE
    {
        return zmq_strerror(errnum);
    }
    int num() const { return errnum; }

  private:
    int errnum;
};

inline int poll(zmq_pollitem_t *items_, size_t nitems_, long timeout_ = -1)
{
    int rc = zmq_poll(items_, static_cast<int>(nitems_), timeout_);
    if (rc < 0)
        throw error_t();
    return rc;
}

ZMQ_DEPRECATED("from 4.3.1, use poll taking non-const items")
inline int poll(zmq_pollitem_t const *items_, size_t nitems_, long timeout_ = -1)
{
    return poll(const_cast<zmq_pollitem_t *>(items_), nitems_, timeout_);
}

#ifdef ZMQ_CPP11
ZMQ_DEPRECATED("from 4.3.1, use poll taking non-const items")
inline int
poll(zmq_pollitem_t const *items, size_t nitems, std::chrono::milliseconds timeout)
{
    return poll(const_cast<zmq_pollitem_t *>(items), nitems,
                static_cast<long>(timeout.count()));
}

ZMQ_DEPRECATED("from 4.3.1, use poll taking non-const items")
inline int poll(std::vector<zmq_pollitem_t> const &items,
                std::chrono::milliseconds timeout)
{
    return poll(const_cast<zmq_pollitem_t *>(items.data()), items.size(),
                static_cast<long>(timeout.count()));
}

ZMQ_DEPRECATED("from 4.3.1, use poll taking non-const items")
inline int poll(std::vector<zmq_pollitem_t> const &items, long timeout_ = -1)
{
    return poll(const_cast<zmq_pollitem_t *>(items.data()), items.size(), timeout_);
}

inline int
poll(zmq_pollitem_t *items, size_t nitems, std::chrono::milliseconds timeout)
{
    return poll(items, nitems, static_cast<long>(timeout.count()));
}

inline int poll(std::vector<zmq_pollitem_t> &items,
                std::chrono::milliseconds timeout)
{
    return poll(items.data(), items.size(), static_cast<long>(timeout.count()));
}

ZMQ_DEPRECATED("from 4.3.1, use poll taking std::chrono instead of long")
inline int poll(std::vector<zmq_pollitem_t> &items, long timeout_ = -1)
{
    return poll(items.data(), items.size(), timeout_);
}

template<std::size_t SIZE>
inline int poll(std::array<zmq_pollitem_t, SIZE>& items,
    std::chrono::milliseconds timeout)
{
    return poll(items.data(), items.size(), static_cast<long>(timeout.count()));
}
#endif


inline void version(int *major_, int *minor_, int *patch_)
{
    zmq_version(major_, minor_, patch_);
}

#ifdef ZMQ_CPP11
inline std::tuple<int, int, int> version()
{
    std::tuple<int, int, int> v;
    zmq_version(&std::get<0>(v), &std::get<1>(v), &std::get<2>(v));
    return v;
}
#endif

class message_t
{
  public:
    message_t() ZMQ_NOTHROW
    {
        int rc = zmq_msg_init(&msg);
        ZMQ_ASSERT(rc == 0);
    }

    explicit message_t(size_t size_)
    {
        int rc = zmq_msg_init_size(&msg, size_);
        if (rc != 0)
            throw error_t();
    }

    template<class ForwardIter> message_t(ForwardIter first, ForwardIter last)
    {
        typedef typename std::iterator_traits<ForwardIter>::value_type value_t;

        assert(std::distance(first, last) >= 0);
        size_t const size_ =
          static_cast<size_t>(std::distance(first, last)) * sizeof(value_t);
        int const rc = zmq_msg_init_size(&msg, size_);
        if (rc != 0)
            throw error_t();
        std::copy(first, last, data<value_t>());
    }

    message_t(const void *data_, size_t size_)
    {
        int rc = zmq_msg_init_size(&msg, size_);
        if (rc != 0)
            throw error_t();
        if (size_)
        {
            // this constructor allows (nullptr, 0),
            // memcpy with a null pointer is UB
            memcpy(data(), data_, size_);
        }
    }

    message_t(void *data_, size_t size_, free_fn *ffn_, void *hint_ = ZMQ_NULLPTR)
    {
        int rc = zmq_msg_init_data(&msg, data_, size_, ffn_, hint_);
        if (rc != 0)
            throw error_t();
    }

#if defined(ZMQ_CPP11) && !defined(ZMQ_CPP11_PARTIAL)
    template<class Range,
             typename = typename std::enable_if<
               detail::is_range<Range>::value
               && ZMQ_IS_TRIVIALLY_COPYABLE(detail::range_value_t<Range>)
               && !std::is_same<Range, message_t>::value>::type>
    explicit message_t(const Range &rng) :
        message_t(detail::ranges::begin(rng), detail::ranges::end(rng))
    {
    }
#endif

#ifdef ZMQ_HAS_RVALUE_REFS
    message_t(message_t &&rhs) ZMQ_NOTHROW : msg(rhs.msg)
    {
        int rc = zmq_msg_init(&rhs.msg);
        ZMQ_ASSERT(rc == 0);
    }

    message_t &operator=(message_t &&rhs) ZMQ_NOTHROW
    {
        std::swap(msg, rhs.msg);
        return *this;
    }
#endif

    ~message_t() ZMQ_NOTHROW
    {
        int rc = zmq_msg_close(&msg);
        ZMQ_ASSERT(rc == 0);
    }

    void rebuild()
    {
        int rc = zmq_msg_close(&msg);
        if (rc != 0)
            throw error_t();
        rc = zmq_msg_init(&msg);
        ZMQ_ASSERT(rc == 0);
    }

    void rebuild(size_t size_)
    {
        int rc = zmq_msg_close(&msg);
        if (rc != 0)
            throw error_t();
        rc = zmq_msg_init_size(&msg, size_);
        if (rc != 0)
            throw error_t();
    }

    void rebuild(const void *data_, size_t size_)
    {
        int rc = zmq_msg_close(&msg);
        if (rc != 0)
            throw error_t();
        rc = zmq_msg_init_size(&msg, size_);
        if (rc != 0)
            throw error_t();
        memcpy(data(), data_, size_);
    }

    void rebuild(void *data_, size_t size_, free_fn *ffn_, void *hint_ = ZMQ_NULLPTR)
    {
        int rc = zmq_msg_close(&msg);
        if (rc != 0)
            throw error_t();
        rc = zmq_msg_init_data(&msg, data_, size_, ffn_, hint_);
        if (rc != 0)
            throw error_t();
    }

    ZMQ_DEPRECATED("from 4.3.1, use move taking non-const reference instead")
    void move(message_t const *msg_)
    {
        int rc = zmq_msg_move(&msg, const_cast<zmq_msg_t *>(msg_->handle()));
        if (rc != 0)
            throw error_t();
    }

    void move(message_t &msg_)
    {
        int rc = zmq_msg_move(&msg, msg_.handle());
        if (rc != 0)
            throw error_t();
    }

    ZMQ_DEPRECATED("from 4.3.1, use copy taking non-const reference instead")
    void copy(message_t const *msg_)
    {
        int rc = zmq_msg_copy(&msg, const_cast<zmq_msg_t *>(msg_->handle()));
        if (rc != 0)
            throw error_t();
    }

    void copy(message_t &msg_)
    {
        int rc = zmq_msg_copy(&msg, msg_.handle());
        if (rc != 0)
            throw error_t();
    }

    bool more() const ZMQ_NOTHROW
    {
        int rc = zmq_msg_more(const_cast<zmq_msg_t *>(&msg));
        return rc != 0;
    }

    void *data() ZMQ_NOTHROW { return zmq_msg_data(&msg); }

    const void *data() const ZMQ_NOTHROW
    {
        return zmq_msg_data(const_cast<zmq_msg_t *>(&msg));
    }

    size_t size() const ZMQ_NOTHROW
    {
        return zmq_msg_size(const_cast<zmq_msg_t *>(&msg));
    }

    ZMQ_NODISCARD bool empty() const ZMQ_NOTHROW { return size() == 0u; }

    template<typename T> T *data() ZMQ_NOTHROW { return static_cast<T *>(data()); }

    template<typename T> T const *data() const ZMQ_NOTHROW
    {
        return static_cast<T const *>(data());
    }

    ZMQ_DEPRECATED("from 4.3.0, use operator== instead")
    bool equal(const message_t *other) const ZMQ_NOTHROW { return *this == *other; }

    bool operator==(const message_t &other) const ZMQ_NOTHROW
    {
        const size_t my_size = size();
        return my_size == other.size() && 0 == memcmp(data(), other.data(), my_size);
    }

    bool operator!=(const message_t &other) const ZMQ_NOTHROW
    {
        return !(*this == other);
    }

#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(3, 2, 0)
    int get(int property_)
    {
        int value = zmq_msg_get(&msg, property_);
        if (value == -1)
            throw error_t();
        return value;
    }
#endif

#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 1, 0)
    const char *gets(const char *property_)
    {
        const char *value = zmq_msg_gets(&msg, property_);
        if (value == ZMQ_NULLPTR)
            throw error_t();
        return value;
    }
#endif

#if defined(ZMQ_BUILD_DRAFT_API) && ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 0)
    uint32_t routing_id() const
    {
        return zmq_msg_routing_id(const_cast<zmq_msg_t *>(&msg));
    }

    void set_routing_id(uint32_t routing_id)
    {
        int rc = zmq_msg_set_routing_id(&msg, routing_id);
        if (rc != 0)
            throw error_t();
    }

    const char *group() const
    {
        return zmq_msg_group(const_cast<zmq_msg_t *>(&msg));
    }

    void set_group(const char *group)
    {
        int rc = zmq_msg_set_group(&msg, group);
        if (rc != 0)
            throw error_t();
    }
#endif

    // interpret message content as a string
    std::string to_string() const
    {
        return std::string(static_cast<const char *>(data()), size());
    }
#if CPPZMQ_HAS_STRING_VIEW
    // interpret message content as a string
    std::string_view to_string_view() const noexcept
    {
        return std::string_view(static_cast<const char *>(data()), size());
    }
#endif

    /** Dump content to string for debugging.
    *   Ascii chars are readable, the rest is printed as hex.
    *   Probably ridiculously slow.
    *   Use to_string() or to_string_view() for
    *   interpreting the message as a string.
    */
    std::string str() const
    {
        // Partly mutuated from the same method in zmq::multipart_t
        std::stringstream os;

        const unsigned char *msg_data = this->data<unsigned char>();
        unsigned char byte;
        size_t size = this->size();
        int is_ascii[2] = {0, 0};

        os << "zmq::message_t [size " << std::dec << std::setw(3)
           << std::setfill('0') << size << "] (";
        // Totally arbitrary
        if (size >= 1000) {
            os << "... too big to print)";
        } else {
            while (size--) {
                byte = *msg_data++;

                is_ascii[1] = (byte >= 32 && byte < 127);
                if (is_ascii[1] != is_ascii[0])
                    os << " "; // Separate text/non text

                if (is_ascii[1]) {
                    os << byte;
                } else {
                    os << std::hex << std::uppercase << std::setw(2)
                       << std::setfill('0') << static_cast<short>(byte);
                }
                is_ascii[0] = is_ascii[1];
            }
            os << ")";
        }
        return os.str();
    }

    void swap(message_t &other) ZMQ_NOTHROW
    {
        // this assumes zmq::msg_t from libzmq is trivially relocatable
        std::swap(msg, other.msg);
    }

    ZMQ_NODISCARD zmq_msg_t *handle() ZMQ_NOTHROW { return &msg; }
    ZMQ_NODISCARD const zmq_msg_t *handle() const ZMQ_NOTHROW { return &msg; }

  private:
    //  The underlying message
    zmq_msg_t msg;

    //  Disable implicit message copying, so that users won't use shared
    //  messages (less efficient) without being aware of the fact.
    message_t(const message_t &) ZMQ_DELETED_FUNCTION;
    void operator=(const message_t &) ZMQ_DELETED_FUNCTION;
};

inline void swap(message_t &a, message_t &b) ZMQ_NOTHROW
{
    a.swap(b);
}

#ifdef ZMQ_CPP11
enum class ctxopt
{
#ifdef ZMQ_BLOCKY
    blocky = ZMQ_BLOCKY,
#endif
#ifdef ZMQ_IO_THREADS
    io_threads = ZMQ_IO_THREADS,
#endif
#ifdef ZMQ_THREAD_SCHED_POLICY
    thread_sched_policy = ZMQ_THREAD_SCHED_POLICY,
#endif
#ifdef ZMQ_THREAD_PRIORITY
    thread_priority = ZMQ_THREAD_PRIORITY,
#endif
#ifdef ZMQ_THREAD_AFFINITY_CPU_ADD
    thread_affinity_cpu_add = ZMQ_THREAD_AFFINITY_CPU_ADD,
#endif
#ifdef ZMQ_THREAD_AFFINITY_CPU_REMOVE
    thread_affinity_cpu_remove = ZMQ_THREAD_AFFINITY_CPU_REMOVE,
#endif
#ifdef ZMQ_THREAD_NAME_PREFIX
    thread_name_prefix = ZMQ_THREAD_NAME_PREFIX,
#endif
#ifdef ZMQ_MAX_MSGSZ
    max_msgsz = ZMQ_MAX_MSGSZ,
#endif
#ifdef ZMQ_ZERO_COPY_RECV
    zero_copy_recv = ZMQ_ZERO_COPY_RECV,
#endif
#ifdef ZMQ_MAX_SOCKETS
    max_sockets = ZMQ_MAX_SOCKETS,
#endif
#ifdef ZMQ_SOCKET_LIMIT
    socket_limit = ZMQ_SOCKET_LIMIT,
#endif
#ifdef ZMQ_IPV6
    ipv6 = ZMQ_IPV6,
#endif
#ifdef ZMQ_MSG_T_SIZE
    msg_t_size = ZMQ_MSG_T_SIZE
#endif
};
#endif

class context_t
{
  public:
    context_t()
    {
        ptr = zmq_ctx_new();
        if (ptr == ZMQ_NULLPTR)
            throw error_t();
    }


    explicit context_t(int io_threads_, int max_sockets_ = ZMQ_MAX_SOCKETS_DFLT)
    {
        ptr = zmq_ctx_new();
        if (ptr == ZMQ_NULLPTR)
            throw error_t();

        int rc = zmq_ctx_set(ptr, ZMQ_IO_THREADS, io_threads_);
        ZMQ_ASSERT(rc == 0);

        rc = zmq_ctx_set(ptr, ZMQ_MAX_SOCKETS, max_sockets_);
        ZMQ_ASSERT(rc == 0);
    }

#ifdef ZMQ_HAS_RVALUE_REFS
    context_t(context_t &&rhs) ZMQ_NOTHROW : ptr(rhs.ptr) { rhs.ptr = ZMQ_NULLPTR; }
    context_t &operator=(context_t &&rhs) ZMQ_NOTHROW
    {
        close();
        std::swap(ptr, rhs.ptr);
        return *this;
    }
#endif

    ~context_t() ZMQ_NOTHROW { close(); }

    ZMQ_CPP11_DEPRECATED("from 4.7.0, use set taking zmq::ctxopt instead")
    int setctxopt(int option_, int optval_)
    {
        int rc = zmq_ctx_set(ptr, option_, optval_);
        ZMQ_ASSERT(rc == 0);
        return rc;
    }

    ZMQ_CPP11_DEPRECATED("from 4.7.0, use get taking zmq::ctxopt instead")
    int getctxopt(int option_) { return zmq_ctx_get(ptr, option_); }

#ifdef ZMQ_CPP11
    void set(ctxopt option, int optval)
    {
        int rc = zmq_ctx_set(ptr, static_cast<int>(option), optval);
        if (rc == -1)
            throw error_t();
    }

    ZMQ_NODISCARD int get(ctxopt option)
    {
        int rc = zmq_ctx_get(ptr, static_cast<int>(option));
        // some options have a default value of -1
        // which is unfortunate, and may result in errors
        // that don't make sense
        if (rc == -1)
            throw error_t();
        return rc;
    }
#endif

    // Terminates context (see also shutdown()).
    void close() ZMQ_NOTHROW
    {
        if (ptr == ZMQ_NULLPTR)
            return;

        int rc;
        do {
            rc = zmq_ctx_destroy(ptr);
        } while (rc == -1 && errno == EINTR);

        ZMQ_ASSERT(rc == 0);
        ptr = ZMQ_NULLPTR;
    }

    // Shutdown context in preparation for termination (close()).
    // Causes all blocking socket operations and any further
    // socket operations to return with ETERM.
    void shutdown() ZMQ_NOTHROW
    {
        if (ptr == ZMQ_NULLPTR)
            return;
        int rc = zmq_ctx_shutdown(ptr);
        ZMQ_ASSERT(rc == 0);
    }

    //  Be careful with this, it's probably only useful for
    //  using the C api together with an existing C++ api.
    //  Normally you should never need to use this.
    ZMQ_EXPLICIT operator void *() ZMQ_NOTHROW { return ptr; }

    ZMQ_EXPLICIT operator void const *() const ZMQ_NOTHROW { return ptr; }

    ZMQ_NODISCARD void *handle() ZMQ_NOTHROW { return ptr; }

    ZMQ_DEPRECATED("from 4.7.0, use handle() != nullptr instead")
    operator bool() const ZMQ_NOTHROW { return ptr != ZMQ_NULLPTR; }

    void swap(context_t &other) ZMQ_NOTHROW { std::swap(ptr, other.ptr); }

  private:
    void *ptr;

    context_t(const context_t &) ZMQ_DELETED_FUNCTION;
    void operator=(const context_t &) ZMQ_DELETED_FUNCTION;
};

inline void swap(context_t &a, context_t &b) ZMQ_NOTHROW
{
    a.swap(b);
}

#ifdef ZMQ_CPP11

struct recv_buffer_size
{
    size_t size;             // number of bytes written to buffer
    size_t untruncated_size; // untruncated message size in bytes

    ZMQ_NODISCARD bool truncated() const noexcept
    {
        return size != untruncated_size;
    }
};

#if CPPZMQ_HAS_OPTIONAL

using send_result_t = std::optional<size_t>;
using recv_result_t = std::optional<size_t>;
using recv_buffer_result_t = std::optional<recv_buffer_size>;

#else

namespace detail
{
// A C++11 type emulating the most basic
// operations of std::optional for trivial types
template<class T> class trivial_optional
{
  public:
    static_assert(std::is_trivial<T>::value, "T must be trivial");
    using value_type = T;

    trivial_optional() = default;
    trivial_optional(T value) noexcept : _value(value), _has_value(true) {}

    const T *operator->() const noexcept
    {
        assert(_has_value);
        return &_value;
    }
    T *operator->() noexcept
    {
        assert(_has_value);
        return &_value;
    }

    const T &operator*() const noexcept
    {
        assert(_has_value);
        return _value;
    }
    T &operator*() noexcept
    {
        assert(_has_value);
        return _value;
    }

    T &value()
    {
        if (!_has_value)
            throw std::exception();
        return _value;
    }
    const T &value() const
    {
        if (!_has_value)
            throw std::exception();
        return _value;
    }

    explicit operator bool() const noexcept { return _has_value; }
    bool has_value() const noexcept { return _has_value; }

  private:
    T _value{};
    bool _has_value{false};
};
} // namespace detail

using send_result_t = detail::trivial_optional<size_t>;
using recv_result_t = detail::trivial_optional<size_t>;
using recv_buffer_result_t = detail::trivial_optional<recv_buffer_size>;

#endif

namespace detail
{
template<class T> constexpr T enum_bit_or(T a, T b) noexcept
{
    static_assert(std::is_enum<T>::value, "must be enum");
    using U = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<U>(a) | static_cast<U>(b));
}
template<class T> constexpr T enum_bit_and(T a, T b) noexcept
{
    static_assert(std::is_enum<T>::value, "must be enum");
    using U = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<U>(a) & static_cast<U>(b));
}
template<class T> constexpr T enum_bit_xor(T a, T b) noexcept
{
    static_assert(std::is_enum<T>::value, "must be enum");
    using U = typename std::underlying_type<T>::type;
    return static_cast<T>(static_cast<U>(a) ^ static_cast<U>(b));
}
template<class T> constexpr T enum_bit_not(T a) noexcept
{
    static_assert(std::is_enum<T>::value, "must be enum");
    using U = typename std::underlying_type<T>::type;
    return static_cast<T>(~static_cast<U>(a));
}
} // namespace detail

// partially satisfies named requirement BitmaskType
enum class send_flags : int
{
    none = 0,
    dontwait = ZMQ_DONTWAIT,
    sndmore = ZMQ_SNDMORE
};

constexpr send_flags operator|(send_flags a, send_flags b) noexcept
{
    return detail::enum_bit_or(a, b);
}
constexpr send_flags operator&(send_flags a, send_flags b) noexcept
{
    return detail::enum_bit_and(a, b);
}
constexpr send_flags operator^(send_flags a, send_flags b) noexcept
{
    return detail::enum_bit_xor(a, b);
}
constexpr send_flags operator~(send_flags a) noexcept
{
    return detail::enum_bit_not(a);
}

// partially satisfies named requirement BitmaskType
enum class recv_flags : int
{
    none = 0,
    dontwait = ZMQ_DONTWAIT
};

constexpr recv_flags operator|(recv_flags a, recv_flags b) noexcept
{
    return detail::enum_bit_or(a, b);
}
constexpr recv_flags operator&(recv_flags a, recv_flags b) noexcept
{
    return detail::enum_bit_and(a, b);
}
constexpr recv_flags operator^(recv_flags a, recv_flags b) noexcept
{
    return detail::enum_bit_xor(a, b);
}
constexpr recv_flags operator~(recv_flags a) noexcept
{
    return detail::enum_bit_not(a);
}


// mutable_buffer, const_buffer and buffer are based on
// the Networking TS specification, draft:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/n4771.pdf

class mutable_buffer
{
  public:
    constexpr mutable_buffer() noexcept : _data(nullptr), _size(0) {}
    constexpr mutable_buffer(void *p, size_t n) noexcept : _data(p), _size(n)
    {
#ifdef ZMQ_CPP14
        assert(p != nullptr || n == 0);
#endif
    }

    constexpr void *data() const noexcept { return _data; }
    constexpr size_t size() const noexcept { return _size; }
    mutable_buffer &operator+=(size_t n) noexcept
    {
        // (std::min) is a workaround for when a min macro is defined
        const auto shift = (std::min)(n, _size);
        _data = static_cast<char *>(_data) + shift;
        _size -= shift;
        return *this;
    }

  private:
    void *_data;
    size_t _size;
};

inline mutable_buffer operator+(const mutable_buffer &mb, size_t n) noexcept
{
    return mutable_buffer(static_cast<char *>(mb.data()) + (std::min)(n, mb.size()),
                          mb.size() - (std::min)(n, mb.size()));
}
inline mutable_buffer operator+(size_t n, const mutable_buffer &mb) noexcept
{
    return mb + n;
}

class const_buffer
{
  public:
    constexpr const_buffer() noexcept : _data(nullptr), _size(0) {}
    constexpr const_buffer(const void *p, size_t n) noexcept : _data(p), _size(n)
    {
#ifdef ZMQ_CPP14
        assert(p != nullptr || n == 0);
#endif
    }
    constexpr const_buffer(const mutable_buffer &mb) noexcept :
        _data(mb.data()), _size(mb.size())
    {
    }

    constexpr const void *data() const noexcept { return _data; }
    constexpr size_t size() const noexcept { return _size; }
    const_buffer &operator+=(size_t n) noexcept
    {
        const auto shift = (std::min)(n, _size);
        _data = static_cast<const char *>(_data) + shift;
        _size -= shift;
        return *this;
    }

  private:
    const void *_data;
    size_t _size;
};

inline const_buffer operator+(const const_buffer &cb, size_t n) noexcept
{
    return const_buffer(static_cast<const char *>(cb.data())
                          + (std::min)(n, cb.size()),
                        cb.size() - (std::min)(n, cb.size()));
}
inline const_buffer operator+(size_t n, const const_buffer &cb) noexcept
{
    return cb + n;
}

// buffer creation

constexpr mutable_buffer buffer(void *p, size_t n) noexcept
{
    return mutable_buffer(p, n);
}
constexpr const_buffer buffer(const void *p, size_t n) noexcept
{
    return const_buffer(p, n);
}
constexpr mutable_buffer buffer(const mutable_buffer &mb) noexcept
{
    return mb;
}
inline mutable_buffer buffer(const mutable_buffer &mb, size_t n) noexcept
{
    return mutable_buffer(mb.data(), (std::min)(mb.size(), n));
}
constexpr const_buffer buffer(const const_buffer &cb) noexcept
{
    return cb;
}
inline const_buffer buffer(const const_buffer &cb, size_t n) noexcept
{
    return const_buffer(cb.data(), (std::min)(cb.size(), n));
}

namespace detail
{
template<class T> struct is_buffer
{
    static constexpr bool value =
      std::is_same<T, const_buffer>::value || std::is_same<T, mutable_buffer>::value;
};

template<class T> struct is_pod_like
{
    // NOTE: The networking draft N4771 section 16.11 requires
    // T in the buffer functions below to be
    // trivially copyable OR standard layout.
    // Here we decide to be conservative and require both.
    static constexpr bool value =
      ZMQ_IS_TRIVIALLY_COPYABLE(T) && std::is_standard_layout<T>::value;
};

template<class C> constexpr auto seq_size(const C &c) noexcept -> decltype(c.size())
{
    return c.size();
}
template<class T, size_t N>
constexpr size_t seq_size(const T (&/*array*/)[N]) noexcept
{
    return N;
}

template<class Seq>
auto buffer_contiguous_sequence(Seq &&seq) noexcept
  -> decltype(buffer(std::addressof(*std::begin(seq)), size_t{}))
{
    using T = typename std::remove_cv<
      typename std::remove_reference<decltype(*std::begin(seq))>::type>::type;
    static_assert(detail::is_pod_like<T>::value, "T must be POD");

    const auto size = seq_size(seq);
    return buffer(size != 0u ? std::addressof(*std::begin(seq)) : nullptr,
                  size * sizeof(T));
}
template<class Seq>
auto buffer_contiguous_sequence(Seq &&seq, size_t n_bytes) noexcept
  -> decltype(buffer_contiguous_sequence(seq))
{
    using T = typename std::remove_cv<
      typename std::remove_reference<decltype(*std::begin(seq))>::type>::type;
    static_assert(detail::is_pod_like<T>::value, "T must be POD");

    const auto size = seq_size(seq);
    return buffer(size != 0u ? std::addressof(*std::begin(seq)) : nullptr,
                  (std::min)(size * sizeof(T), n_bytes));
}

} // namespace detail

// C array
template<class T, size_t N> mutable_buffer buffer(T (&data)[N]) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, size_t N>
mutable_buffer buffer(T (&data)[N], size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
template<class T, size_t N> const_buffer buffer(const T (&data)[N]) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, size_t N>
const_buffer buffer(const T (&data)[N], size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
// std::array
template<class T, size_t N> mutable_buffer buffer(std::array<T, N> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, size_t N>
mutable_buffer buffer(std::array<T, N> &data, size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
template<class T, size_t N>
const_buffer buffer(std::array<const T, N> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, size_t N>
const_buffer buffer(std::array<const T, N> &data, size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
template<class T, size_t N>
const_buffer buffer(const std::array<T, N> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, size_t N>
const_buffer buffer(const std::array<T, N> &data, size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
// std::vector
template<class T, class Allocator>
mutable_buffer buffer(std::vector<T, Allocator> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, class Allocator>
mutable_buffer buffer(std::vector<T, Allocator> &data, size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
template<class T, class Allocator>
const_buffer buffer(const std::vector<T, Allocator> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, class Allocator>
const_buffer buffer(const std::vector<T, Allocator> &data, size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
// std::basic_string
template<class T, class Traits, class Allocator>
mutable_buffer buffer(std::basic_string<T, Traits, Allocator> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, class Traits, class Allocator>
mutable_buffer buffer(std::basic_string<T, Traits, Allocator> &data,
                      size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
template<class T, class Traits, class Allocator>
const_buffer buffer(const std::basic_string<T, Traits, Allocator> &data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, class Traits, class Allocator>
const_buffer buffer(const std::basic_string<T, Traits, Allocator> &data,
                    size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}

#if CPPZMQ_HAS_STRING_VIEW
// std::basic_string_view
template<class T, class Traits>
const_buffer buffer(std::basic_string_view<T, Traits> data) noexcept
{
    return detail::buffer_contiguous_sequence(data);
}
template<class T, class Traits>
const_buffer buffer(std::basic_string_view<T, Traits> data, size_t n_bytes) noexcept
{
    return detail::buffer_contiguous_sequence(data, n_bytes);
}
#endif

// Buffer for a string literal (null terminated)
// where the buffer size excludes the terminating character.
// Equivalent to zmq::buffer(std::string_view("...")).
template<class Char, size_t N>
constexpr const_buffer str_buffer(const Char (&data)[N]) noexcept
{
    static_assert(detail::is_pod_like<Char>::value, "Char must be POD");
#ifdef ZMQ_CPP14
    assert(data[N - 1] == Char{0});
#endif
    return const_buffer(static_cast<const Char *>(data), (N - 1) * sizeof(Char));
}

namespace literals
{
constexpr const_buffer operator"" _zbuf(const char *str, size_t len) noexcept
{
    return const_buffer(str, len * sizeof(char));
}
constexpr const_buffer operator"" _zbuf(const wchar_t *str, size_t len) noexcept
{
    return const_buffer(str, len * sizeof(wchar_t));
}
constexpr const_buffer operator"" _zbuf(const char16_t *str, size_t len) noexcept
{
    return const_buffer(str, len * sizeof(char16_t));
}
constexpr const_buffer operator"" _zbuf(const char32_t *str, size_t len) noexcept
{
    return const_buffer(str, len * sizeof(char32_t));
}
}

#endif // ZMQ_CPP11


#ifdef ZMQ_CPP11
namespace sockopt
{
// There are two types of options,
// integral type with known compiler time size (int, bool, int64_t, uint64_t)
// and arrays with dynamic size (strings, binary data).

// BoolUnit: if true accepts values of type bool (but passed as T into libzmq)
template<int Opt, class T, bool BoolUnit = false> struct integral_option
{
};

// NullTerm:
// 0: binary data
// 1: null-terminated string (`getsockopt` size includes null)
// 2: binary (size 32) or Z85 encoder string of size 41 (null included)
template<int Opt, int NullTerm = 1> struct array_option
{
};

#define ZMQ_DEFINE_INTEGRAL_OPT(OPT, NAME, TYPE)                                    \
    using NAME##_t = integral_option<OPT, TYPE, false>;                             \
    ZMQ_INLINE_VAR ZMQ_CONSTEXPR_VAR NAME##_t NAME{}
#define ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(OPT, NAME, TYPE)                          \
    using NAME##_t = integral_option<OPT, TYPE, true>;                              \
    ZMQ_INLINE_VAR ZMQ_CONSTEXPR_VAR NAME##_t NAME{}
#define ZMQ_DEFINE_ARRAY_OPT(OPT, NAME)                                             \
    using NAME##_t = array_option<OPT>;                                             \
    ZMQ_INLINE_VAR ZMQ_CONSTEXPR_VAR NAME##_t NAME{}
#define ZMQ_DEFINE_ARRAY_OPT_BINARY(OPT, NAME)                                      \
    using NAME##_t = array_option<OPT, 0>;                                          \
    ZMQ_INLINE_VAR ZMQ_CONSTEXPR_VAR NAME##_t NAME{}
#define ZMQ_DEFINE_ARRAY_OPT_BIN_OR_Z85(OPT, NAME)                                  \
    using NAME##_t = array_option<OPT, 2>;                                          \
    ZMQ_INLINE_VAR ZMQ_CONSTEXPR_VAR NAME##_t NAME{}

// duplicate definition from libzmq 4.3.3
#if defined _WIN32
#if defined _WIN64
typedef unsigned __int64 cppzmq_fd_t;
#else
typedef unsigned int cppzmq_fd_t;
#endif
#else
typedef int cppzmq_fd_t;
#endif

#ifdef ZMQ_AFFINITY
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_AFFINITY, affinity, uint64_t);
#endif
#ifdef ZMQ_BACKLOG
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_BACKLOG, backlog, int);
#endif
#ifdef ZMQ_BINDTODEVICE
ZMQ_DEFINE_ARRAY_OPT_BINARY(ZMQ_BINDTODEVICE, bindtodevice);
#endif
#ifdef ZMQ_CONFLATE
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_CONFLATE, conflate, int);
#endif
#ifdef ZMQ_CONNECT_ROUTING_ID
ZMQ_DEFINE_ARRAY_OPT(ZMQ_CONNECT_ROUTING_ID, connect_routing_id);
#endif
#ifdef ZMQ_CONNECT_TIMEOUT
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_CONNECT_TIMEOUT, connect_timeout, int);
#endif
#ifdef ZMQ_CURVE_PUBLICKEY
ZMQ_DEFINE_ARRAY_OPT_BIN_OR_Z85(ZMQ_CURVE_PUBLICKEY, curve_publickey);
#endif
#ifdef ZMQ_CURVE_SECRETKEY
ZMQ_DEFINE_ARRAY_OPT_BIN_OR_Z85(ZMQ_CURVE_SECRETKEY, curve_secretkey);
#endif
#ifdef ZMQ_CURVE_SERVER
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_CURVE_SERVER, curve_server, int);
#endif
#ifdef ZMQ_CURVE_SERVERKEY
ZMQ_DEFINE_ARRAY_OPT_BIN_OR_Z85(ZMQ_CURVE_SERVERKEY, curve_serverkey);
#endif
#ifdef ZMQ_EVENTS
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_EVENTS, events, int);
#endif
#ifdef ZMQ_FD
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_FD, fd, cppzmq_fd_t);
#endif
#ifdef ZMQ_GSSAPI_PLAINTEXT
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_GSSAPI_PLAINTEXT, gssapi_plaintext, int);
#endif
#ifdef ZMQ_GSSAPI_SERVER
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_GSSAPI_SERVER, gssapi_server, int);
#endif
#ifdef ZMQ_GSSAPI_SERVICE_PRINCIPAL
ZMQ_DEFINE_ARRAY_OPT(ZMQ_GSSAPI_SERVICE_PRINCIPAL, gssapi_service_principal);
#endif
#ifdef ZMQ_GSSAPI_SERVICE_PRINCIPAL_NAMETYPE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_GSSAPI_SERVICE_PRINCIPAL_NAMETYPE,
                        gssapi_service_principal_nametype,
                        int);
#endif
#ifdef ZMQ_GSSAPI_PRINCIPAL
ZMQ_DEFINE_ARRAY_OPT(ZMQ_GSSAPI_PRINCIPAL, gssapi_principal);
#endif
#ifdef ZMQ_GSSAPI_PRINCIPAL_NAMETYPE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_GSSAPI_PRINCIPAL_NAMETYPE,
                        gssapi_principal_nametype,
                        int);
#endif
#ifdef ZMQ_HANDSHAKE_IVL
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_HANDSHAKE_IVL, handshake_ivl, int);
#endif
#ifdef ZMQ_HEARTBEAT_IVL
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_HEARTBEAT_IVL, heartbeat_ivl, int);
#endif
#ifdef ZMQ_HEARTBEAT_TIMEOUT
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_HEARTBEAT_TIMEOUT, heartbeat_timeout, int);
#endif
#ifdef ZMQ_HEARTBEAT_TTL
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_HEARTBEAT_TTL, heartbeat_ttl, int);
#endif
#ifdef ZMQ_IMMEDIATE
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_IMMEDIATE, immediate, int);
#endif
#ifdef ZMQ_INVERT_MATCHING
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_INVERT_MATCHING, invert_matching, int);
#endif
#ifdef ZMQ_IPV6
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_IPV6, ipv6, int);
#endif
#ifdef ZMQ_LAST_ENDPOINT
ZMQ_DEFINE_ARRAY_OPT(ZMQ_LAST_ENDPOINT, last_endpoint);
#endif
#ifdef ZMQ_LINGER
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_LINGER, linger, int);
#endif
#ifdef ZMQ_MAXMSGSIZE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_MAXMSGSIZE, maxmsgsize, int64_t);
#endif
#ifdef ZMQ_MECHANISM
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_MECHANISM, mechanism, int);
#endif
#ifdef ZMQ_METADATA
ZMQ_DEFINE_ARRAY_OPT(ZMQ_METADATA, metadata);
#endif
#ifdef ZMQ_MULTICAST_HOPS
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_MULTICAST_HOPS, multicast_hops, int);
#endif
#ifdef ZMQ_MULTICAST_LOOP
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_MULTICAST_LOOP, multicast_loop, int);
#endif
#ifdef ZMQ_MULTICAST_MAXTPDU
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_MULTICAST_MAXTPDU, multicast_maxtpdu, int);
#endif
#ifdef ZMQ_PLAIN_SERVER
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_PLAIN_SERVER, plain_server, int);
#endif
#ifdef ZMQ_PLAIN_PASSWORD
ZMQ_DEFINE_ARRAY_OPT(ZMQ_PLAIN_PASSWORD, plain_password);
#endif
#ifdef ZMQ_PLAIN_USERNAME
ZMQ_DEFINE_ARRAY_OPT(ZMQ_PLAIN_USERNAME, plain_username);
#endif
#ifdef ZMQ_USE_FD
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_USE_FD, use_fd, int);
#endif
#ifdef ZMQ_PROBE_ROUTER
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_PROBE_ROUTER, probe_router, int);
#endif
#ifdef ZMQ_RATE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RATE, rate, int);
#endif
#ifdef ZMQ_RCVBUF
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RCVBUF, rcvbuf, int);
#endif
#ifdef ZMQ_RCVHWM
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RCVHWM, rcvhwm, int);
#endif
#ifdef ZMQ_RCVMORE
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_RCVMORE, rcvmore, int);
#endif
#ifdef ZMQ_RCVTIMEO
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RCVTIMEO, rcvtimeo, int);
#endif
#ifdef ZMQ_RECONNECT_IVL
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RECONNECT_IVL, reconnect_ivl, int);
#endif
#ifdef ZMQ_RECONNECT_IVL_MAX
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RECONNECT_IVL_MAX, reconnect_ivl_max, int);
#endif
#ifdef ZMQ_RECOVERY_IVL
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_RECOVERY_IVL, recovery_ivl, int);
#endif
#ifdef ZMQ_REQ_CORRELATE
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_REQ_CORRELATE, req_correlate, int);
#endif
#ifdef ZMQ_REQ_RELAXED
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_REQ_RELAXED, req_relaxed, int);
#endif
#ifdef ZMQ_ROUTER_HANDOVER
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_ROUTER_HANDOVER, router_handover, int);
#endif
#ifdef ZMQ_ROUTER_MANDATORY
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_ROUTER_MANDATORY, router_mandatory, int);
#endif
#ifdef ZMQ_ROUTER_NOTIFY
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_ROUTER_NOTIFY, router_notify, int);
#endif
#ifdef ZMQ_ROUTING_ID
ZMQ_DEFINE_ARRAY_OPT_BINARY(ZMQ_ROUTING_ID, routing_id);
#endif
#ifdef ZMQ_SNDBUF
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_SNDBUF, sndbuf, int);
#endif
#ifdef ZMQ_SNDHWM
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_SNDHWM, sndhwm, int);
#endif
#ifdef ZMQ_SNDTIMEO
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_SNDTIMEO, sndtimeo, int);
#endif
#ifdef ZMQ_SOCKS_PROXY
ZMQ_DEFINE_ARRAY_OPT(ZMQ_SOCKS_PROXY, socks_proxy);
#endif
#ifdef ZMQ_STREAM_NOTIFY
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_STREAM_NOTIFY, stream_notify, int);
#endif
#ifdef ZMQ_SUBSCRIBE
ZMQ_DEFINE_ARRAY_OPT(ZMQ_SUBSCRIBE, subscribe);
#endif
#ifdef ZMQ_TCP_KEEPALIVE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TCP_KEEPALIVE, tcp_keepalive, int);
#endif
#ifdef ZMQ_TCP_KEEPALIVE_CNT
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TCP_KEEPALIVE_CNT, tcp_keepalive_cnt, int);
#endif
#ifdef ZMQ_TCP_KEEPALIVE_IDLE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TCP_KEEPALIVE_IDLE, tcp_keepalive_idle, int);
#endif
#ifdef ZMQ_TCP_KEEPALIVE_INTVL
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TCP_KEEPALIVE_INTVL, tcp_keepalive_intvl, int);
#endif
#ifdef ZMQ_TCP_MAXRT
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TCP_MAXRT, tcp_maxrt, int);
#endif
#ifdef ZMQ_THREAD_SAFE
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_THREAD_SAFE, thread_safe, int);
#endif
#ifdef ZMQ_TOS
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TOS, tos, int);
#endif
#ifdef ZMQ_TYPE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_TYPE, type, int);
#endif
#ifdef ZMQ_UNSUBSCRIBE
ZMQ_DEFINE_ARRAY_OPT(ZMQ_UNSUBSCRIBE, unsubscribe);
#endif
#ifdef ZMQ_VMCI_BUFFER_SIZE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_VMCI_BUFFER_SIZE, vmci_buffer_size, uint64_t);
#endif
#ifdef ZMQ_VMCI_BUFFER_MIN_SIZE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_VMCI_BUFFER_MIN_SIZE, vmci_buffer_min_size, uint64_t);
#endif
#ifdef ZMQ_VMCI_BUFFER_MAX_SIZE
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_VMCI_BUFFER_MAX_SIZE, vmci_buffer_max_size, uint64_t);
#endif
#ifdef ZMQ_VMCI_CONNECT_TIMEOUT
ZMQ_DEFINE_INTEGRAL_OPT(ZMQ_VMCI_CONNECT_TIMEOUT, vmci_connect_timeout, int);
#endif
#ifdef ZMQ_XPUB_VERBOSE
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_XPUB_VERBOSE, xpub_verbose, int);
#endif
#ifdef ZMQ_XPUB_VERBOSER
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_XPUB_VERBOSER, xpub_verboser, int);
#endif
#ifdef ZMQ_XPUB_MANUAL
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_XPUB_MANUAL, xpub_manual, int);
#endif
#ifdef ZMQ_XPUB_NODROP
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_XPUB_NODROP, xpub_nodrop, int);
#endif
#ifdef ZMQ_XPUB_WELCOME_MSG
ZMQ_DEFINE_ARRAY_OPT(ZMQ_XPUB_WELCOME_MSG, xpub_welcome_msg);
#endif
#ifdef ZMQ_ZAP_ENFORCE_DOMAIN
ZMQ_DEFINE_INTEGRAL_BOOL_UNIT_OPT(ZMQ_ZAP_ENFORCE_DOMAIN, zap_enforce_domain, int);
#endif
#ifdef ZMQ_ZAP_DOMAIN
ZMQ_DEFINE_ARRAY_OPT(ZMQ_ZAP_DOMAIN, zap_domain);
#endif

} // namespace sockopt
#endif // ZMQ_CPP11


namespace detail
{
class socket_base
{
  public:
    socket_base() ZMQ_NOTHROW : _handle(ZMQ_NULLPTR) {}
    ZMQ_EXPLICIT socket_base(void *handle) ZMQ_NOTHROW : _handle(handle) {}

    template<typename T>
    ZMQ_CPP11_DEPRECATED("from 4.7.0, use `set` taking option from zmq::sockopt")
    void setsockopt(int option_, T const &optval)
    {
        setsockopt(option_, &optval, sizeof(T));
    }

    ZMQ_CPP11_DEPRECATED("from 4.7.0, use `set` taking option from zmq::sockopt")
    void setsockopt(int option_, const void *optval_, size_t optvallen_)
    {
        int rc = zmq_setsockopt(_handle, option_, optval_, optvallen_);
        if (rc != 0)
            throw error_t();
    }

    ZMQ_CPP11_DEPRECATED("from 4.7.0, use `get` taking option from zmq::sockopt")
    void getsockopt(int option_, void *optval_, size_t *optvallen_) const
    {
        int rc = zmq_getsockopt(_handle, option_, optval_, optvallen_);
        if (rc != 0)
            throw error_t();
    }

    template<typename T>
    ZMQ_CPP11_DEPRECATED("from 4.7.0, use `get` taking option from zmq::sockopt")
    T getsockopt(int option_) const
    {
        T optval;
        size_t optlen = sizeof(T);
        getsockopt(option_, &optval, &optlen);
        return optval;
    }

#ifdef ZMQ_CPP11
    // Set integral socket option, e.g.
    // `socket.set(zmq::sockopt::linger, 0)`
    template<int Opt, class T, bool BoolUnit>
    void set(sockopt::integral_option<Opt, T, BoolUnit>, const T &val)
    {
        static_assert(std::is_integral<T>::value, "T must be integral");
        set_option(Opt, &val, sizeof val);
    }

    // Set integral socket option from boolean, e.g.
    // `socket.set(zmq::sockopt::immediate, false)`
    template<int Opt, class T>
    void set(sockopt::integral_option<Opt, T, true>, bool val)
    {
        static_assert(std::is_integral<T>::value, "T must be integral");
        T rep_val = val;
        set_option(Opt, &rep_val, sizeof rep_val);
    }

    // Set array socket option, e.g.
    // `socket.set(zmq::sockopt::plain_username, "foo123")`
    template<int Opt, int NullTerm>
    void set(sockopt::array_option<Opt, NullTerm>, const char *buf)
    {
        set_option(Opt, buf, std::strlen(buf));
    }

    // Set array socket option, e.g.
    // `socket.set(zmq::sockopt::routing_id, zmq::buffer(id))`
    template<int Opt, int NullTerm>
    void set(sockopt::array_option<Opt, NullTerm>, const_buffer buf)
    {
        set_option(Opt, buf.data(), buf.size());
    }

    // Set array socket option, e.g.
    // `socket.set(zmq::sockopt::routing_id, id_str)`
    template<int Opt, int NullTerm>
    void set(sockopt::array_option<Opt, NullTerm>, const std::string &buf)
    {
        set_option(Opt, buf.data(), buf.size());
    }

#if CPPZMQ_HAS_STRING_VIEW
    // Set array socket option, e.g.
    // `socket.set(zmq::sockopt::routing_id, id_str)`
    template<int Opt, int NullTerm>
    void set(sockopt::array_option<Opt, NullTerm>, std::string_view buf)
    {
        set_option(Opt, buf.data(), buf.size());
    }
#endif

    // Get scalar socket option, e.g.
    // `auto opt = socket.get(zmq::sockopt::linger)`
    template<int Opt, class T, bool BoolUnit>
    ZMQ_NODISCARD T get(sockopt::integral_option<Opt, T, BoolUnit>) const
    {
        static_assert(std::is_integral<T>::value, "T must be integral");
        T val;
        size_t size = sizeof val;
        get_option(Opt, &val, &size);
        assert(size == sizeof val);
        return val;
    }

    // Get array socket option, writes to buf, returns option size in bytes, e.g.
    // `size_t optsize = socket.get(zmq::sockopt::routing_id, zmq::buffer(id))`
    template<int Opt, int NullTerm>
    ZMQ_NODISCARD size_t get(sockopt::array_option<Opt, NullTerm>,
                             mutable_buffer buf) const
    {
        size_t size = buf.size();
        get_option(Opt, buf.data(), &size);
        return size;
    }

    // Get array socket option as string (initializes the string buffer size to init_size) e.g.
    // `auto s = socket.get(zmq::sockopt::routing_id)`
    // Note: removes the null character from null-terminated string options,
    // i.e. the string size excludes the null character.
    template<int Opt, int NullTerm>
    ZMQ_NODISCARD std::string get(sockopt::array_option<Opt, NullTerm>,
                                  size_t init_size = 1024) const
    {
        if (NullTerm == 2 && init_size == 1024) {
            init_size = 41; // get as Z85 string
        }
        std::string str(init_size, '\0');
        size_t size = get(sockopt::array_option<Opt>{}, buffer(str));
        if (NullTerm == 1) {
            if (size > 0) {
                assert(str[size - 1] == '\0');
                --size;
            }
        } else if (NullTerm == 2) {
            assert(size == 32 || size == 41);
            if (size == 41) {
                assert(str[size - 1] == '\0');
                --size;
            }
        }
        str.resize(size);
        return str;
    }
#endif

    void bind(std::string const &addr) { bind(addr.c_str()); }

    void bind(const char *addr_)
    {
        int rc = zmq_bind(_handle, addr_);
        if (rc != 0)
            throw error_t();
    }

    void unbind(std::string const &addr) { unbind(addr.c_str()); }

    void unbind(const char *addr_)
    {
        int rc = zmq_unbind(_handle, addr_);
        if (rc != 0)
            throw error_t();
    }

    void connect(std::string const &addr) { connect(addr.c_str()); }

    void connect(const char *addr_)
    {
        int rc = zmq_connect(_handle, addr_);
        if (rc != 0)
            throw error_t();
    }

    void disconnect(std::string const &addr) { disconnect(addr.c_str()); }

    void disconnect(const char *addr_)
    {
        int rc = zmq_disconnect(_handle, addr_);
        if (rc != 0)
            throw error_t();
    }

    bool connected() const ZMQ_NOTHROW { return (_handle != ZMQ_NULLPTR); }

    ZMQ_CPP11_DEPRECATED("from 4.3.1, use send taking a const_buffer and send_flags")
    size_t send(const void *buf_, size_t len_, int flags_ = 0)
    {
        int nbytes = zmq_send(_handle, buf_, len_, flags_);
        if (nbytes >= 0)
            return static_cast<size_t>(nbytes);
        if (zmq_errno() == EAGAIN)
            return 0;
        throw error_t();
    }

    ZMQ_CPP11_DEPRECATED("from 4.3.1, use send taking message_t and send_flags")
    bool send(message_t &msg_,
              int flags_ = 0) // default until removed
    {
        int nbytes = zmq_msg_send(msg_.handle(), _handle, flags_);
        if (nbytes >= 0)
            return true;
        if (zmq_errno() == EAGAIN)
            return false;
        throw error_t();
    }

    template<typename T>
    ZMQ_CPP11_DEPRECATED(
      "from 4.4.1, use send taking message_t or buffer (for contiguous "
      "ranges), and send_flags")
    bool send(T first, T last, int flags_ = 0)
    {
        zmq::message_t msg(first, last);
        int nbytes = zmq_msg_send(msg.handle(), _handle, flags_);
        if (nbytes >= 0)
            return true;
        if (zmq_errno() == EAGAIN)
            return false;
        throw error_t();
    }

#ifdef ZMQ_HAS_RVALUE_REFS
    ZMQ_CPP11_DEPRECATED("from 4.3.1, use send taking message_t and send_flags")
    bool send(message_t &&msg_,
              int flags_ = 0) // default until removed
    {
#ifdef ZMQ_CPP11
        return send(msg_, static_cast<send_flags>(flags_)).has_value();
#else
        return send(msg_, flags_);
#endif
    }
#endif

#ifdef ZMQ_CPP11
    send_result_t send(const_buffer buf, send_flags flags = send_flags::none)
    {
        const int nbytes =
          zmq_send(_handle, buf.data(), buf.size(), static_cast<int>(flags));
        if (nbytes >= 0)
            return static_cast<size_t>(nbytes);
        if (zmq_errno() == EAGAIN)
            return {};
        throw error_t();
    }

    send_result_t send(message_t &msg, send_flags flags)
    {
        int nbytes = zmq_msg_send(msg.handle(), _handle, static_cast<int>(flags));
        if (nbytes >= 0)
            return static_cast<size_t>(nbytes);
        if (zmq_errno() == EAGAIN)
            return {};
        throw error_t();
    }

    send_result_t send(message_t &&msg, send_flags flags)
    {
        return send(msg, flags);
    }
#endif

    ZMQ_CPP11_DEPRECATED(
      "from 4.3.1, use recv taking a mutable_buffer and recv_flags")
    size_t recv(void *buf_, size_t len_, int flags_ = 0)
    {
        int nbytes = zmq_recv(_handle, buf_, len_, flags_);
        if (nbytes >= 0)
            return static_cast<size_t>(nbytes);
        if (zmq_errno() == EAGAIN)
            return 0;
        throw error_t();
    }

    ZMQ_CPP11_DEPRECATED(
      "from 4.3.1, use recv taking a reference to message_t and recv_flags")
    bool recv(message_t *msg_, int flags_ = 0)
    {
        int nbytes = zmq_msg_recv(msg_->handle(), _handle, flags_);
        if (nbytes >= 0)
            return true;
        if (zmq_errno() == EAGAIN)
            return false;
        throw error_t();
    }

#ifdef ZMQ_CPP11
    ZMQ_NODISCARD
    recv_buffer_result_t recv(mutable_buffer buf,
                              recv_flags flags = recv_flags::none)
    {
        const int nbytes =
          zmq_recv(_handle, buf.data(), buf.size(), static_cast<int>(flags));
        if (nbytes >= 0) {
            return recv_buffer_size{
              (std::min)(static_cast<size_t>(nbytes), buf.size()),
              static_cast<size_t>(nbytes)};
        }
        if (zmq_errno() == EAGAIN)
            return {};
        throw error_t();
    }

    ZMQ_NODISCARD
    recv_result_t recv(message_t &msg, recv_flags flags = recv_flags::none)
    {
        const int nbytes =
          zmq_msg_recv(msg.handle(), _handle, static_cast<int>(flags));
        if (nbytes >= 0) {
            assert(msg.size() == static_cast<size_t>(nbytes));
            return static_cast<size_t>(nbytes);
        }
        if (zmq_errno() == EAGAIN)
            return {};
        throw error_t();
    }
#endif

#if defined(ZMQ_BUILD_DRAFT_API) && ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 0)
    void join(const char *group)
    {
        int rc = zmq_join(_handle, group);
        if (rc != 0)
            throw error_t();
    }

    void leave(const char *group)
    {
        int rc = zmq_leave(_handle, group);
        if (rc != 0)
            throw error_t();
    }
#endif

    ZMQ_NODISCARD void *handle() ZMQ_NOTHROW { return _handle; }
    ZMQ_NODISCARD const void *handle() const ZMQ_NOTHROW { return _handle; }

    ZMQ_EXPLICIT operator bool() const ZMQ_NOTHROW { return _handle != ZMQ_NULLPTR; }
    // note: non-const operator bool can be removed once
    // operator void* is removed from socket_t
    ZMQ_EXPLICIT operator bool() ZMQ_NOTHROW { return _handle != ZMQ_NULLPTR; }

  protected:
    void *_handle;

  private:
    void set_option(int option_, const void *optval_, size_t optvallen_)
    {
        int rc = zmq_setsockopt(_handle, option_, optval_, optvallen_);
        if (rc != 0)
            throw error_t();
    }

    void get_option(int option_, void *optval_, size_t *optvallen_) const
    {
        int rc = zmq_getsockopt(_handle, option_, optval_, optvallen_);
        if (rc != 0)
            throw error_t();
    }
};
} // namespace detail

#ifdef ZMQ_CPP11
enum class socket_type : int
{
    req = ZMQ_REQ,
    rep = ZMQ_REP,
    dealer = ZMQ_DEALER,
    router = ZMQ_ROUTER,
    pub = ZMQ_PUB,
    sub = ZMQ_SUB,
    xpub = ZMQ_XPUB,
    xsub = ZMQ_XSUB,
    push = ZMQ_PUSH,
    pull = ZMQ_PULL,
#if defined(ZMQ_BUILD_DRAFT_API) && ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 0)
    server = ZMQ_SERVER,
    client = ZMQ_CLIENT,
    radio = ZMQ_RADIO,
    dish = ZMQ_DISH,
#endif
#if ZMQ_VERSION_MAJOR >= 4
    stream = ZMQ_STREAM,
#endif
    pair = ZMQ_PAIR
};
#endif

struct from_handle_t
{
    struct _private
    {
    }; // disabling use other than with from_handle
    ZMQ_CONSTEXPR_FN ZMQ_EXPLICIT from_handle_t(_private /*p*/) ZMQ_NOTHROW {}
};

ZMQ_CONSTEXPR_VAR from_handle_t from_handle =
  from_handle_t(from_handle_t::_private());

// A non-owning nullable reference to a socket.
// The reference is invalidated on socket close or destruction.
class socket_ref : public detail::socket_base
{
  public:
    socket_ref() ZMQ_NOTHROW : detail::socket_base() {}
#ifdef ZMQ_CPP11
    socket_ref(std::nullptr_t) ZMQ_NOTHROW : detail::socket_base() {}
#endif
    socket_ref(from_handle_t /*fh*/, void *handle) ZMQ_NOTHROW
        : detail::socket_base(handle)
    {
    }
};

#ifdef ZMQ_CPP11
inline bool operator==(socket_ref sr, std::nullptr_t /*p*/) ZMQ_NOTHROW
{
    return sr.handle() == nullptr;
}
inline bool operator==(std::nullptr_t /*p*/, socket_ref sr) ZMQ_NOTHROW
{
    return sr.handle() == nullptr;
}
inline bool operator!=(socket_ref sr, std::nullptr_t /*p*/) ZMQ_NOTHROW
{
    return !(sr == nullptr);
}
inline bool operator!=(std::nullptr_t /*p*/, socket_ref sr) ZMQ_NOTHROW
{
    return !(sr == nullptr);
}
#endif

inline bool operator==(socket_ref a, socket_ref b) ZMQ_NOTHROW
{
    return std::equal_to<void *>()(a.handle(), b.handle());
}
inline bool operator!=(socket_ref a, socket_ref b) ZMQ_NOTHROW
{
    return !(a == b);
}
inline bool operator<(socket_ref a, socket_ref b) ZMQ_NOTHROW
{
    return std::less<void *>()(a.handle(), b.handle());
}
inline bool operator>(socket_ref a, socket_ref b) ZMQ_NOTHROW
{
    return b < a;
}
inline bool operator<=(socket_ref a, socket_ref b) ZMQ_NOTHROW
{
    return !(a > b);
}
inline bool operator>=(socket_ref a, socket_ref b) ZMQ_NOTHROW
{
    return !(a < b);
}

} // namespace zmq

#ifdef ZMQ_CPP11
namespace std
{
template<> struct hash<zmq::socket_ref>
{
    size_t operator()(zmq::socket_ref sr) const ZMQ_NOTHROW
    {
        return hash<void *>()(sr.handle());
    }
};
} // namespace std
#endif

namespace zmq
{
class socket_t : public detail::socket_base
{
    friend class monitor_t;

  public:
    socket_t() ZMQ_NOTHROW : detail::socket_base(ZMQ_NULLPTR), ctxptr(ZMQ_NULLPTR) {}

    socket_t(context_t &context_, int type_) :
        detail::socket_base(zmq_socket(context_.handle(), type_)),
        ctxptr(context_.handle())
    {
        if (_handle == ZMQ_NULLPTR)
            throw error_t();
    }

#ifdef ZMQ_CPP11
    socket_t(context_t &context_, socket_type type_) :
        socket_t(context_, static_cast<int>(type_))
    {
    }
#endif

#ifdef ZMQ_HAS_RVALUE_REFS
    socket_t(socket_t &&rhs) ZMQ_NOTHROW : detail::socket_base(rhs._handle),
                                           ctxptr(rhs.ctxptr)
    {
        rhs._handle = ZMQ_NULLPTR;
        rhs.ctxptr = ZMQ_NULLPTR;
    }
    socket_t &operator=(socket_t &&rhs) ZMQ_NOTHROW
    {
        close();
        std::swap(_handle, rhs._handle);
        return *this;
    }
#endif

    ~socket_t() ZMQ_NOTHROW { close(); }

    operator void *() ZMQ_NOTHROW { return _handle; }

    operator void const *() const ZMQ_NOTHROW { return _handle; }

    void close() ZMQ_NOTHROW
    {
        if (_handle == ZMQ_NULLPTR)
            // already closed
            return;
        int rc = zmq_close(_handle);
        ZMQ_ASSERT(rc == 0);
        _handle = ZMQ_NULLPTR;
    }

    void swap(socket_t &other) ZMQ_NOTHROW
    {
        std::swap(_handle, other._handle);
        std::swap(ctxptr, other.ctxptr);
    }

    operator socket_ref() ZMQ_NOTHROW { return socket_ref(from_handle, _handle); }

  private:
    void *ctxptr;

    socket_t(const socket_t &) ZMQ_DELETED_FUNCTION;
    void operator=(const socket_t &) ZMQ_DELETED_FUNCTION;

    // used by monitor_t
    socket_t(void *context_, int type_) :
        detail::socket_base(zmq_socket(context_, type_)), ctxptr(context_)
    {
        if (_handle == ZMQ_NULLPTR)
            throw error_t();
    }
};

inline void swap(socket_t &a, socket_t &b) ZMQ_NOTHROW
{
    a.swap(b);
}

ZMQ_DEPRECATED("from 4.3.1, use proxy taking socket_t objects")
inline void proxy(void *frontend, void *backend, void *capture)
{
    int rc = zmq_proxy(frontend, backend, capture);
    if (rc != 0)
        throw error_t();
}

inline void
proxy(socket_ref frontend, socket_ref backend, socket_ref capture = socket_ref())
{
    int rc = zmq_proxy(frontend.handle(), backend.handle(), capture.handle());
    if (rc != 0)
        throw error_t();
}

#ifdef ZMQ_HAS_PROXY_STEERABLE
ZMQ_DEPRECATED("from 4.3.1, use proxy_steerable taking socket_t objects")
inline void
proxy_steerable(void *frontend, void *backend, void *capture, void *control)
{
    int rc = zmq_proxy_steerable(frontend, backend, capture, control);
    if (rc != 0)
        throw error_t();
}

inline void proxy_steerable(socket_ref frontend,
                            socket_ref backend,
                            socket_ref capture,
                            socket_ref control)
{
    int rc = zmq_proxy_steerable(frontend.handle(), backend.handle(),
                                 capture.handle(), control.handle());
    if (rc != 0)
        throw error_t();
}
#endif

class monitor_t
{
  public:
    monitor_t() : _socket(), _monitor_socket() {}

    virtual ~monitor_t() { close(); }

#ifdef ZMQ_HAS_RVALUE_REFS
    monitor_t(monitor_t &&rhs) ZMQ_NOTHROW : _socket(), _monitor_socket()
    {
        std::swap(_socket, rhs._socket);
        std::swap(_monitor_socket, rhs._monitor_socket);
    }

    monitor_t &operator=(monitor_t &&rhs) ZMQ_NOTHROW
    {
        close();
        _socket = socket_ref();
        std::swap(_socket, rhs._socket);
        std::swap(_monitor_socket, rhs._monitor_socket);
        return *this;
    }
#endif


    void
    monitor(socket_t &socket, std::string const &addr, int events = ZMQ_EVENT_ALL)
    {
        monitor(socket, addr.c_str(), events);
    }

    void monitor(socket_t &socket, const char *addr_, int events = ZMQ_EVENT_ALL)
    {
        init(socket, addr_, events);
        while (true) {
            check_event(-1);
        }
    }

    void init(socket_t &socket, std::string const &addr, int events = ZMQ_EVENT_ALL)
    {
        init(socket, addr.c_str(), events);
    }

    void init(socket_t &socket, const char *addr_, int events = ZMQ_EVENT_ALL)
    {
        int rc = zmq_socket_monitor(socket.handle(), addr_, events);
        if (rc != 0)
            throw error_t();

        _socket = socket;
        _monitor_socket = socket_t(socket.ctxptr, ZMQ_PAIR);
        _monitor_socket.connect(addr_);

        on_monitor_started();
    }

    bool check_event(int timeout = 0)
    {
        assert(_monitor_socket);

        zmq_msg_t eventMsg;
        zmq_msg_init(&eventMsg);

        zmq::pollitem_t items[] = {
          {_monitor_socket.handle(), 0, ZMQ_POLLIN, 0},
        };

        zmq::poll(&items[0], 1, timeout);

        if (items[0].revents & ZMQ_POLLIN) {
            int rc = zmq_msg_recv(&eventMsg, _monitor_socket.handle(), 0);
            if (rc == -1 && zmq_errno() == ETERM)
                return false;
            assert(rc != -1);

        } else {
            zmq_msg_close(&eventMsg);
            return false;
        }

#if ZMQ_VERSION_MAJOR >= 4
        const char *data = static_cast<const char *>(zmq_msg_data(&eventMsg));
        zmq_event_t msgEvent;
        memcpy(&msgEvent.event, data, sizeof(uint16_t));
        data += sizeof(uint16_t);
        memcpy(&msgEvent.value, data, sizeof(int32_t));
        zmq_event_t *event = &msgEvent;
#else
        zmq_event_t *event = static_cast<zmq_event_t *>(zmq_msg_data(&eventMsg));
#endif

#ifdef ZMQ_NEW_MONITOR_EVENT_LAYOUT
        zmq_msg_t addrMsg;
        zmq_msg_init(&addrMsg);
        int rc = zmq_msg_recv(&addrMsg, _monitor_socket.handle(), 0);
        if (rc == -1 && zmq_errno() == ETERM) {
            zmq_msg_close(&eventMsg);
            return false;
        }

        assert(rc != -1);
        const char *str = static_cast<const char *>(zmq_msg_data(&addrMsg));
        std::string address(str, str + zmq_msg_size(&addrMsg));
        zmq_msg_close(&addrMsg);
#else
        // Bit of a hack, but all events in the zmq_event_t union have the same layout so this will work for all event types.
        std::string address = event->data.connected.addr;
#endif

#ifdef ZMQ_EVENT_MONITOR_STOPPED
        if (event->event == ZMQ_EVENT_MONITOR_STOPPED) {
            zmq_msg_close(&eventMsg);
            return false;
        }

#endif

        switch (event->event) {
            case ZMQ_EVENT_CONNECTED:
                on_event_connected(*event, address.c_str());
                break;
            case ZMQ_EVENT_CONNECT_DELAYED:
                on_event_connect_delayed(*event, address.c_str());
                break;
            case ZMQ_EVENT_CONNECT_RETRIED:
                on_event_connect_retried(*event, address.c_str());
                break;
            case ZMQ_EVENT_LISTENING:
                on_event_listening(*event, address.c_str());
                break;
            case ZMQ_EVENT_BIND_FAILED:
                on_event_bind_failed(*event, address.c_str());
                break;
            case ZMQ_EVENT_ACCEPTED:
                on_event_accepted(*event, address.c_str());
                break;
            case ZMQ_EVENT_ACCEPT_FAILED:
                on_event_accept_failed(*event, address.c_str());
                break;
            case ZMQ_EVENT_CLOSED:
                on_event_closed(*event, address.c_str());
                break;
            case ZMQ_EVENT_CLOSE_FAILED:
                on_event_close_failed(*event, address.c_str());
                break;
            case ZMQ_EVENT_DISCONNECTED:
                on_event_disconnected(*event, address.c_str());
                break;
#ifdef ZMQ_BUILD_DRAFT_API
#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 3)
            case ZMQ_EVENT_HANDSHAKE_FAILED_NO_DETAIL:
                on_event_handshake_failed_no_detail(*event, address.c_str());
                break;
            case ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL:
                on_event_handshake_failed_protocol(*event, address.c_str());
                break;
            case ZMQ_EVENT_HANDSHAKE_FAILED_AUTH:
                on_event_handshake_failed_auth(*event, address.c_str());
                break;
            case ZMQ_EVENT_HANDSHAKE_SUCCEEDED:
                on_event_handshake_succeeded(*event, address.c_str());
                break;
#elif ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 1)
            case ZMQ_EVENT_HANDSHAKE_FAILED:
                on_event_handshake_failed(*event, address.c_str());
                break;
            case ZMQ_EVENT_HANDSHAKE_SUCCEED:
                on_event_handshake_succeed(*event, address.c_str());
                break;
#endif
#endif
            default:
                on_event_unknown(*event, address.c_str());
                break;
        }
        zmq_msg_close(&eventMsg);

        return true;
    }

#ifdef ZMQ_EVENT_MONITOR_STOPPED
    void abort()
    {
        if (_socket)
            zmq_socket_monitor(_socket.handle(), ZMQ_NULLPTR, 0);

        _socket = socket_ref();
    }
#endif
    virtual void on_monitor_started() {}
    virtual void on_event_connected(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_connect_delayed(const zmq_event_t &event_,
                                          const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_connect_retried(const zmq_event_t &event_,
                                          const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_listening(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_bind_failed(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_accepted(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_accept_failed(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_closed(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_close_failed(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_disconnected(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 3)
    virtual void on_event_handshake_failed_no_detail(const zmq_event_t &event_,
                                                     const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_handshake_failed_protocol(const zmq_event_t &event_,
                                                    const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_handshake_failed_auth(const zmq_event_t &event_,
                                                const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_handshake_succeeded(const zmq_event_t &event_,
                                              const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
#elif ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 1)
    virtual void on_event_handshake_failed(const zmq_event_t &event_,
                                           const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
    virtual void on_event_handshake_succeed(const zmq_event_t &event_,
                                            const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }
#endif
    virtual void on_event_unknown(const zmq_event_t &event_, const char *addr_)
    {
        (void) event_;
        (void) addr_;
    }

  private:
    monitor_t(const monitor_t &) ZMQ_DELETED_FUNCTION;
    void operator=(const monitor_t &) ZMQ_DELETED_FUNCTION;

    socket_ref _socket;
    socket_t _monitor_socket;

    void close() ZMQ_NOTHROW
    {
        if (_socket)
            zmq_socket_monitor(_socket.handle(), ZMQ_NULLPTR, 0);
        _monitor_socket.close();
    }
};

#if defined(ZMQ_BUILD_DRAFT_API) && defined(ZMQ_CPP11) && defined(ZMQ_HAVE_POLLER)

// polling events
enum class event_flags : short
{
    none = 0,
    pollin = ZMQ_POLLIN,
    pollout = ZMQ_POLLOUT,
    pollerr = ZMQ_POLLERR,
    pollpri = ZMQ_POLLPRI
};

constexpr event_flags operator|(event_flags a, event_flags b) noexcept
{
    return detail::enum_bit_or(a, b);
}
constexpr event_flags operator&(event_flags a, event_flags b) noexcept
{
    return detail::enum_bit_and(a, b);
}
constexpr event_flags operator^(event_flags a, event_flags b) noexcept
{
    return detail::enum_bit_xor(a, b);
}
constexpr event_flags operator~(event_flags a) noexcept
{
    return detail::enum_bit_not(a);
}

struct no_user_data;

// layout compatible with zmq_poller_event_t
template<class T = no_user_data> struct poller_event
{
    socket_ref socket;
#ifdef _WIN32
    SOCKET fd;
#else
    int fd;
#endif
    T *user_data;
    event_flags events;
};

template<typename T = no_user_data> class poller_t
{
  public:
    using event_type = poller_event<T>;

    poller_t() : poller_ptr(zmq_poller_new())
    {
        if (!poller_ptr)
            throw error_t();
    }

    template<
      typename Dummy = void,
      typename =
        typename std::enable_if<!std::is_same<T, no_user_data>::value, Dummy>::type>
    void add(zmq::socket_ref socket, event_flags events, T *user_data)
    {
        add_impl(socket, events, user_data);
    }

    void add(zmq::socket_ref socket, event_flags events)
    {
        add_impl(socket, events, nullptr);
    }

    void remove(zmq::socket_ref socket)
    {
        if (0 != zmq_poller_remove(poller_ptr.get(), socket.handle())) {
            throw error_t();
        }
    }

    void modify(zmq::socket_ref socket, event_flags events)
    {
        if (0
            != zmq_poller_modify(poller_ptr.get(), socket.handle(),
                                 static_cast<short>(events))) {
            throw error_t();
        }
    }

    size_t wait_all(std::vector<event_type> &poller_events,
                    const std::chrono::milliseconds timeout)
    {
        int rc = zmq_poller_wait_all(
          poller_ptr.get(),
          reinterpret_cast<zmq_poller_event_t *>(poller_events.data()),
          static_cast<int>(poller_events.size()),
          static_cast<long>(timeout.count()));
        if (rc > 0)
            return static_cast<size_t>(rc);

#if ZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 2, 3)
        if (zmq_errno() == EAGAIN)
#else
        if (zmq_errno() == ETIMEDOUT)
#endif
            return 0;

        throw error_t();
    }

  private:
    struct destroy_poller_t
    {
        void operator()(void *ptr) noexcept
        {
            int rc = zmq_poller_destroy(&ptr);
            ZMQ_ASSERT(rc == 0);
        }
    };

    std::unique_ptr<void, destroy_poller_t> poller_ptr;

    void add_impl(zmq::socket_ref socket, event_flags events, T *user_data)
    {
        if (0
            != zmq_poller_add(poller_ptr.get(), socket.handle(), user_data,
                              static_cast<short>(events))) {
            throw error_t();
        }
    }
};
#endif //  defined(ZMQ_BUILD_DRAFT_API) && defined(ZMQ_CPP11) && defined(ZMQ_HAVE_POLLER)

inline std::ostream &operator<<(std::ostream &os, const message_t &msg)
{
    return os << msg.str();
}

} // namespace zmq

#endif // __ZMQ_HPP_INCLUDED__


//========= end of #include "zmq.hpp" ============


//========= begin of #include "zmq_addon.hpp" ============

/*
    Copyright (c) 2016-2017 ZeroMQ community
    Copyright (c) 2016 VOCA AS / Harald Nøkland

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
*/

#ifndef __ZMQ_ADDON_HPP_INCLUDED__
#define __ZMQ_ADDON_HPP_INCLUDED__

// ans ignore: #include "zmq.hpp"

#include <deque>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#ifdef ZMQ_CPP11
#include <limits>
#include <functional>
#include <unordered_map>
#endif

namespace zmq
{
#ifdef ZMQ_CPP11

namespace detail
{
template<bool CheckN, class OutputIt>
recv_result_t
recv_multipart_n(socket_ref s, OutputIt out, size_t n, recv_flags flags)
{
    size_t msg_count = 0;
    message_t msg;
    while (true) {
        if (CheckN) {
            if (msg_count >= n)
                throw std::runtime_error(
                  "Too many message parts in recv_multipart_n");
        }
        if (!s.recv(msg, flags)) {
            // zmq ensures atomic delivery of messages
            assert(msg_count == 0);
            return {};
        }
        ++msg_count;
        const bool more = msg.more();
        *out++ = std::move(msg);
        if (!more)
            break;
    }
    return msg_count;
}

inline bool is_little_endian()
{
    const uint16_t i = 0x01;
    return *reinterpret_cast<const uint8_t *>(&i) == 0x01;
}

inline void write_network_order(unsigned char *buf, const uint32_t value)
{
    if (is_little_endian()) {
        ZMQ_CONSTEXPR_VAR uint32_t mask = std::numeric_limits<std::uint8_t>::max();
        *buf++ = (value >> 24) & mask;
        *buf++ = (value >> 16) & mask;
        *buf++ = (value >> 8) & mask;
        *buf++ = value & mask;
    } else {
        std::memcpy(buf, &value, sizeof(value));
    }
}

inline uint32_t read_u32_network_order(const unsigned char *buf)
{
    if (is_little_endian()) {
        return (static_cast<uint32_t>(buf[0]) << 24)
               + (static_cast<uint32_t>(buf[1]) << 16)
               + (static_cast<uint32_t>(buf[2]) << 8)
               + static_cast<uint32_t>(buf[3]);
    } else {
        uint32_t value;
        std::memcpy(&value, buf, sizeof(value));
        return value;
    }
}
} // namespace detail

/*  Receive a multipart message.
    
    Writes the zmq::message_t objects to OutputIterator out.
    The out iterator must handle an unspecified number of writes,
    e.g. by using std::back_inserter.
    
    Returns: the number of messages received or nullopt (on EAGAIN).
    Throws: if recv throws. Any exceptions thrown
    by the out iterator will be propagated and the message
    may have been only partially received with pending
    message parts. It is adviced to close this socket in that event.
*/
template<class OutputIt>
ZMQ_NODISCARD recv_result_t recv_multipart(socket_ref s,
                                           OutputIt out,
                                           recv_flags flags = recv_flags::none)
{
    return detail::recv_multipart_n<false>(s, std::move(out), 0, flags);
}

/*  Receive a multipart message.
    
    Writes at most n zmq::message_t objects to OutputIterator out.
    If the number of message parts of the incoming message exceeds n
    then an exception will be thrown.
    
    Returns: the number of messages received or nullopt (on EAGAIN).
    Throws: if recv throws. Throws std::runtime_error if the number
    of message parts exceeds n (exactly n messages will have been written
    to out). Any exceptions thrown
    by the out iterator will be propagated and the message
    may have been only partially received with pending
    message parts. It is adviced to close this socket in that event.
*/
template<class OutputIt>
ZMQ_NODISCARD recv_result_t recv_multipart_n(socket_ref s,
                                             OutputIt out,
                                             size_t n,
                                             recv_flags flags = recv_flags::none)
{
    return detail::recv_multipart_n<true>(s, std::move(out), n, flags);
}

/*  Send a multipart message.
    
    The range must be a ForwardRange of zmq::message_t,
    zmq::const_buffer or zmq::mutable_buffer.
    The flags may be zmq::send_flags::sndmore if there are 
    more message parts to be sent after the call to this function.
    
    Returns: the number of messages sent (exactly msgs.size()) or nullopt (on EAGAIN).
    Throws: if send throws. Any exceptions thrown
    by the msgs range will be propagated and the message
    may have been only partially sent. It is adviced to close this socket in that event.
*/
template<class Range
#ifndef ZMQ_CPP11_PARTIAL
         ,
         typename = typename std::enable_if<
           detail::is_range<Range>::value
           && (std::is_same<detail::range_value_t<Range>, message_t>::value
               || detail::is_buffer<detail::range_value_t<Range>>::value)>::type
#endif
         >
send_result_t
send_multipart(socket_ref s, Range &&msgs, send_flags flags = send_flags::none)
{
    using std::begin;
    using std::end;
    auto it = begin(msgs);
    const auto end_it = end(msgs);
    size_t msg_count = 0;
    while (it != end_it) {
        const auto next = std::next(it);
        const auto msg_flags =
          flags | (next == end_it ? send_flags::none : send_flags::sndmore);
        if (!s.send(*it, msg_flags)) {
            // zmq ensures atomic delivery of messages
            assert(it == begin(msgs));
            return {};
        }
        ++msg_count;
        it = next;
    }
    return msg_count;
}

/* Encode a multipart message.

   The range must be a ForwardRange of zmq::message_t.  A
   zmq::multipart_t or STL container may be passed for encoding.

   Returns: a zmq::message_t holding the encoded multipart data.

   Throws: std::range_error is thrown if the size of any single part
   can not fit in an unsigned 32 bit integer.

   The encoding is compatible with that used by the CZMQ function
   zmsg_encode(), see https://rfc.zeromq.org/spec/50/.
   Each part consists of a size followed by the data.
   These are placed contiguously into the output message.  A part of
   size less than 255 bytes will have a single byte size value.
   Larger parts will have a five byte size value with the first byte
   set to 0xFF and the remaining four bytes holding the size of the
   part's data.
*/
template<class Range
#ifndef ZMQ_CPP11_PARTIAL
         ,
         typename = typename std::enable_if<
           detail::is_range<Range>::value
           && (std::is_same<detail::range_value_t<Range>, message_t>::value
               || detail::is_buffer<detail::range_value_t<Range>>::value)>::type
#endif
         >
message_t encode(const Range &parts)
{
    size_t mmsg_size = 0;

    // First pass check sizes
    for (const auto &part : parts) {
        const size_t part_size = part.size();
        if (part_size > std::numeric_limits<std::uint32_t>::max()) {
            // Size value must fit into uint32_t.
            throw std::range_error("Invalid size, message part too large");
        }
        const size_t count_size =
          part_size < std::numeric_limits<std::uint8_t>::max() ? 1 : 5;
        mmsg_size += part_size + count_size;
    }

    message_t encoded(mmsg_size);
    unsigned char *buf = encoded.data<unsigned char>();
    for (const auto &part : parts) {
        const uint32_t part_size = part.size();
        const unsigned char *part_data =
          static_cast<const unsigned char *>(part.data());

        if (part_size < std::numeric_limits<std::uint8_t>::max()) {
            // small part
            *buf++ = (unsigned char) part_size;
        } else {
            // big part
            *buf++ = std::numeric_limits<uint8_t>::max();
            detail::write_network_order(buf, part_size);
            buf += sizeof(part_size);
        }
        std::memcpy(buf, part_data, part_size);
        buf += part_size;
    }

    assert(static_cast<size_t>(buf - encoded.data<unsigned char>()) == mmsg_size);
    return encoded;
}

/*  Decode an encoded message to multiple parts.

    The given output iterator must be a ForwardIterator to a container
    holding zmq::message_t such as a zmq::multipart_t or various STL
    containers.

    Returns the ForwardIterator advanced once past the last decoded
    part.

    Throws: a std::out_of_range is thrown if the encoded part sizes
    lead to exceeding the message data bounds.

    The decoding assumes the message is encoded in the manner
    performed by zmq::encode(), see https://rfc.zeromq.org/spec/50/.
 */
template<class OutputIt> OutputIt decode(const message_t &encoded, OutputIt out)
{
    const unsigned char *source = encoded.data<unsigned char>();
    const unsigned char *const limit = source + encoded.size();

    while (source < limit) {
        size_t part_size = *source++;
        if (part_size == std::numeric_limits<std::uint8_t>::max()) {
            if (static_cast<size_t>(limit - source) < sizeof(uint32_t)) {
                throw std::out_of_range(
                  "Malformed encoding, overflow in reading size");
            }
            part_size = detail::read_u32_network_order(source);
            // the part size is allowed to be less than 0xFF
            source += sizeof(uint32_t);
        }

        if (static_cast<size_t>(limit - source) < part_size) {
            throw std::out_of_range("Malformed encoding, overflow in reading part");
        }
        *out = message_t(source, part_size);
        ++out;
        source += part_size;
    }

    assert(source == limit);
    return out;
}

#endif


#ifdef ZMQ_HAS_RVALUE_REFS

/*
    This class handles multipart messaging. It is the C++ equivalent of zmsg.h,
    which is part of CZMQ (the high-level C binding). Furthermore, it is a major
    improvement compared to zmsg.hpp, which is part of the examples in the ØMQ
    Guide. Unnecessary copying is avoided by using move semantics to efficiently
    add/remove parts.
*/
class multipart_t
{
  private:
    std::deque<message_t> m_parts;

  public:
    typedef std::deque<message_t>::value_type value_type;

    typedef std::deque<message_t>::iterator iterator;
    typedef std::deque<message_t>::const_iterator const_iterator;

    typedef std::deque<message_t>::reverse_iterator reverse_iterator;
    typedef std::deque<message_t>::const_reverse_iterator const_reverse_iterator;

    // Default constructor
    multipart_t() {}

    // Construct from socket receive
    multipart_t(socket_t &socket) { recv(socket); }

    // Construct from memory block
    multipart_t(const void *src, size_t size) { addmem(src, size); }

    // Construct from string
    multipart_t(const std::string &string) { addstr(string); }

    // Construct from message part
    multipart_t(message_t &&message) { add(std::move(message)); }

    // Move constructor
    multipart_t(multipart_t &&other) { m_parts = std::move(other.m_parts); }

    // Move assignment operator
    multipart_t &operator=(multipart_t &&other)
    {
        m_parts = std::move(other.m_parts);
        return *this;
    }

    // Destructor
    virtual ~multipart_t() { clear(); }

    message_t &operator[](size_t n) { return m_parts[n]; }

    const message_t &operator[](size_t n) const { return m_parts[n]; }

    message_t &at(size_t n) { return m_parts.at(n); }

    const message_t &at(size_t n) const { return m_parts.at(n); }

    iterator begin() { return m_parts.begin(); }

    const_iterator begin() const { return m_parts.begin(); }

    const_iterator cbegin() const { return m_parts.cbegin(); }

    reverse_iterator rbegin() { return m_parts.rbegin(); }

    const_reverse_iterator rbegin() const { return m_parts.rbegin(); }

    iterator end() { return m_parts.end(); }

    const_iterator end() const { return m_parts.end(); }

    const_iterator cend() const { return m_parts.cend(); }

    reverse_iterator rend() { return m_parts.rend(); }

    const_reverse_iterator rend() const { return m_parts.rend(); }

    // Delete all parts
    void clear() { m_parts.clear(); }

    // Get number of parts
    size_t size() const { return m_parts.size(); }

    // Check if number of parts is zero
    bool empty() const { return m_parts.empty(); }

    // Receive multipart message from socket
    bool recv(socket_t &socket, int flags = 0)
    {
        clear();
        bool more = true;
        while (more) {
            message_t message;
#ifdef ZMQ_CPP11
            if (!socket.recv(message, static_cast<recv_flags>(flags)))
                return false;
#else
            if (!socket.recv(&message, flags))
                return false;
#endif
            more = message.more();
            add(std::move(message));
        }
        return true;
    }

    // Send multipart message to socket
    bool send(socket_t &socket, int flags = 0)
    {
        flags &= ~(ZMQ_SNDMORE);
        bool more = size() > 0;
        while (more) {
            message_t message = pop();
            more = size() > 0;
#ifdef ZMQ_CPP11
            if (!socket.send(message, static_cast<send_flags>(
                                        (more ? ZMQ_SNDMORE : 0) | flags)))
                return false;
#else
            if (!socket.send(message, (more ? ZMQ_SNDMORE : 0) | flags))
                return false;
#endif
        }
        clear();
        return true;
    }

    // Concatenate other multipart to front
    void prepend(multipart_t &&other)
    {
        while (!other.empty())
            push(other.remove());
    }

    // Concatenate other multipart to back
    void append(multipart_t &&other)
    {
        while (!other.empty())
            add(other.pop());
    }

    // Push memory block to front
    void pushmem(const void *src, size_t size)
    {
        m_parts.push_front(message_t(src, size));
    }

    // Push memory block to back
    void addmem(const void *src, size_t size)
    {
        m_parts.push_back(message_t(src, size));
    }

    // Push string to front
    void pushstr(const std::string &string)
    {
        m_parts.push_front(message_t(string.data(), string.size()));
    }

    // Push string to back
    void addstr(const std::string &string)
    {
        m_parts.push_back(message_t(string.data(), string.size()));
    }

    // Push type (fixed-size) to front
    template<typename T> void pushtyp(const T &type)
    {
        static_assert(!std::is_same<T, std::string>::value,
                      "Use pushstr() instead of pushtyp<std::string>()");
        m_parts.push_front(message_t(&type, sizeof(type)));
    }

    // Push type (fixed-size) to back
    template<typename T> void addtyp(const T &type)
    {
        static_assert(!std::is_same<T, std::string>::value,
                      "Use addstr() instead of addtyp<std::string>()");
        m_parts.push_back(message_t(&type, sizeof(type)));
    }

    // Push message part to front
    void push(message_t &&message) { m_parts.push_front(std::move(message)); }

    // Push message part to back
    void add(message_t &&message) { m_parts.push_back(std::move(message)); }

    // Alias to allow std::back_inserter()
    void push_back(message_t &&message) { m_parts.push_back(std::move(message)); }

    // Pop string from front
    std::string popstr()
    {
        std::string string(m_parts.front().data<char>(), m_parts.front().size());
        m_parts.pop_front();
        return string;
    }

    // Pop type (fixed-size) from front
    template<typename T> T poptyp()
    {
        static_assert(!std::is_same<T, std::string>::value,
                      "Use popstr() instead of poptyp<std::string>()");
        if (sizeof(T) != m_parts.front().size())
            throw std::runtime_error(
              "Invalid type, size does not match the message size");
        T type = *m_parts.front().data<T>();
        m_parts.pop_front();
        return type;
    }

    // Pop message part from front
    message_t pop()
    {
        message_t message = std::move(m_parts.front());
        m_parts.pop_front();
        return message;
    }

    // Pop message part from back
    message_t remove()
    {
        message_t message = std::move(m_parts.back());
        m_parts.pop_back();
        return message;
    }

    // get message part from front
    const message_t &front() { return m_parts.front(); }

    // get message part from back
    const message_t &back() { return m_parts.back(); }

    // Get pointer to a specific message part
    const message_t *peek(size_t index) const { return &m_parts[index]; }

    // Get a string copy of a specific message part
    std::string peekstr(size_t index) const
    {
        std::string string(m_parts[index].data<char>(), m_parts[index].size());
        return string;
    }

    // Peek type (fixed-size) from front
    template<typename T> T peektyp(size_t index) const
    {
        static_assert(!std::is_same<T, std::string>::value,
                      "Use peekstr() instead of peektyp<std::string>()");
        if (sizeof(T) != m_parts[index].size())
            throw std::runtime_error(
              "Invalid type, size does not match the message size");
        T type = *m_parts[index].data<T>();
        return type;
    }

    // Create multipart from type (fixed-size)
    template<typename T> static multipart_t create(const T &type)
    {
        multipart_t multipart;
        multipart.addtyp(type);
        return multipart;
    }

    // Copy multipart
    multipart_t clone() const
    {
        multipart_t multipart;
        for (size_t i = 0; i < size(); i++)
            multipart.addmem(m_parts[i].data(), m_parts[i].size());
        return multipart;
    }

    // Dump content to string
    std::string str() const
    {
        std::stringstream ss;
        for (size_t i = 0; i < m_parts.size(); i++) {
            const unsigned char *data = m_parts[i].data<unsigned char>();
            size_t size = m_parts[i].size();

            // Dump the message as text or binary
            bool isText = true;
            for (size_t j = 0; j < size; j++) {
                if (data[j] < 32 || data[j] > 127) {
                    isText = false;
                    break;
                }
            }
            ss << "\n[" << std::dec << std::setw(3) << std::setfill('0') << size
               << "] ";
            if (size >= 1000) {
                ss << "... (too big to print)";
                continue;
            }
            for (size_t j = 0; j < size; j++) {
                if (isText)
                    ss << static_cast<char>(data[j]);
                else
                    ss << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<short>(data[j]);
            }
        }
        return ss.str();
    }

    // Check if equal to other multipart
    bool equal(const multipart_t *other) const
    {
        if (size() != other->size())
            return false;
        for (size_t i = 0; i < size(); i++)
            if (*peek(i) != *other->peek(i))
                return false;
        return true;
    }

#ifdef ZMQ_CPP11

    // Return single part message_t encoded from this multipart_t.
    message_t encode() const { return zmq::encode(*this); }

    // Decode encoded message into multiple parts and append to self.
    void decode_append(const message_t &encoded)
    {
        zmq::decode(encoded, std::back_inserter(*this));
    }

    // Return a new multipart_t containing the decoded message_t.
    static multipart_t decode(const message_t &encoded)
    {
        multipart_t tmp;
        zmq::decode(encoded, std::back_inserter(tmp));
        return tmp;
    }

#endif

  private:
    // Disable implicit copying (moving is more efficient)
    multipart_t(const multipart_t &other) ZMQ_DELETED_FUNCTION;
    void operator=(const multipart_t &other) ZMQ_DELETED_FUNCTION;
}; // class multipart_t

inline std::ostream &operator<<(std::ostream &os, const multipart_t &msg)
{
    return os << msg.str();
}

#endif // ZMQ_HAS_RVALUE_REFS

#if defined(ZMQ_BUILD_DRAFT_API) && defined(ZMQ_CPP11) && defined(ZMQ_HAVE_POLLER)
class active_poller_t
{
  public:
    active_poller_t() = default;
    ~active_poller_t() = default;

    active_poller_t(const active_poller_t &) = delete;
    active_poller_t &operator=(const active_poller_t &) = delete;

    active_poller_t(active_poller_t &&src) = default;
    active_poller_t &operator=(active_poller_t &&src) = default;

    using handler_type = std::function<void(event_flags)>;

    void add(zmq::socket_ref socket, event_flags events, handler_type handler)
    {
        auto it = decltype(handlers)::iterator{};
        auto inserted = bool{};
        std::tie(it, inserted) = handlers.emplace(
          socket, std::make_shared<handler_type>(std::move(handler)));
        try {
            base_poller.add(socket, events,
                            inserted && *(it->second) ? it->second.get() : nullptr);
            need_rebuild |= inserted;
        }
        catch (const zmq::error_t &) {
            // rollback
            if (inserted) {
                handlers.erase(socket);
            }
            throw;
        }
    }

    void remove(zmq::socket_ref socket)
    {
        base_poller.remove(socket);
        handlers.erase(socket);
        need_rebuild = true;
    }

    void modify(zmq::socket_ref socket, event_flags events)
    {
        base_poller.modify(socket, events);
    }

    size_t wait(std::chrono::milliseconds timeout)
    {
        if (need_rebuild) {
            poller_events.resize(handlers.size());
            poller_handlers.clear();
            poller_handlers.reserve(handlers.size());
            for (const auto &handler : handlers) {
                poller_handlers.push_back(handler.second);
            }
            need_rebuild = false;
        }
        const auto count = base_poller.wait_all(poller_events, timeout);
        std::for_each(poller_events.begin(),
                      poller_events.begin() + static_cast<ptrdiff_t>(count),
                      [](decltype(base_poller)::event_type &event) {
                          if (event.user_data != nullptr)
                              (*event.user_data)(event.events);
                      });
        return count;
    }

    ZMQ_NODISCARD bool empty() const noexcept { return handlers.empty(); }

    size_t size() const noexcept { return handlers.size(); }

  private:
    bool need_rebuild{false};

    poller_t<handler_type> base_poller{};
    std::unordered_map<socket_ref, std::shared_ptr<handler_type>> handlers{};
    std::vector<decltype(base_poller)::event_type> poller_events{};
    std::vector<std::shared_ptr<handler_type>> poller_handlers{};
};     // class active_poller_t
#endif //  defined(ZMQ_BUILD_DRAFT_API) && defined(ZMQ_CPP11) && defined(ZMQ_HAVE_POLLER)


} // namespace zmq

#endif // __ZMQ_ADDON_HPP_INCLUDED__


//========= end of #include "zmq_addon.hpp" ============


//========= begin of #include "zmq_utils.h" ============

/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*  This file is deprecated, and all its functionality provided by zmq.h     */
/*  Note that -Wpedantic compilation requires GCC to avoid using its custom
    extensions such as #warning, hence the trick below. Also, pragmas for
    warnings or other messages are not standard, not portable, and not all
    compilers even have an equivalent concept.
    So in the worst case, this include file is treated as silently empty. */

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)               \
  || defined(_MSC_VER)
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcpp"
#pragma GCC diagnostic ignored "-Werror"
#pragma GCC diagnostic ignored "-Wall"
#endif
//#pragma message(                                                               
//  "Warning: zmq_utils.h is deprecated. All its functionality is provided by zmq.h.")
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
#endif


//========= end of #include "zmq_utils.h" ============


#endif  // __UNION_EMQ_HPP

